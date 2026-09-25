from abc import ABC, abstractmethod
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from app.core.config import settings


class PlateDetector(ABC):
    @abstractmethod
    def detect(self, image): ...


class MockPlateDetector(PlateDetector):
    """Explicit demo/test mode only."""
    def detect(self, image):
        h, w = image.shape[:2]
        return [(int(w*.3), int(h*.42), int(w*.4), int(h*.16), .72)]


class OnnxPlateDetector(PlateDetector):
    """YOLOv8/11 raw FP32 detect exports: [1, 4 + classes, anchors]."""
    def __init__(self, path, classes=1, allowed=None):
        if not Path(path).is_file():
            raise RuntimeError(f'Model not found: {path}')
        options = ort.SessionOptions()
        options.intra_op_num_threads = settings.inference_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        requested = settings.onnx_provider
        if requested not in ort.get_available_providers():
            raise RuntimeError(f'ONNX provider unavailable: {requested}')
        self.session = ort.InferenceSession(path, sess_options=options, providers=[requested, 'CPUExecutionProvider'] if requested != 'CPUExecutionProvider' else [requested])
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        if len(shape) != 4 or shape[1] != 3 or self.input.type != 'tensor(float)':
            raise RuntimeError('Expected FP32 NCHW RGB model input.')
        self.height = shape[2] if isinstance(shape[2], int) else settings.input_size
        self.width = shape[3] if isinstance(shape[3], int) else settings.input_size
        self.classes, self.allowed = classes, allowed
        self.session.run(None, {self.input.name: np.zeros((1,3,self.height,self.width), np.float32)})

    def objects(self, image, threshold=None):
        h, w = image.shape[:2]
        scale = min(self.width/w, self.height/h)
        rw, rh = round(w*scale), round(h*scale)
        left, top = (self.width-rw)//2, (self.height-rh)//2
        padded = np.full((self.height,self.width,3), 114, np.uint8)
        padded[top:top+rh,left:left+rw] = cv2.resize(image,(rw,rh))
        tensor = self.tensor(padded)
        output = self.infer(tensor)
        return self.postprocess(output, scale, left, top, w, h, threshold)

    def infer(self, tensor):
        return self.session.run(None,{self.input.name:tensor})[0]

    def tensor(self, padded):
        return np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255

    def postprocess(self, output, scale, left, top, width, height, threshold=None):
        threshold=settings.confidence_threshold if threshold is None else threshold
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 4+self.classes:
            raise RuntimeError('Unsupported model output; export YOLOv8/11 detect with nms=False, batch=1.')
        rows = output[0].T
        ids = rows[:,4:].argmax(axis=1)
        scores = rows[np.arange(len(rows)),4+ids]
        mask = np.isfinite(rows).all(axis=1) & (scores >= threshold)
        if self.allowed is not None:
            mask &= np.isin(ids,list(self.allowed))
        boxes, confidences, class_ids = [], [], []
        for row, score, cls in zip(rows[mask],scores[mask],ids[mask]):
            cx,cy,bw,bh = row[:4]
            x1 = max(0,min(width,int((cx-bw/2-left)/scale)))
            y1 = max(0,min(height,int((cy-bh/2-top)/scale)))
            x2 = max(0,min(width,int((cx+bw/2-left)/scale)))
            y2 = max(0,min(height,int((cy+bh/2-top)/scale)))
            if x2 <= x1 or y2 <= y1: continue
            boxes.append([x1,y1,x2-x1,y2-y1]); confidences.append(float(score)); class_ids.append(int(cls))
        kept=[]
        for cls in ([None] if self.allowed is not None else set(class_ids)):
            group=[i for i,c in enumerate(class_ids) if cls is None or c==cls]
            indices=cv2.dnn.NMSBoxes([boxes[i] for i in group],[confidences[i] for i in group],threshold,.45)
            for index in np.asarray(indices).reshape(-1):
                i=group[int(index)]; kept.append((*boxes[i],confidences[i],class_ids[i]))
        return sorted(kept,key=lambda item:item[4],reverse=True)[:settings.max_detections]

    def detect(self, image, threshold=None):
        return [item[:5] for item in self.objects(image, threshold)]

    @staticmethod
    def tile_starts(length, size, overlap):
        if length <= size:
            return [0]
        stride = max(1, round(size*(1-overlap)))
        starts = list(range(0, length-size+1, stride))
        if starts[-1] != length-size:
            starts.append(length-size)
        return starts

    def detect_plates(self, image, refine=None):
        """Search every image region; vehicle detection is never a prerequisite."""
        height, width = image.shape[:2]
        size = settings.plate_tile_size
        refine = settings.plate_refinement_enabled if refine is None else refine
        threshold=min(.15,settings.confidence_threshold) if refine else settings.confidence_threshold
        candidates = list(self.detect(image,threshold=threshold))
        if max(width, height) > size:
            for top in self.tile_starts(height,size,settings.plate_tile_overlap):
                for left in self.tile_starts(width,size,settings.plate_tile_overlap):
                    tile = image[top:min(top+size,height), left:min(left+size,width)]
                    th, tw = tile.shape[:2]
                    for x,y,w,h,score in self.detect(tile,threshold=threshold):
                        # A cut-off fragment will be seen intact by the overlapping
                        # neighbour. Retain plates at the actual image boundary.
                        if ((left>0 and x<=2) or (top>0 and y<=2)
                            or (left+tw<width and x+w>=tw-2)
                            or (top+th<height and y+h>=th-2)):
                            continue
                        candidates.append((x+left,y+top,w,h,score))
        if refine:
            candidates.extend(self.refine_plates(image,self.merge_plates(candidates)))
        return self.merge_plates([b for b in candidates if b[4]>=settings.confidence_threshold])

    def refine_plates(self,image,proposals):
        """Zoom small, uncertain plate proposals. No vehicle detector is used."""
        height,width=image.shape[:2]
        refined=[]
        weak=[b for b in proposals if b[4]<.65 and max(b[2:4])<=180]
        for x,y,w,h,_ in weak[:settings.max_plate_refinements]:
            size=max(192,round(max(w,h)*4))
            left=max(0,min(width-size,round(x+w/2-size/2)))
            top=max(0,min(height-size,round(y+h/2-size/2)))
            for bx,by,bw,bh,score in self.detect(image[top:top+size,left:left+size]):
                ix=max(0,min(bx+bw+left,x+w)-max(bx+left,x))
                iy=max(0,min(by+bh+top,y+h)-max(by+top,y))
                if ix*iy/max(1,min(w*h,bw*bh))>=.5:
                    refined.append((bx+left,by+top,bw,bh,score))
        return refined

    @staticmethod
    def merge_plates(candidates):
        """Keep strong boxes; prefer complete stacked plates at similar confidence.

        A weak, oversized prediction must not swallow several nearby plates.
        Containment only merges similarly sized boxes at the same location.
        """
        kept=[]
        for box in sorted(candidates,key=lambda b:b[4],reverse=True):
            x,y,w,h,score=box
            if w<=0 or h<=0 or not .65<=w/h<=7:
                continue
            duplicate=False
            for index,(bx,by,bw,bh,bscore) in enumerate(kept):
                area=max(0,min(x+w,bx+bw)-max(x,bx))*max(0,min(y+h,by+bh)-max(y,by))
                contained=area/max(1,min(w*h,bw*bh))>.8
                ratio=min(w*h,bw*bh)/max(1,max(w*h,bw*bh))
                if area/max(1,w*h+bw*bh-area)>.4 or (contained and ratio>=.4):
                    duplicate=True
                    if contained and w*h>bw*bh and score+.10+1e-6>=bscore:
                        kept[index]=box
                    break
            if not duplicate:
                kept.append(box)
        return sorted(kept,key=lambda b:b[4],reverse=True)[:settings.max_detections]



class OpenVINOPlateDetector(OnnxPlateDetector):
    """Run the unchanged ONNX weights on Intel hardware; reuse all box logic."""
    def __init__(self, path, device='GPU', input_shape=None):
        import openvino as ov
        core = ov.Core()
        if device not in core.available_devices:
            raise RuntimeError(f'OpenVINO device {device} unavailable: {core.available_devices}')
        model = core.read_model(str(path))
        if input_shape is not None:
            from app.services.model_shape import resize_plate_model
            model=resize_plate_model(model,*input_shape)
        shape = list(model.input(0).shape)
        self.height, self.width = int(shape[2]), int(shape[3])
        # Fuse conversion, channel order and normalization into the device graph.
        prep = ov.preprocess.PrePostProcessor(model)
        prep.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout('NHWC')).set_color_format(ov.preprocess.ColorFormat.BGR)
        prep.input().model().set_layout(ov.Layout('NCHW'))
        prep.input().preprocess().convert_color(ov.preprocess.ColorFormat.RGB).convert_element_type(ov.Type.f32).scale(255.)
        self.compiled = core.compile_model(prep.build(), device, {'PERFORMANCE_HINT': 'LATENCY'})
        self.request = self.compiled.create_infer_request()
        self.classes, self.allowed = 1, None
        self.runtime = f'OpenVINO {device}'
        for _ in range(3):
            self.infer(np.zeros((1,self.height,self.width,3), np.uint8))

    def tensor(self, padded):
        return padded[None]

    def infer(self, tensor):
        self.request.infer({0: tensor}, share_inputs=True)
        return self.request.get_output_tensor(0).data

    def detect_fast(self,image):
        models=getattr(self,'fast_models',None)
        if not models:
            return self.detect_plates(image,refine=False)
        detector=models['landscape' if image.shape[1]>=image.shape[0] else 'portrait']
        return self.merge_plates(detector.detect(image))


def get_detector():
    if settings.detector_mode == 'mock': return MockPlateDetector()
    if settings.detector_mode != 'onnx': raise RuntimeError('DETECTOR_MODE must be onnx or mock.')
    if settings.acceleration != 'onnx':
        try:
            detector=OpenVINOPlateDetector(settings.model_path, settings.openvino_device)
            try:
                detector.fast_models={
                    'landscape':OpenVINOPlateDetector(settings.model_path,settings.openvino_device,(864,1536)),
                    'portrait':OpenVINOPlateDetector(settings.model_path,settings.openvino_device,(1536,864)),
                }
            except Exception:
                import logging
                logging.getLogger(__name__).warning('High-resolution detector unavailable; retaining tiled search.',exc_info=True)
            return detector
        except Exception:
            if settings.acceleration == 'openvino':
                raise
            import logging
            logging.getLogger(__name__).warning('OpenVINO unavailable; using ONNX CPU.', exc_info=True)
    detector = OnnxPlateDetector(settings.model_path)
    detector.runtime = f'ONNX {settings.onnx_provider}'
    return detector
