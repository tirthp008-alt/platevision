from abc import ABC, abstractmethod
import ast
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from app.core.config import settings


def detection_output_format(shape, classes=1, metadata=None):
    """Validate an export's class identity and choose its explicit box layout.

    Raw YOLO heads expose the class count in their channel dimension. Native
    end-to-end heads do not, so they must also carry a class-name vocabulary.
    This prevents a COCO YOLO26 export silently becoming a plate detector.
    """
    metadata = metadata or {}
    names = metadata.get('names')
    if isinstance(names, str):
        try:
            if len(names) > 20000:
                raise ValueError('Class metadata too large.')
            names = ast.literal_eval(names)
        except (SyntaxError, ValueError) as exc:
            raise RuntimeError('Invalid detector class-name metadata.') from exc
    if names is not None:
        if isinstance(names, dict):
            try:
                names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
            except KeyError as exc:
                raise RuntimeError('Detector classes must use contiguous zero-based IDs.') from exc
        if not isinstance(names, (list, tuple)) or len(names) != classes:
            raise RuntimeError(f'Expected {classes} detector classes; model metadata does not match.')
        if classes == 1 and 'plate' not in str(names[0]).lower():
            raise RuntimeError('The single-class detector must be trained for number plates.')
    if metadata.get('task', 'detect') != 'detect':
        raise RuntimeError('Expected an object detection model.')
    if len(shape) != 3 or (isinstance(shape[0], int) and shape[0] != 1):
        raise RuntimeError('Expected a batch-one YOLO detection output.')
    args = metadata.get('args', {})
    if isinstance(args, str):
        try:
            if len(args) > 20000:
                raise ValueError('Export metadata too large.')
            args = ast.literal_eval(args)
        except (SyntaxError, ValueError) as exc:
            raise RuntimeError('Invalid detector export metadata.') from exc
    if not isinstance(args, dict):
        raise RuntimeError('Expected detector export arguments to be a mapping.')
    def flag(name):
        value = metadata.get(name, args.get(name))
        if value in (True, 'True', 'true', '1', 1):
            return True
        if value in (False, 'False', 'false', '0', 0):
            return False
        return None
    end2end, nms = flag('end2end'), flag('nms')
    layout_hint = ('end_to_end' if end2end is True or nms is True else
                   'raw' if end2end is False and nms is False else None)
    raw_shape, end_shape = shape[1] == 4 + classes, shape[2] == 6
    if raw_shape and end_shape and layout_hint is None:
        raise RuntimeError('Ambiguous detection output; export end2end/nms metadata is required.')
    # Ultralytics' dynamic native export marks every output dimension symbolic
    # in ONNX even though OpenVINO can infer the six channels. Trust an explicit
    # layout hint only for unknown dimensions, then verify the real output at
    # warm-up. Known, contradictory dimensions must still fail immediately.
    raw_channels_known = isinstance(shape[1], (int, np.integer))
    end_channels_known = isinstance(shape[2], (int, np.integer))
    if ((layout_hint == 'end_to_end' and end_channels_known and not end_shape)
            or (layout_hint == 'raw' and raw_channels_known and not raw_shape)):
        raise RuntimeError('Detector output shape conflicts with its end2end/nms metadata.')
    if layout_hint == 'raw' or (raw_shape and layout_hint != 'end_to_end'):
        return 'raw'
    if layout_hint == 'end_to_end' or end_shape:
        if names is None:
            raise RuntimeError('End-to-end detector exports must include class-name metadata.')
        return 'end_to_end'
    raise RuntimeError('Unsupported output: expected raw [1, 4 + classes, N] or end-to-end [1, N, 6].')


class PlateDetector(ABC):
    @abstractmethod
    def detect(self, image): ...


class MockPlateDetector(PlateDetector):
    """Explicit demo/test mode only."""
    def detect(self, image):
        h, w = image.shape[:2]
        return [(int(w*.3), int(h*.42), int(w*.4), int(h*.16), .72)]


class OnnxPlateDetector(PlateDetector):
    """YOLO raw heads and native end-to-end FP32 detection exports."""
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
        if (len(shape) != 4 or shape[1] != 3 or self.input.type != 'tensor(float)'
                or (isinstance(shape[0], int) and shape[0] != 1)):
            raise RuntimeError('Expected FP32 NCHW RGB model input.')
        self.height = shape[2] if isinstance(shape[2], int) else settings.input_size
        self.width = shape[3] if isinstance(shape[3], int) else settings.input_size
        self.classes, self.allowed = classes, allowed
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise RuntimeError('Expected one YOLO detection output.')
        self.output_format = detection_output_format(outputs[0].shape, classes,
                                                     self.session.get_modelmeta().custom_metadata_map)
        output = self.session.run(None, {self.input.name: np.zeros((1,3,self.height,self.width), np.float32)})[0]
        # A warm-up also verifies symbolic output dimensions against real data.
        self.postprocess(output, 1, 0, 0, self.width, self.height)

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
        end_to_end = getattr(self, 'output_format', 'raw') == 'end_to_end'
        if output.ndim != 3 or output.shape[0] != 1:
            raise RuntimeError('Expected batch-one rank-three detection output.')
        if end_to_end:
            if output.shape[2] != 6:
                raise RuntimeError('Expected end-to-end [1, N, 6] xyxy/score/class output.')
            rows = output[0]
            scores, ids = rows[:,4], rows[:,5]
            mask = np.isfinite(rows).all(axis=1) & (ids >= 0) & (ids < self.classes) & (ids == np.floor(ids))
        else:
            if output.shape[1] != 4+self.classes:
                raise RuntimeError('Expected raw YOLO [1, 4 + classes, N] output.')
            rows = output[0].T
            ids = rows[:,4:].argmax(axis=1)
            scores = rows[np.arange(len(rows)),4+ids]
            mask = np.isfinite(rows).all(axis=1)
        mask &= (scores >= threshold) & (scores <= 1)
        if self.allowed is not None:
            mask &= np.isin(ids,list(self.allowed))
        boxes, confidences, class_ids = [], [], []
        for row, score, cls in zip(rows[mask],scores[mask],ids[mask]):
            if end_to_end:
                bx1,by1,bx2,by2 = row[:4]
            else:
                cx,cy,bw,bh = row[:4]
                bx1,by1,bx2,by2 = cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2
            x1 = max(0,min(width,int((bx1-left)/scale)))
            y1 = max(0,min(height,int((by1-top)/scale)))
            x2 = max(0,min(width,int((bx2-left)/scale)))
            y2 = max(0,min(height,int((by2-top)/scale)))
            if x2 <= x1 or y2 <= y1: continue
            boxes.append([x1,y1,x2-x1,y2-y1]); confidences.append(float(score)); class_ids.append(int(cls))
        if end_to_end:
            # YOLO26's exported head has already selected distinct detections.
            # Another NMS here would discard nearby/overlapping true plates.
            kept = [(*box, score, cls) for box,score,cls in zip(boxes,confidences,class_ids)]
            return sorted(kept,key=lambda item:item[4],reverse=True)[:settings.max_detections]
        kept=[]
        # Filtering the vocabulary must not switch to class-agnostic NMS.
        # Different vehicles can overlap heavily in dense traffic; suppress
        # repeated boxes only within the same predicted category.
        for cls in set(class_ids):
            group=[i for i,c in enumerate(class_ids) if c==cls]
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

    def detect_plates(self, image, refine=None, threshold=None):
        """Search every image region; vehicle detection is never a prerequisite."""
        height, width = image.shape[:2]
        size = settings.plate_tile_size
        refine = settings.plate_refinement_enabled if refine is None else refine
        final_threshold=settings.confidence_threshold if threshold is None else threshold
        threshold=min(.15,final_threshold) if refine else final_threshold
        candidates = list(self.detect(image,threshold=threshold))
        if not refine and max(width,height) <= size:
            return self.finalize_plate_candidates(candidates)
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
        return self.merge_plates([b for b in candidates if b[4]>=final_threshold])

    def detect_regions(self, image, profile='fast'):
        """Independent plate localization, with no OCR gate or vehicle prerequisite."""
        threshold=settings.region_confidence_threshold
        if profile == 'fast':
            models=getattr(self,'fast_models',None)
            if models:
                engine=models['landscape' if image.shape[1]>=image.shape[0] else 'portrait']
                boxes=engine.finalize_plate_candidates(engine.detect(image,threshold=threshold))
            else:
                boxes=self.detect_plates(image,refine=False,threshold=threshold)
        else:
            boxes=self.detect_plates(image,threshold=threshold)
        supplement=getattr(self,'region_supplement',None)
        if supplement is not None:
            # A second plate-trained model searches the whole scene too. No
            # ground-truth hints, plate colors, vehicle boxes or text are gates.
            boxes=self.merge_plates([*boxes,*supplement.detect(image,threshold=threshold)])
        return boxes

    def detect_vehicle_regions(self, image):
        """One plate-model pass on a vehicle crop, at the existing threshold.

        Use this detector's base input size. Whole-frame fast models, tiled
        searches, proposal refinements and OCR are intentionally not invoked.
        """
        return self.finalize_plate_candidates(
            self.detect(image, threshold=settings.region_confidence_threshold))

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

    def finalize_plate_candidates(self,candidates):
        """Finalize one inference pass; native heads already select detections.

        Bounds and classes are checked in postprocess. Retain the plate-shape
        guard and output limit here, without suppressing overlapping real
        plates from a native end-to-end head. Multi-pass tiled results still
        use merge_plates to reconcile observations of the same plate.
        """
        if getattr(self,'output_format','raw') != 'end_to_end':
            return self.merge_plates(candidates)
        kept=[box for box in candidates if np.isfinite(box).all()
              and box[0]>=0 and box[1]>=0 and box[2]>0 and box[3]>0
              and 1/10<=box[2]/box[3]<=10 and 0<=box[4]<=1]
        return sorted(kept,key=lambda b:b[4],reverse=True)[:settings.max_detections]

    @staticmethod
    def merge_plates(candidates):
        """Keep strong boxes; prefer complete stacked plates at similar confidence.

        A weak, oversized prediction must not swallow several nearby plates.
        Containment only merges similarly sized boxes at the same location.
        """
        kept=[]
        for box in sorted(candidates,key=lambda b:b[4],reverse=True):
            x,y,w,h,score=box
            # Supplied phone images also contain sideways plates and very slim
            # front scooter plates. Shape alone must not discard these learned
            # detections before the OCR/evidence checks can inspect them.
            if w<=0 or h<=0 or not 1/10<=w/h<=10:
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
    def __init__(self, path, device='GPU', input_shape=None, classes=1, allowed=None):
        import openvino as ov
        core = ov.Core()
        if device not in core.available_devices:
            raise RuntimeError(f'OpenVINO device {device} unavailable: {core.available_devices}')
        model = core.read_model(str(path))
        self.classes, self.allowed = classes, allowed
        if len(model.outputs) != 1:
            raise RuntimeError('Expected one YOLO detection output.')
        output_shape = [int(d.get_length()) if d.is_static else None for d in model.output(0).partial_shape]
        metadata = model.get_rt_info('framework').value if model.has_rt_info('framework') else {}
        self.output_format = detection_output_format(output_shape, self.classes, metadata)
        input_dimensions = model.input(0).partial_shape
        if (input_dimensions.rank.is_dynamic or input_dimensions.rank.get_length() != 4
                or input_dimensions[1] != 3 or model.input(0).element_type != ov.Type.f32
                or (input_dimensions[0].is_static and input_dimensions[0].get_length() != 1)):
            raise RuntimeError('Expected batch-one FP32 NCHW RGB model input.')
        if input_shape is not None or input_dimensions.is_dynamic:
            from app.services.model_shape import resize_plate_model
            model=resize_plate_model(model,*(input_shape or (settings.input_size,settings.input_size)))
        shape = list(model.input(0).shape)
        self.height, self.width = int(shape[2]), int(shape[3])
        # Fuse conversion, channel order and normalization into the device graph.
        prep = ov.preprocess.PrePostProcessor(model)
        prep.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout('NHWC')).set_color_format(ov.preprocess.ColorFormat.BGR)
        prep.input().model().set_layout(ov.Layout('NCHW'))
        prep.input().preprocess().convert_color(ov.preprocess.ColorFormat.RGB).convert_element_type(ov.Type.f32).scale(255.)
        self.compiled = core.compile_model(prep.build(), device, {'PERFORMANCE_HINT': 'LATENCY'})
        self.request = self.compiled.create_infer_request()
        self.runtime = f'OpenVINO {device}'
        for _ in range(3):
            output=self.infer(np.zeros((1,self.height,self.width,3), np.uint8))
        self.postprocess(output, 1, 0, 0, self.width, self.height)

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
        return detector.finalize_plate_candidates(detector.detect(image))


def describe_detector(detector):
    from hashlib import sha256
    detector.model_id=settings.detector_name
    detector.weights_retrained=settings.detector_retrained
    detector.model_sha256=sha256(Path(settings.model_path).read_bytes()).hexdigest()
    if settings.region_supplement_path:
        if isinstance(detector,OpenVINOPlateDetector):
            detector.region_supplement=OpenVINOPlateDetector(settings.region_supplement_path,settings.openvino_device)
        else:
            detector.region_supplement=OnnxPlateDetector(settings.region_supplement_path)
        detector.region_supplement_sha256=sha256(Path(settings.region_supplement_path).read_bytes()).hexdigest()
    return detector


class VehicleDetector:
    """A separate warmed vehicle model. Its vocabulary is never guessed from boxes."""
    def __init__(self, engine, taxonomy, names=None):
        self.engine, self.taxonomy, self.names=engine,taxonomy,names
        self.runtime=engine.runtime

    def objects(self, image):
        objects=self.engine.objects(image,threshold=settings.vehicle_confidence_threshold)
        if len(objects)>1:
            # Raw heads can label nearly the same vehicle as both bus/truck
            # (or another pair). Reconcile those duplicate observations only
            # at the vehicle boundary, with a much stricter overlap gate than
            # per-class NMS, so ordinary occlusion retains both vehicles.
            indices=cv2.dnn.NMSBoxes([list(box[:4]) for box in objects],
                                     [float(box[4]) for box in objects],0.,.90)
            objects=[objects[int(index)] for index in np.asarray(indices).reshape(-1)]
        if self.names is not None:
            return [(*box[:5],self.names[int(box[5])]) for box in objects]
        return objects


def get_vehicle_detector():
    if not settings.vehicle_context_enabled or not settings.vehicle_model_path:
        return None
    names=None
    if settings.vehicle_taxonomy == 'uvh':
        import onnx
        model=onnx.load(settings.vehicle_model_path,load_external_data=False)
        metadata={item.key:item.value for item in model.metadata_props}
        names=ast.literal_eval(metadata.get('names','{}'))
        if not isinstance(names,dict) or not names or set(names)!=set(range(len(names))):
            raise RuntimeError('Indian vehicle model needs contiguous class-name metadata.')
        normalized={str(name).lower().replace('-','').replace('_','').replace(' ','') for name in names.values()}
        if not {'twowheeler','threewheeler','tempotraveller','lcv'}.issubset(normalized):
            raise RuntimeError('Vehicle vocabulary does not match the UVH Indian-traffic model.')
        classes=len(names); allowed=set(range(classes))
    elif settings.vehicle_taxonomy == 'coco':
        classes=80; allowed={1,2,3,5,7}
    else:
        raise RuntimeError('Unsupported vehicle taxonomy.')
    if settings.acceleration != 'onnx':
        try:
            engine=OpenVINOPlateDetector(settings.vehicle_model_path,settings.openvino_device,
                                         classes=classes,allowed=allowed)
            return VehicleDetector(engine,settings.vehicle_taxonomy,names)
        except Exception:
            if settings.acceleration == 'openvino': raise
            import logging
            logging.getLogger(__name__).warning('Vehicle GPU runtime unavailable; using CPU.',exc_info=True)
    engine=OnnxPlateDetector(settings.vehicle_model_path,classes=classes,allowed=allowed)
    engine.runtime=f'ONNX {settings.onnx_provider}'
    return VehicleDetector(engine,settings.vehicle_taxonomy,names)


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
            return describe_detector(detector)
        except Exception:
            if settings.acceleration == 'openvino':
                raise
            import logging
            logging.getLogger(__name__).warning('OpenVINO unavailable; using ONNX CPU.', exc_info=True)
    detector = OnnxPlateDetector(settings.model_path)
    detector.runtime = f'ONNX {settings.onnx_provider}'
    return describe_detector(detector)
