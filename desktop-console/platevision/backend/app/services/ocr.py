"""Reuse OCR sessions; retry uncertain single-line crops with text localization."""
from app.services.normalization import normalize_plate
import re
import math
import cv2
import numpy as np
from pathlib import Path
from app.core.config import settings


class BatchPlateRecognizer:
    """Fixed-shape GPU recognition: one inference for multiple plate crops.

    Keep the supplied OCR vocabulary and CTC decoder unchanged. Confidence is
    the recognizer score, not an independently calibrated probability.
    """
    def __init__(self, recognizer):
        import openvino as ov
        import rapidocr_onnxruntime
        self.recognizer=recognizer
        self.batch_size=settings.fast_ocr_batch_size
        core=ov.Core()
        path=Path(rapidocr_onnxruntime.__file__).parent/'models/ch_PP-OCRv4_rec_infer.onnx'
        self.sessions={}
        for size in sorted({1,min(4,self.batch_size),self.batch_size}):
            model=core.read_model(str(path))
            model.reshape({0:[size,3,48,320]})
            # Keep original CTC probabilities; reduce vocabulary on the device.
            from openvino import opset13 as ops
            top=ops.topk(model.get_results()[0].input_value(0),ops.constant(1,np.int32),axis=2,mode='max',sort='value',index_element_type='i32')
            model=ov.Model([top.output(0),top.output(1)],model.get_parameters())
            compiled=core.compile_model(model,settings.openvino_device,{'PERFORMANCE_HINT':'LATENCY'})
            request=compiled.create_infer_request()
            tensor=np.zeros((size,3,48,320),np.float32)
            for _ in range(3):request.infer({0:tensor},share_inputs=True)
            self.sessions[size]=(compiled,request,tensor)

    def recognize(self,crops):
        output=[]
        for start in range(0,len(crops),self.batch_size):
            batch=crops[start:start+self.batch_size]
            size=min(n for n in self.sessions if n>=len(batch))
            _,request,tensor=self.sessions[size]
            tensor.fill(0)
            for index,crop in enumerate(batch):
                tensor[index]=self.recognizer.resize_norm_img(crop,320/48)
            request.infer({0:tensor},share_inputs=True)
            result=self.recognizer.postprocess_op.decode(request.get_output_tensor(1).data[:len(batch),:,0],request.get_output_tensor(0).data[:len(batch),:,0],is_remove_duplicate=True)
            output.extend((str(text),float(score)) for text,score in result)
        return output

class PlateOCR:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1,
                               det_limit_side_len=320, det_limit_type='max')
        self.batch=None
        if settings.acceleration!='onnx':
            try:
                self.batch=BatchPlateRecognizer(self.engine.text_rec)
            except Exception:
                import logging
                logging.getLogger(__name__).warning('Batched GPU OCR unavailable; retaining full OCR.',exc_info=True)
        self.runtime=f'OpenVINO {settings.openvino_device} batch OCR' if self.batch else 'ONNX CPU OCR'

    def read_many(self,pairs):
        """Read every candidate, without cached images or a plate-count cutoff.

        Single-line plates use tight crops, then batch uncertain context retries.
        Stacked plates retain the full reader instead of pretending a
        guessed line split is an accurate result. Its extra cost is measured.
        """
        if not self.batch:
            return [self.read_plate(context,tight) for context,tight in pairs]
        # First recognize every single-line tight crop once. Only retry crops
        # that need context; detailed mode remains available for full text search.
        indices=[i for i,(_,tight) in enumerate(pairs) if tight.shape[1]/max(1,tight.shape[0])>=2.5]
        results=[('',0.) for _ in pairs]
        readings=self.batch.recognize([pairs[i][1] for i in indices])
        retry=[]
        for i,reading in zip(indices,readings):
            results[i]=reading
            if reading[1]<.88 or normalize_plate(reading[0])[2]!='valid':
                retry.append(i)
        alternatives=self.batch.recognize([pairs[i][0] for i in retry])
        for i,reading in zip(retry,alternatives):
            results[i]=max([results[i],reading],key=self.rank)
        for i,(context,tight) in enumerate(pairs):
            if i not in indices:
                results[i]=self.read_plate(context,tight)
        return results

    @staticmethod
    def join_rows(result):
        if not result:
            return '', 0.0
        # Stacked plates often include a small IND mark or a nearby manufacturer
        # badge. Use text geometry to exclude small marks and standalone words.
        heights=[(math.dist(row[0][0],row[0][3])+math.dist(row[0][1],row[0][2]))/2 for row in result]
        largest=max(heights)
        filtered=[]
        for row,height in zip(result,heights):
            text=re.sub(r'[^A-Z0-9]','',row[1].upper())
            if height < largest*.45 or not text:
                continue
            if text.isalpha() and len(text)>3:
                continue
            filtered.append(row)
        if not filtered:
            return '', 0.0
        # Group tokens into lines before sorting left-to-right. Sorting every
        # token by its y coordinate scrambles a sloping single-line plate.
        lines=[]
        for row in sorted(filtered,key=lambda r:sum(p[1] for p in r[0])/4):
            center=sum(p[1] for p in row[0])/4
            line=next((line for line in lines if abs(line[0]-center)<largest*.55),None)
            if line is None:
                lines.append([center,[row]])
            else:
                line[1].append(row)
        ordered=[row for _,line in lines for row in sorted(line,key=lambda r:min(p[0] for p in r[0]))]
        return ''.join(row[1] for row in ordered), sum(float(row[2]) for row in ordered)/len(ordered)

    @staticmethod
    def rank(reading):
        text,score=reading
        # Format can resolve a close reading, but cannot rescue low OCR quality.
        return (score>=.7 and normalize_plate(text)[2]=='valid', score)

    def read_plate(self,context,tight):
        reading=self.read(context)
        if reading[1]>=.94 and normalize_plate(reading[0])[2]=='valid':
            return reading
        return max([reading,self.read(tight)],key=self.rank)

    def read(self, crop):
        readings=[('',0.0)]
        if crop.shape[1] / max(1,crop.shape[0]) >= 2.5:
            result, _ = self.engine(crop, use_det=False, use_cls=False)
            if result:
                readings.append((str(result[0][0]),float(result[0][1])))
                if readings[-1][1]>=.94 and normalize_plate(readings[-1][0])[2]=='valid':
                    return readings[-1]
        result, _ = self.engine(crop, use_cls=False)
        readings.append(self.join_rows(result))
        best=max(readings,key=self.rank)
        if best[1]>=.9 and normalize_plate(best[0])[2]=='valid':
            return best
        # One bounded retry for small/low-contrast crops, keeping original text
        # as an alternative. Never synthesize a registration from the format.
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
        scale=min(3,max(1,96/max(1,gray.shape[0])))
        enhanced=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        enhanced=cv2.createCLAHE(clipLimit=2,tileGridSize=(4,4)).apply(enhanced)
        result,_=self.engine(enhanced,use_cls=False)
        readings.append(self.join_rows(result))
        return max(readings,key=self.rank)
