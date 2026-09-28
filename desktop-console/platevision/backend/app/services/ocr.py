"""Reuse OCR sessions and retain evidence from bounded uncertain-crop retries."""
from app.services.normalization import normalize_plate
import re
import math
import time
import cv2
import numpy as np
from pathlib import Path
from app.core.config import settings


def preprocessing_candidates(crop):
    """Return bounded contrast/binary alternatives without changing the crop.

    Morphology is only useful when thresholding reveals isolated small specks.
    Its candidate is discarded if opening removes over 20% of foreground ink.
    These transforms are OCR inputs, never reconstructed registration strings.
    """
    if (not isinstance(crop, np.ndarray) or crop.dtype != np.uint8
            or crop.ndim not in (2, 3) or min(crop.shape[:2]) < 3
            or (crop.ndim == 3 and crop.shape[2] != 3)):
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    if float(gray.std()) < 2.:
        return []
    h, w = gray.shape
    scale = min(3., max(1., 96 / h), 640 / max(h, w))
    gray = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    enhanced = cv2.createCLAHE(clipLimit=2., tileGridSize=(4, 4)).apply(gray)
    candidates = [('grayscale_clahe', enhanced)]
    block = min(31, min(enhanced.shape))
    if block % 2 == 0:
        block -= 1
    if block >= 3:
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, 7)
        if float(binary.mean()) < 127:
            binary = 255 - binary
        candidates.append(('adaptive_threshold', binary))
        foreground = 255 - binary
        _, _, components, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        areas = components[1:, cv2.CC_STAT_AREA]
        specks = areas[areas <= 4]
        ink = int(np.count_nonzero(foreground))
        if len(specks) >= 8 and float(specks.sum()) >= .02 * max(1, ink):
            opened = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            remaining = int(np.count_nonzero(opened))
            if .8 * ink <= remaining < ink:
                candidates.append(('morphology_open', 255 - opened))
    return [(method, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)) for method, image in candidates]


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

    def __call__(self, images, return_word_box=False):
        if return_word_box:
            raise ValueError('Batched plate recognition does not return character boxes.')
        if isinstance(images, np.ndarray):
            images = [images]
        started = time.perf_counter()
        result = self.recognize(images)
        return result, time.perf_counter() - started

class PlateOCR:
    def __init__(self, backend='ppocr'):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1,
                               det_limit_side_len=320, det_limit_type='max')
        self.batch=None
        if backend == 'resnet34':
            from app.services.resnet_ocr import ResNetPlateRecognizer
            self.batch = ResNetPlateRecognizer(settings.resnet_ocr_model_path,
                                               settings.openvino_device, settings.fast_ocr_batch_size)
        elif backend != 'ppocr':
            raise ValueError(f'Unknown OCR model: {backend}')
        elif settings.acceleration!='onnx':
            try:
                self.batch=BatchPlateRecognizer(self.engine.text_rec)
            except Exception:
                import logging
                logging.getLogger(__name__).warning('Batched GPU OCR unavailable; retaining full OCR.',exc_info=True)
        name = 'ResNet34 + BiLSTM + CTC' if backend == 'resnet34' else 'PP-OCRv4'
        if self.batch:
            # Localized rows and contrast retries also use the warm GPU reader.
            # Keep text detection for stacked/uncertain plates, without falling
            # back to the slow CPU recognition session after every localization.
            self.engine.text_rec = self.batch
        self.runtime=f'OpenVINO {settings.openvino_device} batch {name}' if self.batch else f'ONNX CPU {name}'

    def read_many(self,pairs):
        return [(detail['text'], detail['confidence']) for detail in self.read_many_details(pairs)]

    def read_many_details(self, pairs, *, angle_correction=False):
        """Batch original crops first; retry only the uncertain observations."""
        if not getattr(self, 'batch', None):
            return [self.read_plate_details(context, tight, angle_correction=angle_correction)
                    for context, tight in pairs]
        indices = [i for i, (_, tight) in enumerate(pairs) if self._single_line(tight)]
        attempts = [[] for _ in pairs]
        context_retry_ms = [0. for _ in pairs]
        if indices:
            readings = self.batch.recognize([pairs[i][1] for i in indices])
            if len(readings) != len(indices):
                raise ValueError('OCR returned an incomplete plate batch.')
            for index, reading in zip(indices, readings):
                attempts[index].append(self._attempt('tight_original', reading))
        retry = [index for index in indices if self._selection(attempts[index])['requires_review']
                 and not self._same_crop(*pairs[index]) and self._single_line(pairs[index][0])]
        if retry:
            retry_started = time.perf_counter()
            try:
                readings = self.batch.recognize([pairs[i][0] for i in retry])
                if len(readings) != len(retry):
                    raise ValueError('OCR returned an incomplete retry batch.')
                for index, reading in zip(retry, readings):
                    attempts[index].append(self._attempt('context_original', reading))
            except Exception:
                for index in retry:
                    attempts[index].append(self._attempt('context_original', ('', 0.), error='unavailable'))
            # Attribute shared batch work equally; this is not a separate
            # sequential recognition duration for every affected plate.
            shared_ms = (time.perf_counter() - retry_started) * 1000 / len(retry)
            for index in retry:
                context_retry_ms[index] = shared_ms
        results = []
        for index, (context, tight) in enumerate(pairs):
            if not attempts[index]:
                results.append(self.read_plate_details(context, tight, angle_correction=angle_correction))
            else:
                results.append(self._finish_details(context, tight, attempts[index], angle_correction,
                                                    context_retry_ms=context_retry_ms[index]))
        return results

    @staticmethod
    def _single_line(crop):
        return crop.shape[1] / max(1, crop.shape[0]) >= 2.5

    @staticmethod
    def _same_crop(context, tight):
        return context is tight or (context.shape == tight.shape and np.array_equal(context, tight))

    @staticmethod
    def _attempt(method, reading, **extra):
        text, confidence = str(reading[0]), float(reading[1])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('OCR confidence must be finite and between zero and one.')
        return dict(method=method, text=text, confidence=confidence,
                    format_status=normalize_plate(text)[2], **extra)

    @staticmethod
    def _selection(attempts):
        # Agreement compares actual alphanumeric readings. In particular O/0
        # alternatives cannot become consensus via normalization replacements.
        groups = {}
        for attempt in attempts:
            key = re.sub('[^A-Z0-9]', '', attempt['text'].upper())
            if key:
                groups.setdefault(key, []).append(attempt)
        if not groups:
            selected = attempts[0] if attempts else dict(text='', confidence=0., method='none', format_status='uncertain')
            agreement = 0
            conflicts = []
        else:
            def rank(group):
                best = max(group, key=lambda attempt: attempt['confidence'])
                credible = sum(attempt['confidence'] >= .7 for attempt in group)
                return (best['format_status'] == 'valid' and best['confidence'] >= .7,
                        min(2, credible), best['confidence'])
            group = max(groups.values(), key=rank)
            selected = max(group, key=lambda attempt: attempt['confidence'])
            agreement = sum(attempt['confidence'] >= .7 for attempt in group)
            plausible = [key for key, items in groups.items() if any(
                item['format_status'] == 'valid' and item['confidence'] >= .7 for item in items)]
            conflicts = plausible if len(plausible) > 1 else []
        transformed = selected['method'].startswith(('rectified', 'grayscale_', 'adaptive_', 'morphology_'))
        review = (selected['confidence'] < .88 or selected['format_status'] != 'valid'
                  or bool(conflicts) or (transformed and agreement < 2))
        return dict(text=selected['text'], confidence=selected['confidence'], selected_method=selected['method'],
                    agreeing_attempts=agreement, requires_review=bool(review), conflicting_readings=conflicts)

    def _recognize_crop(self, crop, *, localized=False):
        if not localized and self._single_line(crop):
            if getattr(self, 'batch', None):
                return self.batch.recognize([crop])[0]
            result, _ = self.engine(crop, use_det=False, use_cls=False)
            return (str(result[0][0]), float(result[0][1])) if result else ('', 0.)
        result, _ = self.engine(crop, use_cls=False)
        return self.join_rows(result)

    def _retry(self, crop, method, attempts, *, localized=False):
        try:
            attempts.append(self._attempt(method, self._recognize_crop(crop, localized=localized)))
        except Exception:
            # Filtering/deskewing is best effort; an optional retry cannot
            # erase a successful original observation or its model score.
            attempts.append(self._attempt(method, ('', 0.), error='unavailable'))

    def read_plate_details(self, context, tight, *, initial_reading=None, angle_correction=False):
        reading = initial_reading if initial_reading is not None else self._recognize_crop(tight)
        attempts = [self._attempt('tight_original', reading)]
        context_retry_ms = 0.
        if self._selection(attempts)['requires_review'] and not self._same_crop(context, tight):
            retry_started = time.perf_counter()
            self._retry(context, 'context_original', attempts)
            context_retry_ms = (time.perf_counter() - retry_started) * 1000
        return self._finish_details(context, tight, attempts, angle_correction,
                                    context_retry_ms=context_retry_ms)

    def _finish_details(self, context, tight, attempts, angle_correction, *, context_retry_ms=0.):
        started = time.perf_counter()
        preprocessing_ms = 0.
        rectification = None
        corrected = None
        selection = self._selection(attempts)
        if selection['requires_review'] and self._single_line(tight) and hasattr(self, 'engine'):
            self._retry(context, 'context_localized', attempts, localized=True)
            selection = self._selection(attempts)
        if selection['requires_review'] and angle_correction:
            from app.services.rectification import rectify_plate
            tick = time.perf_counter()
            try:
                candidate, rectification = rectify_plate(context)
            except Exception:
                candidate, rectification = None, dict(applied=False, reason='unavailable')
            preprocessing_ms += (time.perf_counter() - tick) * 1000
            if rectification['applied']:
                corrected = candidate
                self._retry(corrected, 'rectified_original', attempts)
                selection = self._selection(attempts)
        if selection['requires_review']:
            tick = time.perf_counter()
            try:
                variants = preprocessing_candidates(corrected if corrected is not None else tight)
            except Exception:
                variants = []
            preprocessing_ms += (time.perf_counter() - tick) * 1000
            for method, crop in variants:
                self._retry(crop, method, attempts)
                attempts[-1]['source'] = 'rectified' if corrected is not None else 'tight'
                selection = self._selection(attempts)
                if not selection['requires_review']:
                    break
        result = dict(selection, original_text=attempts[0]['text'],
                      original_confidence=attempts[0]['confidence'], attempts=attempts,
                      retry_count=max(0, len(attempts) - 1),
                      preprocessing_time_ms=round(preprocessing_ms, 3),
                      retry_time_ms=round(context_retry_ms + (time.perf_counter() - started) * 1000, 3),
                      retry_timing_note='Includes context retry; shared context-batch time is divided equally among its plates.',
                      note='Scores are actual OCR model scores, not calibrated accuracy. Filter agreement is not independent verification.')
        if rectification is not None:
            result['rectification'] = rectification
            result['_rectification_attempt'] = dict(source='context', transform='rectify_plate')
        if corrected is not None:
            result['rectified_crop'] = corrected
        return result

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
        detail = self.read_plate_details(context, tight)
        return detail['text'], detail['confidence']

    def read(self, crop):
        return self.read_plate(crop, crop)
