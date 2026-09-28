"""Optional, pretrained PaddleOCR ResNet34 + BiLSTM + CTC recognizer.

This reads plate crops; the YOLO plate detector still searches the full scene.
Weights and vocabulary come from scripts/setup_resnet_ocr.py, never from an
ImageNet classifier. No network access or model training happens at startup.
"""
import math
import time
from pathlib import Path

import cv2
import numpy as np


class ResNetPlateRecognizer:
    def __init__(self, model_path, device='GPU', batch_size=16):
        import openvino as ov
        from openvino import opset13 as ops
        from rapidocr_onnxruntime.ch_ppocr_rec.utils import CTCLabelDecode

        path = Path(model_path)
        self.postprocess_op = CTCLabelDecode(character_path=path.with_name('ppocr_keys_v1.txt'))
        self.batch_size = batch_size
        self.sessions = {}
        core = ov.Core()
        for size in sorted({1, min(4, batch_size), batch_size}):
            model = core.read_model(str(path))
            model.reshape({0: [size, 3, 32, 320]})
            if model.output().shape[-1] != len(self.postprocess_op.character):
                raise ValueError('ResNet OCR vocabulary does not match the model output.')
            # Transfer only the winning character and its original probability.
            top = ops.topk(model.get_results()[0].input_value(0), ops.constant(1, np.int32),
                           axis=2, mode='max', sort='value', index_element_type='i32')
            model = ov.Model([top.output(0), top.output(1)], model.get_parameters())
            compiled = core.compile_model(model, device, {'PERFORMANCE_HINT': 'LATENCY'})
            request = compiled.create_infer_request()
            tensor = np.zeros((size, 3, 32, 320), np.float32)
            for _ in range(3):
                request.infer({0: tensor}, share_inputs=True)
            self.sessions[size] = (compiled, request, tensor)

    @staticmethod
    def preprocess(crop):
        if crop.size == 0:
            raise ValueError('Cannot recognize an empty plate crop.')
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        width = max(1, min(320, math.ceil(32 * crop.shape[1] / crop.shape[0])))
        image = cv2.resize(crop, (width, 32)).astype(np.float32)
        tensor = np.zeros((3, 32, 320), np.float32)
        # Preserve PaddleOCR's BGR, [-1,1], zero-padding convention.
        tensor[:, :, :width] = (image.transpose(2, 0, 1) / 255 - .5) / .5
        return tensor

    def recognize(self, crops):
        result = []
        for start in range(0, len(crops), self.batch_size):
            batch = crops[start:start + self.batch_size]
            size = min(n for n in self.sessions if n >= len(batch))
            _, request, tensor = self.sessions[size]
            tensor.fill(0)
            for index, crop in enumerate(batch):
                tensor[index] = self.preprocess(crop)
            request.infer({0: tensor}, share_inputs=True)
            readings = self.postprocess_op.decode(
                request.get_output_tensor(1).data[:len(batch), :, 0],
                request.get_output_tensor(0).data[:len(batch), :, 0],
                is_remove_duplicate=True)
            result.extend((str(text), float(score)) for text, score in readings)
        return result

    def __call__(self, images, return_word_box=False):
        # RapidOCR supplies localized text rows for stacked plates in this path.
        if return_word_box:
            raise ValueError('ResNet plate recognition does not return character boxes.')
        if isinstance(images, np.ndarray):
            images = [images]
        started = time.perf_counter()
        result = self.recognize(images)
        return result, time.perf_counter() - started
