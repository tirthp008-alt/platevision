"""Official VeRi FastReID embeddings; similarity is not an identity probability."""
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image


def prepare_crop(crop):
    """Official test transform: BGR->RGB, bicubic 256x256, raw 0..255 FP32."""
    if not isinstance(crop, np.ndarray) or crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3 or min(crop.shape[:2]) < 2:
        raise ValueError('Re-ID requires a nonempty uint8 BGR vehicle crop.')
    rgb = Image.fromarray(crop[:, :, ::-1]).resize((256, 256), Image.Resampling.BICUBIC)
    return np.ascontiguousarray(np.asarray(rgb).transpose(2, 0, 1)[None], dtype=np.float32)


class VehicleReID:
    dimension = 2048
    model_id = 'FastReID SBS R50-IBN (VeRi)'

    def __init__(self, path, device='GPU', *, _allow_unverified=False):
        import openvino as ov
        self.path = Path(path)
        if not self.path.is_file():
            raise RuntimeError('Vehicle Re-ID export is missing; run scripts/setup_vehicle_reid.py.')
        card_path = self.path.parent / 'model-card.json'
        if not card_path.is_file():
            raise RuntimeError('Vehicle Re-ID model card is missing; re-run verified setup.')
        card = json.loads(card_path.read_text(encoding='utf-8'))
        if card.get('parity_status') != 'passed' and not _allow_unverified:
            raise RuntimeError('Vehicle Re-ID export parity is not verified; run setup --verify-runtime.')
        self.model_sha256 = sha256(self.path.read_bytes()).hexdigest()
        if self.model_sha256 != card.get('onnx_sha256'):
            raise RuntimeError('Vehicle Re-ID export hash does not match its model card.')
        core = ov.Core()
        if device not in core.available_devices:
            raise RuntimeError(f'Re-ID device {device} unavailable: {core.available_devices}')
        model = core.read_model(str(self.path))
        if len(model.inputs) != 1 or len(model.outputs) != 1 or list(model.input(0).shape) != [1, 3, 256, 256] or list(model.output(0).shape) != [1, 2048]:
            raise RuntimeError('Expected verified FastReID [1,3,256,256] -> [1,2048] export.')
        self.compiled = core.compile_model(model, device, {'PERFORMANCE_HINT': 'LATENCY'})
        self.request = self.compiled.create_infer_request()
        self.lock = Lock()
        self.runtime = f'OpenVINO {device}'
        self.metadata = dict(model=self.model_id, runtime=self.runtime, dimension=2048,
                             model_sha256=self.model_sha256, checkpoint_sha256=card['checkpoint_sha256'],
                             training_dataset='VeRi', source_revision=card['source_revision'],
                             similarity_note='Cosine appearance similarity, not verified identity or calibrated probability.')
        self.embed([np.zeros((256, 256, 3), np.uint8)])

    def embed(self, crops_bgr):
        if len(crops_bgr) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = []
        for crop in crops_bgr:
            tensor = prepare_crop(crop)
            with self.lock:
                self.request.infer({0: tensor}, share_inputs=True)
                vector = np.array(self.request.get_output_tensor(0).data, dtype=np.float32, copy=True).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if vector.shape != (self.dimension,) or not np.isfinite(vector).all() or norm < 1e-8:
                raise RuntimeError('Vehicle Re-ID returned an invalid embedding.')
            embeddings.append(vector / norm)
        return np.stack(embeddings).astype(np.float32, copy=False)
