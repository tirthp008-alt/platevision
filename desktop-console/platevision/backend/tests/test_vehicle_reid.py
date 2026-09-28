from hashlib import sha256
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.vehicle_reid import VehicleReID, prepare_crop


def test_preprocessing_uses_official_rgb_range_and_never_mutates_source():
    crop=np.full((60,100,3),[10,20,30],np.uint8)
    original=crop.copy()
    tensor=prepare_crop(crop)
    assert tensor.shape==(1,3,256,256) and tensor.dtype==np.float32
    assert tensor.flags.c_contiguous
    assert tensor[0,:,0,0].tolist()==[30.,20.,10.]
    assert np.array_equal(crop,original)


@pytest.mark.parametrize('crop',[np.zeros((0,3,3),np.uint8),np.zeros((10,10),np.uint8),
                                np.zeros((10,10,4),np.uint8),np.zeros((10,10,3),np.float32)])
def test_invalid_crop_cannot_silently_create_an_embedding(crop):
    with pytest.raises(ValueError,match='BGR'):
        prepare_crop(crop)


@pytest.fixture
def fake_runtime(tmp_path,monkeypatch):
    path=tmp_path/'model.onnx';path.write_bytes(b'unit-test-placeholder')
    card=dict(onnx_sha256=sha256(path.read_bytes()).hexdigest(),checkpoint_sha256='checkpoint',
              source_revision='source',parity_status='passed')
    (tmp_path/'model-card.json').write_text(json.dumps(card))
    class Request:
        def __init__(self):
            self.output=np.zeros((1,2048),np.float32)
        def infer(self,inputs,share_inputs=True):
            self.output.fill(0)
            # A deterministic fake network lets us test channel handling and
            # output normalization without loading any weights or GPU runtime.
            self.output[0,:3]=np.mean(inputs[0],axis=(0,2,3))+1
        def get_output_tensor(self,index):
            return SimpleNamespace(data=self.output)
    request=Request()
    compiled=SimpleNamespace(create_infer_request=lambda:request)
    model=SimpleNamespace(inputs=[1],outputs=[1],input=lambda index:SimpleNamespace(shape=[1,3,256,256]),
                          output=lambda index:SimpleNamespace(shape=[1,2048]))
    core=SimpleNamespace(available_devices=['GPU'],read_model=lambda path:model,
                         compile_model=lambda *args:compiled)
    monkeypatch.setitem(sys.modules,'openvino',SimpleNamespace(Core=lambda:core))
    return path,request


def test_runtime_returns_distinct_unit_vectors_and_empty_batch(fake_runtime):
    path,_=fake_runtime
    model=VehicleReID(path)
    assert model.embed([]).shape==(0,2048)
    outputs=model.embed([np.full((20,30,3),[10,20,30],np.uint8),
                         np.full((20,30,3),[30,20,10],np.uint8)])
    assert outputs.shape==(2,2048) and outputs.dtype==np.float32
    np.testing.assert_allclose(np.linalg.norm(outputs,axis=1),[1,1],atol=1e-6)
    assert not np.array_equal(outputs[0],outputs[1])
    assert model.metadata['training_dataset']=='VeRi'


def test_runtime_rejects_tampered_or_unverified_weights(fake_runtime):
    path,_=fake_runtime
    card_path=path.parent/'model-card.json'
    card=json.loads(card_path.read_text());card['parity_status']='awaiting_runtime_verification'
    card_path.write_text(json.dumps(card))
    with pytest.raises(RuntimeError,match='parity'):
        VehicleReID(path)
    card['parity_status']='passed';card_path.write_text(json.dumps(card))
    path.write_bytes(b'changed weights')
    with pytest.raises(RuntimeError,match='hash'):
        VehicleReID(path)


def test_runtime_rejects_nonfinite_embedding(fake_runtime,monkeypatch):
    path,request=fake_runtime
    model=VehicleReID(path)
    monkeypatch.setattr(request,'infer',lambda *args,**kwargs:request.output.fill(float('nan')))
    with pytest.raises(RuntimeError,match='invalid embedding'):
        model.embed([np.zeros((20,30,3),np.uint8)])
