"""Export compatibility tests, with no GPU inference or downloaded weights."""
import numpy as np
import pytest

from app.core.config import settings
from app.services.detector import OnnxPlateDetector, OpenVINOPlateDetector, detection_output_format
from app.services.model_shape import resize_plate_model


def detector(output_format='end_to_end', classes=1, allowed=None):
    result = object.__new__(OnnxPlateDetector)
    result.output_format, result.classes, result.allowed = output_format, classes, allowed
    return result


def test_plate_export_layouts_and_metadata():
    names = {'names': "{0: 'number_plate'}", 'task': 'detect'}
    assert detection_output_format([1,5,8400], metadata=names) == 'raw'
    assert detection_output_format(['batch',5,'anchors'], metadata=names) == 'raw'
    assert detection_output_format([1,300,6], metadata=names) == 'end_to_end'
    assert detection_output_format([1,'detections',6], metadata=names) == 'end_to_end'


def test_coco_and_non_plate_exports_are_rejected():
    coco_names = {'names': str({i: str(i) for i in range(80)})}
    for shape in ([1,84,8400], [1,300,6]):
        with pytest.raises(RuntimeError, match='classes'):
            detection_output_format(shape, metadata=coco_names)
    with pytest.raises(RuntimeError, match='number plates'):
        detection_output_format([1,5,8400], metadata={'names': "{0: 'car'}"})
    with pytest.raises(RuntimeError, match='class-name metadata'):
        detection_output_format([1,300,6])
    with pytest.raises(RuntimeError, match='Unsupported output'):
        detection_output_format([1,84,8400])
    with pytest.raises(RuntimeError, match='zero-based'):
        detection_output_format([1,5,8400], metadata={'names': "{1: 'plate'}"})


def test_ambiguous_shape_requires_explicit_export_layout_metadata():
    names = {'names': "{0: 'plate'}"}
    with pytest.raises(RuntimeError, match='Ambiguous'):
        detection_output_format([1,5,6], metadata=names)
    native = {**names, 'args': "{'end2end': True, 'nms': False}"}
    assert detection_output_format([1,5,6], metadata=native) == 'end_to_end'
    raw = {**names, 'end2end': 'False', 'args': "{'nms': False}"}
    assert detection_output_format([1,5,6], metadata=raw) == 'raw'
    with pytest.raises(RuntimeError, match='conflicts'):
        detection_output_format([1,5,8400], metadata=native)
    with pytest.raises(RuntimeError, match='conflicts'):
        detection_output_format([1,300,6], metadata=raw)


def test_symbolic_native_export_requires_metadata_and_actual_warmup_validation(monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    import app.services.detector as module
    metadata = {'names': "{0: 'plate'}", 'end2end': 'True', 'args': "{'nms': False}"}
    symbolic_shape = ['batch','Concatoutput0_dim_1','anchors']
    assert detection_output_format(symbolic_shape, metadata=metadata) == 'end_to_end'
    with pytest.raises(RuntimeError, match='Unsupported output'):
        detection_output_format(symbolic_shape, metadata={'names': "{0: 'plate'}"})
    class Session:
        output = np.zeros((1,300,6),np.float32)
        def get_inputs(self):
            return [SimpleNamespace(shape=['batch',3,'height','width'], type='tensor(float)', name='images')]
        def get_outputs(self):
            return [SimpleNamespace(shape=symbolic_shape)]
        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map=metadata)
        def run(self,*args,**kwargs):
            return [self.output]
    monkeypatch.setattr(module.ort, 'InferenceSession', lambda *args,**kwargs: Session())
    monkeypatch.setattr(module.ort, 'get_available_providers', lambda: ['CPUExecutionProvider'])
    monkeypatch.setattr(settings, 'onnx_provider', 'CPUExecutionProvider')
    # A real file is required by the constructor; inference is deliberately
    # mocked, so this test needs no model artifact or accelerated hardware.
    model = OnnxPlateDetector(str(Path(__file__)))
    assert model.output_format == 'end_to_end'
    Session.output = np.zeros((1,300,7),np.float32)
    with pytest.raises(RuntimeError, match='end-to-end'):
        OnnxPlateDetector(str(Path(__file__)))
    Session.output = np.zeros((2,300,6),np.float32)
    with pytest.raises(RuntimeError, match='batch-one'):
        OnnxPlateDetector(str(Path(__file__)))


def test_vehicle_detector_still_accepts_eighty_classes():
    names = {'names': str({i: str(i) for i in range(80)})}
    assert detection_output_format([1,84,8400], 80, names) == 'raw'
    assert detection_output_format([1,300,6], 80, names) == 'end_to_end'
    output = np.array([[[0,0,30,10,.95,0],[40,0,70,10,.9,2],[80,0,100,10,.8,7]]], np.float32)
    results = detector(classes=80, allowed={2,3,5,7}).postprocess(output,1,0,0,100,100)
    assert [result[-1] for result in results] == [2,7]


def test_end_to_end_decodes_xyxy_letterbox_and_clips():
    output = np.array([[[16,36,116,76,.9,0],[196,56,236,96,.8,0]]], np.float32)
    results = detector().postprocess(output,2,16,16,100,40)
    assert [row[:4] for row in results] == [(0,10,50,20),(90,20,10,20)]


def test_end_to_end_preserves_overlapping_plates_without_second_nms(monkeypatch):
    import cv2
    def forbidden(*args, **kwargs):
        raise AssertionError('Native end-to-end exports must not run a second NMS.')
    monkeypatch.setattr(cv2.dnn, 'NMSBoxes', forbidden)
    output = np.array([[[10,10,110,40,.9,0],[12,11,112,41,.8,0]]], np.float32)
    assert len(detector().postprocess(output,1,0,0,200,100)) == 2


def test_fast_single_pass_keeps_overlapping_native_plates(monkeypatch):
    output = np.array([[[10,10,110,40,.9,0],[12,11,112,41,.8,0],
                        [150,20,190,21,.7,0],[220,20,240,40,.95,0]]], np.float32)
    graph = detector()
    graph.detect = lambda image: [row[:5] for row in graph.postprocess(output,1,0,0,200,100)]
    main = object.__new__(OpenVINOPlateDetector)
    main.fast_models = {'landscape':graph,'portrait':graph}
    def forbidden(*args, **kwargs):
        raise AssertionError('Single-pass native detections must not be merged.')
    monkeypatch.setattr(graph, 'merge_plates', forbidden)
    # The two overlapping plates survive; the invalid aspect-ratio and
    # entirely out-of-image detections do not.
    result = main.detect_fast(np.zeros((100,200,3),np.uint8))
    assert [row[:4] for row in result] == [(10,10,100,30),(12,11,100,30)]
    monkeypatch.setattr(settings, 'max_detections', 1)
    assert len(main.detect_fast(np.zeros((100,200,3),np.uint8))) == 1


def test_native_tiled_passes_still_merge_repeated_observations(monkeypatch):
    graph = detector()
    monkeypatch.setattr(settings,'plate_tile_size',320)
    calls = []
    def detect(image,threshold=None):
        calls.append(image.shape)
        if image.shape[1] == 500:
            return [(190,10,60,20,.9)]
        return [(190,10,60,20,.8)] if len(calls)==2 else [(10,10,60,20,.8)]
    graph.detect = detect
    assert len(graph.detect_plates(np.zeros((100,500,3),np.uint8),refine=False)) == 1
    assert len(calls) == 3


def test_end_to_end_keeps_all_candidates_up_to_configured_limit(monkeypatch):
    output = np.array([[[i*20,10,i*20+15,20,.3+i*.02,0] for i in range(18)]], np.float32)
    monkeypatch.setattr(settings, 'max_detections', 12)
    results = detector().postprocess(output,1,0,0,400,100)
    assert len(results) == 12
    assert [row[0] for row in results] == list(range(340,100,-20))


def test_end_to_end_rejects_invalid_rows_and_padded_zero_detections():
    output = np.array([[
        [10,10,40,20,.9,0],
        [10,10,40,20,float('nan'),0],
        [10,10,40,20,1.1,0],
        [10,10,40,20,.9,1],
        [10,10,40,20,.9,.5],
        [float('inf'),10,40,20,.9,0],
        [40,10,10,20,.9,0],
        [-40,10,-10,20,.9,0],
        [0,0,0,0,0,0],
    ]], np.float32)
    result = detector().postprocess(output,1,0,0,100,100)
    assert len(result) == 1 and result[0][:4] == (10,10,30,10)
    with pytest.raises(RuntimeError, match='end-to-end'):
        detector().postprocess(np.zeros((1,5,8400)),1,0,0,100,100)


def test_raw_export_still_deduplicates_but_rejects_impossible_confidence():
    output = np.array([[[50,51,150],[20,21,20],[80,80,80],[20,20,20],[.9,.8,1.1]]], np.float32)
    result = detector('raw').postprocess(output,1,0,0,200,100)
    assert len(result) == 1 and result[0][:4] == (10,10,80,20)


def test_dynamic_graph_uses_standard_reshape_for_both_orientations():
    ov = pytest.importorskip('openvino')
    from openvino import opset13 as ops
    def make_model():
        image = ops.parameter(ov.PartialShape([-1,3,-1,-1]), np.float32)
        channel = ops.reduce_mean(image, ops.constant([1]), True)
        flat = ops.reshape(channel, ops.constant([0,1,-1]), True)
        output = ops.tile(flat, ops.constant([1,5,1]))
        return ov.Model([output], [image])
    for height,width in ((864,1536),(1536,864)):
        resized = resize_plate_model(make_model(),height,width)
        assert list(resized.input(0).shape) == [1,3,height,width]
        assert list(resized.output(0).shape) == [1,5,height*width]


def test_unknown_fixed_grid_requires_dynamic_reexport():
    ov = pytest.importorskip('openvino')
    from openvino import opset13 as ops
    image = ops.parameter([1,3,640,640], np.float32)
    flat = ops.reshape(image, ops.constant([1,3,-1]), False)
    model = ov.Model([flat], [image])
    assert resize_plate_model(model,640,640) is model
    with pytest.raises(ValueError, match='dynamic=True'):
        resize_plate_model(model,864,1536)
    with pytest.raises(ValueError, match='positive multiples'):
        resize_plate_model(model,0,640)
