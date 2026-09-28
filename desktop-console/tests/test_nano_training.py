"""CPU-only orchestration tests; no Torch import, model training or GPU work."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_nano_candidates as training


def test_original_defaults_and_new_full_training_options():
    default=training.parse_args([])
    assert (default.epochs,default.device,default.freeze,default.warmup_epochs,default.close_mosaic)==(12,'cpu',10,1.,3)
    assert default.dependency_dir==training.ROOT/'.cache/nano-training-deps'
    full=training.parse_args(['--models','yolo11n','--epochs','100','--device','xpu:0',
                             '--freeze','0','--warmup-epochs','3','--close-mosaic','10',
                             '--dependency-dir','separate-dependencies'])
    assert (full.device,full.freeze,full.warmup_epochs,full.close_mosaic)==('xpu:0',0,3.,10)
    assert full.dependency_dir==Path('separate-dependencies')


@pytest.mark.parametrize('arguments',[
    ['--device','cuda:0'],['--freeze','-1'],['--warmup-epochs','nan'],
    ['--warmup-epochs','13'],['--close-mosaic','13'],['--close-mosaic','-1'],
])
def test_invalid_runtime_or_schedule_options_fail_before_loading_models(arguments):
    with pytest.raises(SystemExit):
        training.parse_args(arguments)


@pytest.mark.parametrize('backend',[None,SimpleNamespace(is_available=lambda:False),
                                   SimpleNamespace(is_available=lambda:True,device_count=lambda:0)])
def test_unavailable_xpu_is_not_replaced_with_cpu(backend):
    torch=SimpleNamespace(xpu=backend)
    with pytest.raises(RuntimeError,match='CPU fallback is disabled'):
        training.require_device(torch,'xpu:0')
    training.require_device(torch,'cpu')


def test_trainer_device_and_precision_are_verified():
    trainer=SimpleNamespace(device='cpu',args=SimpleNamespace(amp=False,compile=False,workers=0,freeze=0))
    with pytest.raises(RuntimeError,match='refusing fallback'):
        training.verify_training_runtime(trainer,'xpu:0',0)
    trainer.device='xpu:0'
    assert training.verify_training_runtime(trainer,'xpu:0',0)=='xpu:0'
    trainer.args.amp=True
    with pytest.raises(RuntimeError,match='amp setting'):
        training.verify_training_runtime(trainer,'xpu:0',0)


def test_resume_preserves_original_device_freeze_and_total_schedule():
    args=training.parse_args(['--epochs','100','--device','xpu:0','--freeze','0',
                              '--warmup-epochs','3','--close-mosaic','10'])
    saved=dict(epochs=100,batch=8,device='xpu:0',freeze=0,warmup_epochs=3.,close_mosaic=10,
               amp=False,compile=False,workers=0)
    provenance=dict(threads=4)
    training.verify_resume_options(saved,provenance,args)
    for key,value in [('device','cpu'),('freeze',10),('epochs',112),('warmup_epochs',1.),('close_mosaic',3)]:
        with pytest.raises(ValueError,match='Resume'):
            training.verify_resume_options({**saved,key:value},provenance,args)
    legacy=training.parse_args([])
    training.verify_resume_options(dict(epochs=12,batch=8,device='cpu',freeze=10),provenance,legacy)


def test_deadline_stops_at_boundary_and_final_validation_cannot_add_an_epoch():
    cutoff=datetime(2030,1,1,tzinfo=timezone.utc)
    deadline=training.TrainingDeadline(cutoff,20)
    trainer=SimpleNamespace(epoch=4,epoch_time=60.,stop=False)
    deadline.epoch_finished(trainer)
    deadline.maybe_stop(trainer,now=cutoff-timedelta(seconds=30))
    assert trainer.stop and deadline.completed==5
    trainer.epoch+=1  # Ultralytics increments this only to log its final validation.
    deadline.maybe_stop(trainer,now=cutoff)
    report=deadline.report()
    assert report['completed_epochs']==5 and report['epochs_requested']==20
    assert report['schedule_completed'] is False
    assert report['stop_reason']=='deadline_epoch_boundary'


def test_stop_time_requires_timezone_and_normalizes_to_utc():
    value=training.parse_args(['--stop-at-utc','2030-01-01T05:30:00+05:30']).stop_at_utc
    assert value==datetime(2030,1,1,tzinfo=timezone.utc)
    with pytest.raises(SystemExit):
        training.parse_args(['--stop-at-utc','2030-01-01T05:30:00'])


@pytest.mark.parametrize('device,freeze,epochs,warmup,close',[
    ('cpu',10,12,1.,3),('xpu:0',0,100,3.,10),
])
def test_new_run_records_actual_runtime_and_exports_on_cpu(tmp_path,monkeypatch,device,freeze,epochs,warmup,close):
    """Drive main with fake engines, then inspect produced audit files and calls."""
    monkeypatch.setattr(training,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'path',list(sys.path))
    for name in ('YOLO_CONFIG_DIR','MPLCONFIGDIR','YOLO_AUTOINSTALL','OMP_NUM_THREADS'):
        monkeypatch.setenv(name,'test-original')
    overlay=tmp_path/'isolated-dependencies'; overlay.mkdir()
    dataset=tmp_path/'dataset'; dataset.mkdir()
    (dataset/'data.yaml').write_text('names: [number_plate]\n',encoding='utf-8')
    (dataset/'manifest.json').write_text('{}',encoding='utf-8')
    source=tmp_path/'platevision/models/base/yolo11n.pt'
    source.parent.mkdir(parents=True); source.write_bytes(b'local source checkpoint')
    calls=[]

    class FakeYOLO:
        names={0:'number_plate'}

        def __init__(self,path):
            self.path=Path(path)
            self.callbacks={}

        def add_callback(self,name,callback):
            self.callbacks[name]=callback

        def train(self,**kwargs):
            calls.append(('train',kwargs))
            destination=Path(kwargs['project'])/kwargs['name']
            destination.mkdir(parents=True,exist_ok=False)
            weights=destination/'weights'; weights.mkdir()
            best=weights/'best.pt'; best.write_bytes(b'new checkpoint')
            self.trainer=SimpleNamespace(save_dir=destination,start_epoch=0,epochs=kwargs['epochs'],
                device=kwargs['device'],args=SimpleNamespace(**kwargs),best=best,epoch=kwargs['epochs']-1)
            self.callbacks['on_pretrain_routine_end'](self.trainer)
            self.callbacks['on_train_epoch_end'](self.trainer)

        def export(self,**kwargs):
            calls.append(('export',kwargs))
            result=self.path.with_suffix('.onnx'); result.write_bytes(b'export')
            return result

    torch=ModuleType('torch'); torch.__version__='fixture+xpu'; torch.__file__=str(tmp_path/'runtime/torch/__init__.py')
    torch.xpu=SimpleNamespace(is_available=lambda:True,device_count=lambda:1)
    torch.set_num_threads=lambda _:None; torch.set_num_interop_threads=lambda _:None
    ultralytics=ModuleType('ultralytics'); ultralytics.__version__='fixture'
    ultralytics.__file__=str(overlay/'ultralytics/__init__.py'); ultralytics.YOLO=FakeYOLO
    utils=ModuleType('ultralytics.utils'); utils.SETTINGS={}; utils.torch_utils=SimpleNamespace()
    onnx=ModuleType('onnx')
    onnx.load=lambda _:SimpleNamespace(metadata_props=[SimpleNamespace(key='names',value="{0: 'number_plate'}")])
    onnx.checker=SimpleNamespace(check_model=lambda _:None)
    for name,module in [('torch',torch),('ultralytics',ultralytics),('ultralytics.utils',utils),('onnx',onnx)]:
        monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr(sys,'argv',['train_nano_candidates.py','--models','yolo11n','--tag','fixture',
        '--device',device,'--freeze',str(freeze),'--epochs',str(epochs),'--warmup-epochs',str(warmup),
        '--close-mosaic',str(close),'--data',str(dataset/'data.yaml'),'--dependency-dir',str(overlay)])

    training.main()

    run=tmp_path/'platevision/models/runs/yolo11n_fixture'
    provenance=json.loads((run/'training-source.json').read_text())
    assert {key:provenance[key] for key in ('device','freeze','amp','compile','workers')}==dict(
        device=device,freeze=freeze,amp=False,compile=False,workers=0)
    assert provenance['dependency_dir']==str(overlay.resolve())
    assert provenance['torch_module']==str(Path(torch.__file__).resolve())
    assert provenance['warmup_epochs']==warmup and provenance['close_mosaic']==close
    train_args=calls[0][1]
    assert train_args['epochs']==epochs and train_args['patience']==epochs+1
    assert train_args['device']==device and train_args['freeze']==freeze
    assert train_args['warmup_epochs']==warmup and train_args['close_mosaic']==close
    assert train_args['amp'] is False and train_args['compile'] is False
    assert calls[1][0]=='export' and calls[1][1]['device']=='cpu'
    card=json.loads((run/'model-card.json').read_text())
    assert card['export_device']=='cpu' and card['device']==device
    assert card['completed_epochs']==epochs and card['schedule_completed'] is True
    assert source.read_bytes()==b'local source checkpoint'
