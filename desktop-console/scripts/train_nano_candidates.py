"""Train YOLO11n and YOLO26n on identical plate data before comparing speed.

Uses isolated Ultralytics dependencies; does not alter the live model or OCR.
Validation selects checkpoints. The existing test split is never used here.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def write_progress(path,data):
    """Keep a forced stop from leaving the last completed epoch record truncated."""
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(6):
        try:
            os.replace(temporary,path)
            return
        except PermissionError:
            if attempt==5:
                raise
            time.sleep(.05*(attempt+1))


def parse_utc(value):
    try:
        result=datetime.fromisoformat(value.replace('Z','+00:00'))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError('Timezone is required')
        return result.astimezone(timezone.utc)
    except (ValueError,AttributeError) as error:
        raise argparse.ArgumentTypeError('Use an ISO timestamp with timezone, such as 2026-09-27T22:47:46Z.') from error


class TrainingDeadline:
    """Count actual epochs and request stopping only at a completed epoch boundary."""
    def __init__(self,stop_at,total,completed=0):
        self.stop_at,self.total,self.completed=stop_at,total,completed
        self.stop_requested=False

    def epoch_finished(self,trainer):
        self.completed=max(self.completed,min(trainer.epoch+1,self.total))

    def maybe_stop(self,trainer,now=None):
        if self.stop_at is None or self.completed>=self.total:
            return
        now=now or datetime.now(timezone.utc)
        next_epoch=max(0.,float(getattr(trainer,'epoch_time',0.) or 0.))
        if now.timestamp()+next_epoch>=self.stop_at.timestamp():
            trainer.stop=True
            self.stop_requested=True

    def report(self):
        return dict(completed_epochs=self.completed,epochs_requested=self.total,
                    schedule_completed=self.completed>=self.total,
                    stop_at_utc=self.stop_at.isoformat() if self.stop_at else None,
                    stop_reason='deadline_epoch_boundary' if self.stop_requested else None)


def require_device(torch, device):
    """Reject an unavailable requested accelerator before creating a run."""
    if device == 'cpu':
        return
    if device != 'xpu:0':
        raise ValueError('Training device must be cpu or xpu:0.')
    backend=getattr(torch,'xpu',None)
    if backend is None or not backend.is_available() or backend.device_count()<1:
        raise RuntimeError('xpu:0 is unavailable. Use an XPU PyTorch build and compatible Intel driver; CPU fallback is disabled.')


def verify_training_runtime(trainer, device, freeze):
    """Confirm the trainer actually honored the recorded runtime policy."""
    actual=str(trainer.device)
    if actual!=device:
        raise RuntimeError(f'Trainer selected {actual}, but {device} was requested; refusing fallback.')
    expected={'amp':False,'compile':False,'workers':0,'freeze':freeze}
    for key,value in expected.items():
        if getattr(trainer.args,key,None)!=value:
            raise RuntimeError(f'Trainer changed the requested {key} setting.')
    return actual


def verify_resume_options(saved_args, provenance, args):
    """Resume the same run configuration, including device and frozen layers."""
    if (args.epochs!=saved_args['epochs'] or args.batch!=saved_args['batch']
            or args.threads!=provenance['threads']):
        raise ValueError('Resume preserves the original epochs, batch size and thread count.')
    if args.device!=saved_args.get('device',provenance.get('device','cpu')):
        raise ValueError('Resume must use the original training device; a device change requires a new run.')
    if args.freeze!=saved_args.get('freeze',provenance.get('freeze',10)):
        raise ValueError('Resume must preserve the original frozen-layer setting.')
    for key,default in {'warmup_epochs':1.,'close_mosaic':3}.items():
        if getattr(args,key)!=saved_args.get(key,provenance.get(key,default)):
            raise ValueError(f'Resume must preserve the original {key} setting.')
    for key,expected in {'amp':False,'compile':False,'workers':0}.items():
        if saved_args.get(key,expected)!=expected:
            raise ValueError(f'Resume requires the original {key}={expected} policy.')


def resume_candidate(path, args, torch, ultralytics, YOLO, dataset, manifest, output):
    """Restore one interrupted run, without extending its original schedule."""
    path=path.resolve()
    destination=path.parent.parent
    if path.name!='last.pt' or path.parent.name!='weights' or destination.parent!=output.resolve():
        raise ValueError('Resume requires an existing run under platevision/models/runs/.../weights/last.pt.')
    provenance=json.loads((destination/'training-source.json').read_text(encoding='utf-8'))
    name=provenance['architecture']
    if name not in ('yolo11n','yolo26n'):
        raise ValueError('Unknown nano training architecture.')
    source=Path(provenance.get('source_weights',ROOT/f'platevision/models/base/{name}.pt'))
    if sha256(source.read_bytes()).hexdigest()!=provenance['source_weights_sha256']:
        raise ValueError('Original initialization weights changed; refusing resume.')
    if sha256(manifest.read_bytes()).hexdigest()!=provenance['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest changed; refusing resume.')
    if ultralytics.__version__!=provenance['ultralytics'] or torch.__version__!=provenance['torch']:
        raise ValueError('Resume requires the original Ultralytics and torch versions.')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    saved_args=checkpoint['train_args']
    completed=int(checkpoint['epoch'])+1
    total=int(saved_args['epochs'])
    if not 0<completed<total or not checkpoint.get('optimizer'):
        raise ValueError('Checkpoint is complete or lacks resumable optimizer state.')
    if Path(saved_args['data']).resolve()!=dataset.resolve():
        raise ValueError('Checkpoint refers to a different dataset.')
    verify_resume_options(saved_args,provenance,args)
    previous=json.loads((destination/'progress.json').read_text(encoding='utf-8'))
    if previous['epoch']!=completed or previous['epochs']!=total:
        raise ValueError('Progress metadata does not match the resumable checkpoint.')
    previous_elapsed=float(previous['elapsed_seconds'])
    resume_metadata=dict(resumed_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(path),checkpoint_sha256=sha256(path.read_bytes()).hexdigest(),
        completed_epochs_before_resume=completed,next_epoch=completed+1,total_epochs=total,
        elapsed_seconds_before_resume=previous_elapsed,
        source_weights_sha256=provenance['source_weights_sha256'],
        dataset_manifest_sha256=provenance['dataset_manifest_sha256'],
        original_train_args=saved_args,optimizer_restored=False,scheduler_restored=False)
    # Preserve the original provenance and an audit copy before last.pt evolves.
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    audit=destination/f'resume-{stamp}.json'
    audit.write_text(json.dumps(resume_metadata,indent=2),encoding='utf-8')
    saved_optimizer=checkpoint['optimizer']
    expected_steps=sorted(float(value.get('step',0)) for value in saved_optimizer['state'].values())
    expected_updates=int(checkpoint.get('updates',0))
    del checkpoint
    model=YOLO(str(path))
    tick=time.perf_counter()
    deadline=TrainingDeadline(args.stop_at_utc,total,completed)
    def started(trainer):
        if trainer.save_dir.resolve()!=destination or trainer.start_epoch!=completed or trainer.epochs!=total:
            raise RuntimeError('Trainer failed to restore the original run and epoch schedule.')
        ignored={'model','resume','save_dir'}
        changed={key:(value,getattr(trainer.args,key,None)) for key,value in saved_args.items()
                 if key not in ignored and getattr(trainer.args,key,None)!=value}
        if changed:
            raise RuntimeError(f'Resumed training settings differ: {changed}')
        actual_device=verify_training_runtime(trainer,args.device,args.freeze)
        restored=trainer.optimizer.state_dict()
        actual_steps=sorted(float(value.get('step',0)) for value in restored['state'].values())
        if actual_steps!=expected_steps or trainer.ema.updates!=expected_updates:
            raise RuntimeError('Optimizer/EMA update counters were not restored.')
        if [group['lr'] for group in restored['param_groups']]!=[group['lr'] for group in saved_optimizer['param_groups']]:
            raise RuntimeError('Optimizer learning rates were not restored.')
        if trainer.scheduler.last_epoch!=completed-1:
            raise RuntimeError('Learning-rate scheduler did not resume at the checkpoint epoch.')
        resume_metadata.update(optimizer_restored=True,scheduler_restored=True,
                               ema_updates_restored=expected_updates,device=actual_device,
                               amp=False,compile=False,workers=0)
        audit.write_text(json.dumps(resume_metadata,indent=2),encoding='utf-8')
        print('RESUME_VERIFIED '+json.dumps(resume_metadata),flush=True)
    def progress(trainer):
        resumed_elapsed=time.perf_counter()-tick
        deadline.maybe_stop(trainer)
        status=dict(model=name,epoch=deadline.completed,epochs=total,
                    elapsed_seconds=round(previous_elapsed+resumed_elapsed,2),
                    resumed_elapsed_seconds=round(resumed_elapsed,2),
                    elapsed_seconds_before_resume=previous_elapsed,
                    **deadline.report(),
                    metrics={k:float(v) for k,v in trainer.metrics.items()})
        write_progress(destination/'progress.json',status)
        print('TRAIN_PROGRESS '+json.dumps(status),flush=True)
    model.add_callback('on_pretrain_routine_end',started)
    model.add_callback('on_train_epoch_end',deadline.epoch_finished)
    model.add_callback('on_fit_epoch_end',progress)
    # Supplying resume only lets Ultralytics restore data, optimizer, augmentations,
    # schedule and the TOTAL epoch count from last.pt; it does not add 12 epochs.
    model.train(resume=str(path))
    resumed_elapsed=time.perf_counter()-tick
    best=Path(model.trainer.best)
    exported=Path(YOLO(str(best)).export(format='onnx',imgsz=saved_args['imgsz'],batch=1,
                     simplify=False,opset=17,dynamic=True,nms=False,device='cpu'))
    import onnx
    exported_model=onnx.load(exported)
    onnx.checker.check_model(exported_model)
    metadata={prop.key:prop.value for prop in exported_model.metadata_props}
    if 'plate' not in metadata.get('names','').lower():
        raise RuntimeError('The exported checkpoint is not labelled as a plate model.')
    resume_metadata.update(resumed_training_elapsed_seconds=round(resumed_elapsed,2),
                           cumulative_training_elapsed_seconds=round(previous_elapsed+resumed_elapsed,2),
                           **deadline.report(),export_verified=True)
    audit.write_text(json.dumps(resume_metadata,indent=2),encoding='utf-8')
    provenance.update(training_elapsed_seconds=round(previous_elapsed+resumed_elapsed,2),
        **deadline.report(),
        elapsed_seconds_before_resume=previous_elapsed,
        resumed_training_elapsed_seconds=round(resumed_elapsed,2),
        resume=resume_metadata,checkpoint=str(best),onnx=str(exported),
        onnx_sha256=sha256(exported.read_bytes()).hexdigest(),export_dynamic=True,
        export_nms=False,export_device='cpu',license='AGPL-3.0 (Ultralytics)')
    (destination/'model-card.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('CANDIDATE_COMPLETE '+json.dumps(provenance),flush=True)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=['yolo11n','yolo26n'],default=['yolo11n','yolo26n'])
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--device',choices=['cpu','xpu:0'],default='cpu',help='Explicit training device; unavailable XPU fails without CPU fallback.')
    parser.add_argument('--freeze',type=int,default=10,help='Number of initial layers to freeze; 0 disables requested layer freezing.')
    parser.add_argument('--warmup-epochs',type=float,default=1.,help='Warmup duration, between 0 and the requested epoch count.')
    parser.add_argument('--close-mosaic',type=int,default=3,help='Disable mosaic for this many final epochs; 0 disables that change.')
    parser.add_argument('--dependency-dir',type=Path,default=ROOT/'.cache/nano-training-deps',help='Isolated dependency overlay; use a separate Ultralytics-only directory for an XPU environment.')
    parser.add_argument('--tag',default='plates_20260921')
    parser.add_argument('--data',type=Path,default=ROOT/'platevision/data/combined-green-20260919/data.yaml')
    parser.add_argument('--initial-weights',type=Path,help='Start a new schedule from an existing local plate-trained nano checkpoint; does not resume its optimizer.')
    parser.add_argument('--lr0',type=float,default=.001)
    parser.add_argument('--resume',type=Path,help='Resume one existing last.pt to its original total epoch count.')
    parser.add_argument('--stop-at-utc',type=parse_utc,help='Stop at an epoch boundary before this time; report actual completed epochs and retain epoch checkpoints.')
    args=parser.parse_args(argv)
    if Path(args.tag).name!=args.tag:parser.error('Tag must be a directory name')
    if args.epochs<1 or args.batch<1 or args.threads<1 or args.freeze<0 or not 0<args.lr0<=.01:
        parser.error('Positive epochs, batch and threads, nonnegative freeze, and lr0 in (0,.01], are required.')
    if (not math.isfinite(args.warmup_epochs) or not 0<=args.warmup_epochs<=args.epochs
            or not 0<=args.close_mosaic<=args.epochs):
        parser.error('Warmup epochs and close mosaic must be between 0 and the requested epoch count.')
    if args.initial_weights and (len(args.models)!=1 or args.resume):
        parser.error('Initial weights require one model and a new run, not --resume.')
    return args


def main():
    args=parse_args()
    if args.stop_at_utc is not None and datetime.now(timezone.utc)>=args.stop_at_utc:
        raise ValueError('The training stop time has already passed; no run was started.')
    dependencies=args.dependency_dir.resolve()
    if not dependencies.is_dir():
        raise FileNotFoundError(f'Dependency directory does not exist: {dependencies}')
    sys.path.insert(0,str(dependencies))
    config=ROOT/'.cache/ultralytics-nano'
    (config/'Ultralytics').mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(config)
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['OMP_NUM_THREADS']=str(args.threads)
    import torch
    require_device(torch,args.device)
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS,torch_utils
    SETTINGS.update({key:False for key in ('sync','wandb','clearml','comet',
                     'mlflow','neptune','tensorboard') if key in SETTINGS})
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch_utils.NUM_THREADS=args.threads
    dataset=args.data.resolve()
    manifest=dataset.parent/'manifest.json'
    if not dataset.is_file() or not manifest.is_file():
        raise FileNotFoundError('Training requires a prepared dataset YAML and adjacent provenance manifest.')
    output=ROOT/'platevision/models/runs'
    if args.resume:
        resume_candidate(args.resume,args,torch,ultralytics,YOLO,dataset,manifest,output)
        return
    for name in args.models:
        destination=output/f'{name}_{args.tag}'
        if destination.exists():raise FileExistsError(f'Refusing to overwrite {destination}')
        source=args.initial_weights.resolve() if args.initial_weights else ROOT/f'platevision/models/base/{name}.pt'
        if not source.is_file():raise FileNotFoundError(source)
        parent_source=None
        if args.initial_weights:
            parent_path=source.parent.parent/'training-source.json'
            parent_source=json.loads(parent_path.read_text(encoding='utf-8'))
            if parent_source.get('architecture')!=name:
                raise ValueError('Initialization checkpoint architecture does not match the requested nano model.')
        provenance={'architecture':name,'initialization':'plate-trained checkpoint; fresh optimizer and schedule' if args.initial_weights else 'COCO-pretrained official nano checkpoint',
            'source_weights':str(source.resolve()),'dataset':str(dataset),
            'source_weights_sha256':sha256(source.read_bytes()).hexdigest(),
            'dataset_manifest_sha256':sha256(manifest.read_bytes()).hexdigest(),
            'ultralytics':ultralytics.__version__,'torch':torch.__version__,
            'training_split':'train','checkpoint_selection_split':'val','test_used':False,
            'epochs_requested':args.epochs,'seed':42,'imgsz':640,'batch':args.batch,
            'threads':args.threads,'freeze':args.freeze,'optimizer':'AdamW','lr0':args.lr0,
            'warmup_epochs':args.warmup_epochs,'close_mosaic':args.close_mosaic,
            'requested_device':args.device,'device':args.device,'amp':False,'compile':False,'workers':0,
            'stop_at_utc':args.stop_at_utc.isoformat() if args.stop_at_utc else None,
            'save_period':1 if args.stop_at_utc else -1,
            'dependency_dir':str(dependencies),'torch_module':str(Path(torch.__file__).resolve()),
            'ultralytics_module':str(Path(ultralytics.__file__).resolve()),
            'weights_retrained':True,'ocr_retrained':False,'live_model_overwritten':False}
        if parent_source is not None:
            provenance['parent_training_source']=parent_source
        model=YOLO(str(source))
        deadline=TrainingDeadline(args.stop_at_utc,args.epochs)
        if args.initial_weights and (len(model.names)!=1 or 'plate' not in str(model.names).lower()):
            raise ValueError('Initial weights must be the recorded single-class plate checkpoint.')
        def started(trainer):
            if (trainer.save_dir.resolve()!=destination.resolve() or trainer.start_epoch!=0
                    or trainer.epochs!=args.epochs):
                raise RuntimeError('Fresh training did not use the requested output directory and epoch schedule.')
            provenance['device']=verify_training_runtime(trainer,args.device,args.freeze)
            (trainer.save_dir/'training-source.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        def progress(trainer):
            deadline.maybe_stop(trainer)
            status={'model':name,'epoch':deadline.completed,'epochs':trainer.epochs,
                    **deadline.report(),
                    'elapsed_seconds':round(time.perf_counter()-tick,2),
                    'metrics':{k:float(v) for k,v in trainer.metrics.items()}}
            write_progress(trainer.save_dir/'progress.json',status)
            print('TRAIN_PROGRESS '+json.dumps(status),flush=True)
        model.add_callback('on_pretrain_routine_end',started)
        model.add_callback('on_train_epoch_end',deadline.epoch_finished)
        model.add_callback('on_fit_epoch_end',progress)
        tick=time.perf_counter()
        model.train(data=str(dataset),epochs=args.epochs,imgsz=640,batch=args.batch,
            patience=args.epochs+1,project=str(output),name=destination.name,
            seed=42,device=args.device,workers=0,freeze=args.freeze,cache=False,plots=False,
            amp=False,compile=False,optimizer='AdamW',lr0=args.lr0,weight_decay=.0005,
            warmup_epochs=args.warmup_epochs,warmup_bias_lr=.0003,close_mosaic=args.close_mosaic,
            degrees=15,translate=.1,scale=.5,fliplr=.25,hsv_h=.005,
            hsv_s=.3,hsv_v=.3,mosaic=.8,mixup=0,cos_lr=True,save_period=1 if args.stop_at_utc else -1)
        elapsed=time.perf_counter()-tick
        checkpoint=Path(model.trainer.best)
        exported=Path(YOLO(str(checkpoint)).export(format='onnx',imgsz=640,batch=1,
                         simplify=False,opset=17,dynamic=True,nms=False,device='cpu'))
        import onnx
        exported_model=onnx.load(exported)
        onnx.checker.check_model(exported_model)
        if 'plate' not in {prop.key:prop.value for prop in exported_model.metadata_props}.get('names','').lower():
            raise RuntimeError('Export is missing its plate class metadata.')
        provenance.update(training_elapsed_seconds=round(elapsed,2),
            **deadline.report(),
            checkpoint=str(checkpoint),onnx=str(exported),
            onnx_sha256=sha256(exported.read_bytes()).hexdigest(),
            export_dynamic=True,export_nms=False,export_device='cpu',license='AGPL-3.0 (Ultralytics)')
        (destination/'model-card.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        print('CANDIDATE_COMPLETE '+json.dumps(provenance),flush=True)


if __name__=='__main__':main()
