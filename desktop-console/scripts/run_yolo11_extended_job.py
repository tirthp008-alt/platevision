"""Supervise one local YOLO11 training run followed by a fresh model comparison.

No automatic retries or model activation. Job status and logs survive a closed
terminal. Launch separately only when data preparation and runtime checks pass.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from train_nano_candidates import parse_utc

ROOT=Path(__file__).resolve().parents[1]
ES_CONTINUOUS=0x80000000
ES_SYSTEM_REQUIRED=0x00000001


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    # Windows file watchers can briefly hold the destination without delete
    # sharing. Retry only this atomic rename, never training or comparison.
    for attempt in range(6):
        try:
            os.replace(temporary,path)
            break
        except PermissionError:
            if attempt==5:
                raise
            time.sleep(.05*(attempt+1))


@contextmanager
def keep_system_awake(setter=None):
    """Hold only this thread's system-sleep request, then restore its old state."""
    if setter is None:
        if os.name!='nt':
            yield False
            return
        import ctypes
        from ctypes import wintypes
        setter=ctypes.windll.kernel32.SetThreadExecutionState
        setter.argtypes=[wintypes.DWORD]
        setter.restype=wintypes.DWORD
    previous=setter(ES_CONTINUOUS|ES_SYSTEM_REQUIRED)
    if not previous:
        raise RuntimeError('Windows refused the temporary system-awake request.')
    try:
        yield True
    finally:
        if not setter(ES_CONTINUOUS|previous):
            raise RuntimeError('Windows failed to restore this thread\'s execution state.')


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','xpu:0'],required=True)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--initial-weights',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--report-test',action='store_true',help='Report the reserved split only after validation selection freezes.')
    parser.add_argument('--deadline-utc',type=parse_utc,required=True,help='Hard completion deadline for training and comparison.')
    parser.add_argument('--reserve-evaluation-minutes',type=float,default=45.,help='Time reserved before the hard deadline for comparison.')
    parser.add_argument('--validation-dataset',type=Path,help='Frozen labelled evaluation dataset; never use new training images here.')
    parser.add_argument('--reference-model',type=Path,help='Reference plate-detector ONNX export.')
    parser.add_argument('--prior-model',type=Path,help='Previous YOLO11 plate-detector ONNX export.')
    parser.add_argument('--training-python',type=Path,help='Python interpreter in the separate training environment.')
    parser.add_argument('--comparison-python',type=Path,help='Python interpreter with the desktop backend dependencies.')
    parser.add_argument('--comparison-device',choices=['CPU','GPU'],default='GPU')
    args=parser.parse_args(argv)
    if not 15<=args.epochs<=100 or args.batch<1:
        parser.error('Use 15 to 100 epochs and a positive fixed batch size.')
    if not math.isfinite(args.reserve_evaluation_minutes) or args.reserve_evaluation_minutes<1:
        parser.error('Reserve at least one finite minute for evaluation.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',args.tag) or args.tag.endswith('.'):
        parser.error('Tag must be a simple directory label using letters, digits, underscores, dots or hyphens.')
    return args


def job_plan(args, root=ROOT):
    root=root.resolve()
    folder=root/'.cache'/f'yolo11-extended-{args.tag}'
    run=root/'platevision/models/runs'/f'yolo11n_{args.tag}'
    interpreter=Path('Scripts/python.exe') if os.name=='nt' else Path('bin/python')
    training_python=args.training_python or root/('.venv-train-xpu' if args.device=='xpu:0' else '.venv-train')/interpreter
    comparison_python=args.comparison_python or root/'.venv'/interpreter
    data=args.data.resolve()
    initial=args.initial_weights.resolve()
    frozen=args.validation_dataset or root/'platevision/data/combined-yolo11-20260927'
    green=args.reference_model or root/'platevision/models/runs/combined_green_v3/weights/best.onnx'
    prior=args.prior_model or root/'platevision/models/runs/yolo11n_plates_20260921/weights/best.onnx'
    exported=run/'weights/best.onnx'
    report=folder/'comparison.json'
    training_cutoff=args.deadline_utc-timedelta(minutes=args.reserve_evaluation_minutes)
    stop_at=training_cutoff-timedelta(minutes=10)
    train=[str(training_python),str(root/'scripts/train_nano_candidates.py'),
           '--models','yolo11n','--device',args.device,'--epochs',str(args.epochs),
           '--batch',str(args.batch),'--threads','4','--freeze','0','--warmup-epochs','3',
           '--close-mosaic',str(min(10,args.epochs//4)),'--lr0','.0003','--tag',args.tag,
           '--stop-at-utc',stop_at.isoformat(),
           '--data',str(data),'--initial-weights',str(initial)]
    compare=[str(comparison_python),str(root/'scripts/compare_nano_models.py'),
             '--model',f'green8={green}','--model',f'prior11={prior}','--model',f'new11={exported}',
             '--reference','green8','--candidate','prior11','--candidate','new11',
             '--dataset',str(frozen),'--output',str(report),'--runs','30','--warmup','5',
             '--device',args.comparison_device,'--confidence','.25','--threads','4']
    if args.report_test:
        compare.append('--report-test')
    resume=train[:-2]+['--resume',str(run/'weights/last.pt')]
    required=[training_python,comparison_python,root/'scripts/train_nano_candidates.py',
              root/'scripts/compare_nano_models.py',initial,
              initial.parent.parent/'training-source.json',data,data.parent/'manifest.json',
              frozen/'manifest.json',green,prior,root/'tests/ten-plate-1080p.jpg',
              root/'tests/ten-plate-fast-benchmark.json']
    return dict(root=root,folder=folder,run=run,exported=exported,report=report,
                frozen=frozen,initial=initial,data=data,green=green,prior=prior,
                training_cutoff=training_cutoff,stop_at=stop_at,deadline=args.deadline_utc,
                train=train,compare=compare,resume=resume,required=required)


def file_digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_progress(run):
    progress=run/'progress.json'
    try:
        if progress.stat().st_size>1_000_000:
            return None
        data=json.loads(progress.read_text(encoding='utf-8'))
        return data if isinstance(data,dict) else None
    except (OSError,ValueError):
        return None


def stop_child(process,force=False,windows=None,run_command=subprocess.run):
    """Interrupt only the child created by this supervisor, with a bounded wait."""
    if process.poll() is None:
        if windows if windows is not None else os.name=='nt':
            # Windows venv python.exe redirects to a real Python child. Killing
            # only the redirector leaves training alive, so stop its owned tree.
            result=run_command(['taskkill.exe','/PID',str(process.pid),'/T','/F'],
                               stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=15,check=False)
            if result.returncode and process.poll() is None:
                raise RuntimeError(f'Failed to terminate owned Windows process tree {process.pid}: {result.stdout!r}')
            return process.wait(timeout=15)
        process.kill() if force else process.terminate()
        try:
            return process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=15)
    return process.returncode


def run_job(args, root=ROOT, popen=subprocess.Popen, awake=keep_system_awake, now=None):
    now=now or (lambda:datetime.now(timezone.utc))
    plan=job_plan(args,root)
    plan['folder'].mkdir(parents=True,exist_ok=False)
    status={
        'schema_version':1,'state':'starting','supervisor_pid':os.getpid(),
        'created_utc':utc_now(),'started_utc':utc_now(),'finished_utc':None,
        'arguments':{key:value.isoformat() if isinstance(value,datetime) else str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        'deadline_utc':plan['deadline'].isoformat(),'training_cutoff_utc':plan['training_cutoff'].isoformat(),
        'training_stop_at_utc':plan['stop_at'].isoformat(),
        'job_folder':str(plan['folder']),'training_run':str(plan['run']),
        'comparison_report':str(plan['report']),'comparison_dataset':str(plan['frozen']),
        'stages':{name:{'state':'pending','command':plan[command],'pid':None,'exit_code':None,
                         'started_utc':None,'finished_utc':None,'log':str(plan['folder']/f'{name}.log')}
                  for name,command in [('training','train'),('comparison','compare')]},
        'automatic_retry':False,'default_model_changed':False,
        'recovery':{
            'training_resume_command':plan['resume'],
            'note':'No automatic retries or work after this deadline. Any later continuation needs a new authorized deadline. For an interrupted run, verify a resumable last.pt and matching progress.json before using the saved resume command with the same runtime/data and a new stop-at-utc. Gracefully finalized checkpoints may have stripped optimizer state; initialize a new run from those weights instead. If only comparison failed, rerun its saved command with a fresh output filename. Do not rerun this supervisor with the same tag.',
        },
    }
    status_path=plan['folder']/'status.json'

    def save():
        status['updated_utc']=utc_now()
        progress=read_progress(plan['run'])
        if progress is not None:
            status['training_progress']=progress
        atomic_json(status_path,status)

    def log(message):
        line=f'{utc_now()} {message}'
        with (plan['folder']/'supervisor.log').open('a',encoding='utf-8') as stream:
            stream.write(line+'\n')
        print(line,flush=True)

    def run_stage(name):
        stage=status['stages'][name]
        cutoff=plan['training_cutoff'] if name=='training' else plan['deadline']
        if now()>=cutoff:
            stage.update(state='timed_out',finished_utc=utc_now())
            raise TimeoutError(f'{name.capitalize()} deadline has passed; no subprocess was started.')
        status['state']=name
        stage.update(state='starting',started_utc=utc_now())
        save()
        process=None
        try:
            with Path(stage['log']).open('x',encoding='utf-8') as output:
                process=popen(stage['command'],cwd=str(plan['root']),stdin=subprocess.DEVNULL,
                              stdout=output,stderr=subprocess.STDOUT,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0) if os.name=='nt' else 0)
                stage.update(state='running',pid=process.pid)
                save()
                log(f'{name.capitalize()} started, PID {process.pid}; log: {stage["log"]}')
                while True:
                    remaining=(cutoff-now()).total_seconds()
                    if remaining<=0:
                        stage.update(state='timed_out',deadline_exceeded=True)
                        raise TimeoutError(f'{name.capitalize()} exceeded its deadline; retained checkpoints/logs are partial results.')
                    try:
                        code=process.wait(timeout=min(30.,remaining))
                        break
                    except subprocess.TimeoutExpired:
                        save()
                if now()>cutoff:
                    stage.update(state='timed_out',deadline_exceeded=True)
                    raise TimeoutError(f'{name.capitalize()} finished after its deadline; results require review.')
                stage.update(state='completed' if code==0 else 'failed',exit_code=code,finished_utc=utc_now())
                save()
                if code!=0:
                    raise RuntimeError(f'{name.capitalize()} exited with code {code}; see {stage["log"]}')
        except BaseException:
            if process is not None:
                stage['exit_code']=stop_child(process,force=isinstance(sys.exc_info()[1],TimeoutError))
            stage['finished_utc']=utc_now()
            if stage['state'] not in {'completed','timed_out'}:
                stage.update(state='interrupted' if isinstance(sys.exc_info()[1],KeyboardInterrupt) else 'failed',
                             finished_utc=utc_now())
            save()
            raise

    save()
    try:
        if now()>=plan['stop_at']:
            raise TimeoutError('Insufficient time remains before the deadline for training, export and reserved evaluation; no run was started.')
        if plan['run'].exists():
            raise FileExistsError(f'Refusing to overwrite existing training run: {plan["run"]}')
        for path in plan['required']:
            if not path.is_file():
                raise FileNotFoundError(f'Required input or interpreter is missing: {path}')
        for split in ['val']+(['test'] if args.report_test else []):
            images=plan['frozen']/split/'images'
            if not any('_context_' not in path.stem for path in images.glob('*.jpg')):
                raise FileNotFoundError(f'No whole-image {split} fixtures in the frozen comparison dataset.')
        status['input_sha256']={str(path):file_digest(path) for path in (
            plan['initial'],plan['data'],plan['data'].parent/'manifest.json',
            plan['frozen']/'manifest.json',plan['green'],plan['prior'])}
        save()
        with awake() as supported:
            status['system_awake_request']=bool(supported)
            save()
            log('Starting supervised training; the reference API and model files are not activated or changed.')
            run_stage('training')
            if not plan['exported'].is_file():
                raise FileNotFoundError(f'Training exited successfully without the required ONNX export: {plan["exported"]}')
            for filename,expected in status['input_sha256'].items():
                if file_digest(Path(filename))!=expected:
                    raise RuntimeError(f'An input changed during training; comparison was not started: {filename}')
            if plan['report'].exists():
                raise FileExistsError(f'Refusing to overwrite comparison report: {plan["report"]}')
            status['candidate_sha256']=file_digest(plan['exported'])
            progress=read_progress(plan['run'])
            completed=progress.get('completed_epochs',progress.get('epoch')) if progress else None
            status['completed_training_epochs']=completed
            status['training_schedule_completed']=isinstance(completed,int) and completed>=args.epochs
            run_stage('comparison')
            result=json.loads(plan['report'].read_text(encoding='utf-8'))
            status['comparison_selection']=result['selection']
            status['comparison_report_sha256']=file_digest(plan['report'])
        status.update(state='completed' if status['training_schedule_completed'] else 'completed_partial_training',
                      exit_code=0,finished_utc=utc_now(),system_awake_request=False)
        save()
        log(f'Comparison completed after {status["completed_training_epochs"]} of {args.epochs} requested epochs. Results are saved; no model was activated.')
        return 0
    except BaseException as error:
        code=130 if isinstance(error,KeyboardInterrupt) else 1
        status.update(state='interrupted' if code==130 else 'timed_out' if isinstance(error,TimeoutError) else 'failed',exit_code=code,
                      error=f'{type(error).__name__}: {error}',finished_utc=utc_now(),system_awake_request=False)
        save()
        log(status['error'])
        with (plan['folder']/'supervisor.log').open('a',encoding='utf-8') as stream:
            traceback.print_exc(file=stream)
        return code


def main():
    args=parse_args()
    try:
        return run_job(args)
    except FileExistsError as error:
        print(f'Refusing to overwrite an existing job folder: {error}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
