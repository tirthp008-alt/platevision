"""Subprocess orchestration checks with fake engines; never launch training."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_yolo11_extended_job as job


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    real_stop=job.stop_child
    # Fake PIDs must never reach the operating system's taskkill command.
    monkeypatch.setattr(job,'stop_child',lambda process,force=False:real_stop(process,force=force,windows=False))
    started=datetime(2030,1,1,tzinfo=timezone.utc)
    args=job.parse_args(['--device','xpu:0','--initial-weights',str(tmp_path/'old/weights/best.pt'),
        '--data',str(tmp_path/'new-data/data.yaml'),'--tag','fixture',
        '--deadline-utc',(started+timedelta(hours=5)).isoformat(),'--report-test'])
    plan=job.job_plan(args,tmp_path)
    for path in plan['required']:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'unchanged input')
    for split in ('val','test'):
        images=plan['frozen']/split/'images'; images.mkdir(parents=True)
        (images/'scene.jpg').write_bytes(b'fixture')
    clock=[started]
    events=[]

    @contextmanager
    def awake():
        events.append('awake')
        try:
            yield True
        finally:
            events.append('restored')

    def status():
        return json.loads((plan['folder']/'status.json').read_text())

    return args,plan,clock,events,awake,status


class FakeProcess:
    def __init__(self,name,events,complete,pid):
        self.name,self.events,self.complete,self.pid=name,events,complete,pid
        self.returncode=None
        self.terminated=False
        self.killed=False

    def wait(self,timeout=None):
        if self.returncode is None:
            self.events.append(f'wait_{self.name}')
            self.returncode=self.complete(timeout)
            self.events.append(f'exit_{self.name}')
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated=True
        self.returncode=-15

    def kill(self):
        self.killed=True
        self.returncode=-9


def factory(prepared,training_code=0,comparison_code=0,completed_epochs=None,timeout_stage=None,interrupt=False):
    args,plan,clock,events,awake,status=prepared
    processes=[]

    def popen(command,**kwargs):
        name='training' if command[1].endswith('train_nano_candidates.py') else 'comparison'
        if name=='comparison':
            assert processes[0].returncode==0
        events.append(f'start_{name}')
        kwargs['stdout'].write(f'{name} persistent log\n')
        kwargs['stdout'].flush()

        def complete(timeout):
            assert status()['stages'][name]['pid']==len(processes)+100
            if interrupt:
                raise KeyboardInterrupt()
            if timeout_stage==name:
                clock[0]=(plan['training_cutoff'] if name=='training' else plan['deadline'])+timedelta(seconds=1)
                raise subprocess.TimeoutExpired(command,timeout)
            if name=='training' and training_code==0:
                plan['exported'].parent.mkdir(parents=True)
                plan['exported'].write_bytes(b'new export')
                (plan['run']/'progress.json').write_text(json.dumps({
                    'epoch':completed_epochs if completed_epochs is not None else args.epochs,
                    'completed_epochs':completed_epochs if completed_epochs is not None else args.epochs,
                    'epochs':args.epochs}))
            if name=='comparison' and comparison_code==0:
                plan['report'].write_text(json.dumps({'selection':{'recommended_for_review':None,'default_model_changed':False}}))
            return training_code if name=='training' else comparison_code

        process=FakeProcess(name,events,complete,len(processes)+101)
        processes.append(process)
        return process

    return popen,processes


def test_commands_use_selected_interpreter_frozen_evaluation_and_requested_schedule(prepared):
    args,plan,*_=prepared
    assert '.venv-train-xpu' in plan['train'][0]
    assert Path(plan['train'][0]).name in {'python.exe','python'}
    for key,value in [('--epochs','20'),('--freeze','0'),('--warmup-epochs','3'),('--close-mosaic','5')]:
        assert plan['train'][plan['train'].index(key)+1]==value
    assert plan['compare'][plan['compare'].index('--dataset')+1]==str(plan['frozen'])
    assert '--report-test' in plan['compare']
    assert plan['compare'][plan['compare'].index('--runs')+1]=='30'
    assert plan['compare'][plan['compare'].index('--warmup')+1]=='5'
    assert '--initial-weights' not in plan['resume'] and '--resume' in plan['resume']
    args.device='cpu'; args.epochs=100
    other=job.job_plan(args,plan['root'])
    assert '.venv-train-xpu' not in other['train'][0]
    assert other['train'][other['train'].index('--close-mosaic')+1]=='10'


def test_explicit_reproduction_paths_and_cpu_comparison_are_preserved(tmp_path):
    args=job.parse_args(['--device','cpu','--initial-weights',str(tmp_path/'best.pt'),
        '--data',str(tmp_path/'training/data.yaml'),'--tag','portable',
        '--deadline-utc','2030-01-01T12:00:00Z',
        '--validation-dataset',str(tmp_path/'frozen'),
        '--reference-model',str(tmp_path/'reference.onnx'),
        '--prior-model',str(tmp_path/'prior.onnx'),
        '--training-python',str(tmp_path/'training-python'),
        '--comparison-python',str(tmp_path/'comparison-python'),
        '--comparison-device','CPU'])
    plan=job.job_plan(args,tmp_path)
    assert plan['train'][0]==str(tmp_path/'training-python')
    assert plan['compare'][0]==str(tmp_path/'comparison-python')
    assert plan['frozen']==tmp_path/'frozen'
    assert plan['green']==tmp_path/'reference.onnx'
    assert plan['prior']==tmp_path/'prior.onnx'
    assert plan['compare'][plan['compare'].index('--device')+1]=='CPU'


def test_success_waits_for_training_and_persists_reviewable_status(prepared):
    args,plan,clock,events,awake,status=prepared
    popen,processes=factory(prepared)
    assert job.run_job(args,plan['root'],popen,awake,lambda:clock[0])==0
    result=status()
    assert result['state']=='completed' and result['completed_training_epochs']==20
    assert result['default_model_changed'] is False
    assert events.index('exit_training')<events.index('start_comparison')
    assert events[-1]=='restored'
    assert [result['stages'][key]['exit_code'] for key in ('training','comparison')]==[0,0]
    assert all(result['stages'][key]['started_utc'] and result['stages'][key]['finished_utc'] for key in ('training','comparison'))
    assert (plan['folder']/'training.log').read_text()=='training persistent log\n'
    assert plan['initial'].read_bytes()==b'unchanged input'
    original=(plan['folder']/'status.json').read_bytes()
    with pytest.raises(FileExistsError):
        job.run_job(args,plan['root'],popen,awake,lambda:clock[0])
    assert (plan['folder']/'status.json').read_bytes()==original


def test_graceful_short_run_is_reported_as_partial_even_when_comparison_passes(prepared):
    args,plan,clock,events,awake,status=prepared
    popen,_=factory(prepared,completed_epochs=7)
    assert job.run_job(args,plan['root'],popen,awake,lambda:clock[0])==0
    assert status()['state']=='completed_partial_training'
    assert status()['completed_training_epochs']==7
    assert status()['training_schedule_completed'] is False


def test_training_failure_does_not_start_comparison_or_retry(prepared):
    args,plan,clock,events,awake,status=prepared
    popen,processes=factory(prepared,training_code=7)
    assert job.run_job(args,plan['root'],popen,awake,lambda:clock[0])==1
    assert len(processes)==1 and status()['stages']['training']['exit_code']==7
    assert status()['stages']['comparison']['state']=='pending'
    assert events[-1]=='restored'


@pytest.mark.parametrize('stage',['training','comparison'])
def test_deadline_kills_only_owned_child_and_preserves_partial_results(prepared,stage):
    args,plan,clock,events,awake,status=prepared
    popen,processes=factory(prepared,timeout_stage=stage)
    assert job.run_job(args,plan['root'],popen,awake,lambda:clock[0])==1
    assert processes[-1].killed and not processes[-1].terminated
    assert status()['state']=='timed_out'
    assert status()['stages'][stage]['state']=='timed_out'
    assert status()['stages'][stage]['exit_code']==-9
    assert status()['stages'][stage]['finished_utc']
    assert events[-1]=='restored'
    assert len(processes)==(1 if stage=='training' else 2)
    if stage=='comparison':
        assert plan['exported'].read_bytes()==b'new export'


def test_late_launch_creates_failure_record_without_starting_a_child(prepared):
    args,plan,clock,events,awake,status=prepared
    clock[0]=plan['deadline']
    popen,processes=factory(prepared)
    assert job.run_job(args,plan['root'],popen,awake,lambda:clock[0])==1
    assert processes==[] and events==[] and status()['state']=='timed_out'


def test_interrupt_reaps_child_and_releases_awake_request(prepared):
    args,plan,clock,events,awake,status=prepared
    popen,processes=factory(prepared,interrupt=True)
    assert job.run_job(args,plan['root'],popen,awake,lambda:clock[0])==130
    assert processes[0].terminated and status()['state']=='interrupted'
    assert events[-1]=='restored'


def test_windows_sleep_request_uses_no_display_flag_and_restores_previous_state():
    calls=[]
    def setter(value):
        calls.append(value)
        return job.ES_CONTINUOUS
    with pytest.raises(RuntimeError,match='fixture failure'):
        with job.keep_system_awake(setter) as supported:
            assert supported
            raise RuntimeError('fixture failure')
    assert calls==[job.ES_CONTINUOUS|job.ES_SYSTEM_REQUIRED,job.ES_CONTINUOUS]


def test_windows_timeout_stops_owned_venv_redirector_and_its_worker_tree():
    process=FakeProcess('training',[],lambda _:0,12345)
    calls=[]
    def taskkill(command,**kwargs):
        assert process.poll() is None
        calls.append((command,kwargs))
        process.returncode=1
        return type('Result',(),{'returncode':0,'stdout':b'terminated fixture tree'})()
    assert job.stop_child(process,force=True,windows=True,run_command=taskkill)==1
    assert calls[0][0]==['taskkill.exe','/PID','12345','/T','/F']
    assert calls[0][1]['timeout']==15
    assert calls[0][1]['creationflags']==getattr(subprocess,'CREATE_NO_WINDOW',0)
    # An exited Popen handle must never target a possibly reused PID.
    assert job.stop_child(process,force=True,windows=True,run_command=taskkill)==1
    assert len(calls)==1
