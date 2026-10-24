# CELL 3 — Điều phối nhóm (worker Mốc 4 ở cell phía trên KHÔNG đổi).
import os, sys, json, time, hashlib, zipfile, shutil, subprocess, signal, traceback
from pathlib import Path
from datetime import datetime, timezone
from IPython.display import display, FileLink

BASE = Path('/kaggle/working/M4_IP_jobs')
SESSION_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
GROUP_DIR = Path('/kaggle/working') / ('M4_IP_GROUP_'+SESSION_ID)
GROUP_DIR.mkdir(parents=True, exist_ok=False)
GROUP_STARTED = time.monotonic()
GROUP_END = GROUP_STARTED + GROUP_WALLTIME_SECONDS


def _digest_stream(file_obj):
    h=hashlib.sha256()
    while True:
        block=file_obj.read(1024*1024)
        if not block:break
        h.update(block)
    return h.hexdigest()


def _read_zip_json(z, member):
    if member not in z.namelist(): return None
    return json.loads(z.read(member).decode('utf-8'))


def _inspect_archive(path, arm, seed):
    """Kiểm tên job, checkpoint contract, và weights trước khi skip/resume."""
    with zipfile.ZipFile(path) as z:
        names=set(z.namelist())
        done=_read_zip_json(z,'job/completed.json')
        if done:
            if (done.get('stage'),done.get('arm'),done.get('seed'))!=('final',arm,seed):
                raise RuntimeError(f'Wrong completed identity in {path}')
            if done.get('complete') and done.get('c_updates') == TARGET_C_UPDATES:
                if done.get('worker_sha256')!=WORKER_EXPECTED_SHA256:
                    raise RuntimeError(f'Worker mismatched complete ZIP: {path}')
                for name, field in [('job/classifier.weights.h5','model_sha256'),
                                    ('job/contract.json','job_contract_sha256')]:
                    if name not in names:raise RuntimeError(f'Missing {name} in {path}')
                    with z.open(name) as f: digest=_digest_stream(f)
                    if digest!=done.get(field):raise RuntimeError(f'Checksum mismatch: {name} in {path}')
                return {'type':'COMPLETE','progress':8000,'model_sha256':done['model_sha256'],'path':str(path)}
        ptr=_read_zip_json(z,'job/state/latest.json')
        if ptr:
            meta_name=f"job/state/{ptr['name']}/meta.json"
            if meta_name not in names:raise RuntimeError(f'Incomplete state in {path}')
            metadata=z.read(meta_name)
            if hashlib.sha256(metadata).hexdigest()!=ptr['meta_sha256']:
                raise RuntimeError(f'Checkpoint metadata tampered: {path}')
            meta=json.loads(metadata)
            contract=meta['contract']
            if (contract.get('stage'),contract.get('arm'),contract.get('seed'))!=('final',arm,seed):
                raise RuntimeError(f'Wrong state identity in {path}')
            if contract.get('sha_worker')!=WORKER_EXPECTED_SHA256:
                raise RuntimeError(f'Worker changed; cannot resume: {path}')
            if (meta.get('c_updates') != ptr['c_updates']
                or meta.get('optimizer_clock') != 17*ptr['c_updates']):
                raise RuntimeError(f'Invalid Adam clock in {path}')
            for fn,expected in meta['weights_hashes'].items():
                member=f"job/state/{ptr['name']}/{fn}"
                if member not in names:raise RuntimeError(f'Missing state weight: {member}')
                with z.open(member) as f: digest=_digest_stream(f)
                if digest!=expected:raise RuntimeError(f'State weight corrupt: {member}')
            return {'type':'PAUSED','progress':ptr['c_updates'],
                    'meta_sha256':ptr['meta_sha256'],'path':str(path)}
        raise RuntimeError(f'No COMPLETE model or valid checkpoint in {path}')


def _discover_prior(arm,seed):
    filename=f'M4_IP_final_{arm}_seed{seed}_REPORT.zip'
    paths=sorted(Path('/kaggle/input').rglob(filename))
    if not paths:return None
    records=[_inspect_archive(p,arm,seed) for p in paths]
    completed=[r for r in records if r['type']=='COMPLETE']
    if completed:
        if len({r['model_sha256'] for r in completed})!=1:
            raise RuntimeError(f'Conflicting complete jobs {arm} {seed}; inspect inputs before training')
        return completed[0]
    progress=max(r['progress'] for r in records)
    tied=[r for r in records if r['progress']==progress]
    if len({r['meta_sha256'] for r in tied})!=1:
        raise RuntimeError(f'Conflicting equal-progress checkpoints {arm} {seed}')
    return tied[0]


def _import_resume_zip(path, target):
    # Copy a known checkpoint ZIP from read-only Input; never alter it.
    if target.exists():raise RuntimeError('Resume extraction target exists: '+str(target))
    target.mkdir(parents=True)
    with zipfile.ZipFile(path) as z:
        for m in z.infolist():
            parts=Path(m.filename).parts
            if len(parts)<2 or parts[0]!='job' or '..' in parts or Path(m.filename).is_absolute():
                raise RuntimeError('Unsafe resume ZIP path')
            rel=Path(*parts[1:])
            dest=target/rel
            if m.is_dir():dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(m) as r, dest.open('wb') as f:shutil.copyfileobj(r,f)
    if not (target/'state'/'latest.json').is_file():
        raise RuntimeError('Resume archive lacks state/latest.json')
    return target


def _bundle_job(folder, archive):
    # Giữ checkpoint active, complete model, metadata, log, không đóng gói checkpoint cũ.
    ptr=folder/'state'/'latest.json'
    active=json.loads(ptr.read_text())['name'] if ptr.is_file() else None
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(folder.rglob('*')):
            if not p.is_file():continue
            relative=p.relative_to(folder)
            if relative.parts[0]=='state' and len(relative.parts)>1:
                if relative.parts[1] not in ('latest.json',active):continue
            z.write(p,arcname=str(Path('job')/relative))
    return archive


def _completed_local(folder,arm,seed):
    p=folder/'completed.json'
    if not p.exists():return False
    r=json.loads(p.read_text())
    if (r.get('stage'),r.get('arm'),r.get('seed'))!=('final',arm,seed):
        raise RuntimeError('Local COMPLETE identity differs')
    if not (r.get('complete') and r.get('c_updates')==TARGET_C_UPDATES
            and r.get('worker_sha256')==WORKER_EXPECTED_SHA256):
        raise RuntimeError('Incomplete or mismatched local COMPLETE metadata')
    model=folder/'classifier.weights.h5'
    if not model.is_file():
        raise RuntimeError('Local complete classifier weights missing')
    with model.open('rb') as f:
        digest=_digest_stream(f)
    if digest!=r['model_sha256']:
        raise RuntimeError('Local complete classifier weights corrupt')
    return True


def _preflight():
    if not RUN_NOW:
        print('DRY STOP: RUN_NOW=False; no jobs started.');return False
    if not (Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir()):
        raise RuntimeError('This notebook must run on Kaggle')
    b=list(Path('/kaggle/input').rglob('dev_train_val.npz')) if B_RUN is None else [Path(B_RUN)/'development'/'dev_train_val.npz']
    b=[p for p in b if p.is_file() and p.parent.name=='development' and p.parent.parent.name=='KLTN_HyperGAMO_B_50e988fae260']
    if not b:
        raise RuntimeError('Missing Protocol B cache in THIS Kaggle account: Add accessible B Dataset under Input first. A private dataset owned by another account is not mounted automatically.')
    import shutil as s
    if s.which('nvidia-smi') is None:raise RuntimeError('GPU not enabled; select GPU accelerator before Save & Run All')
    gpu=subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,timeout=30)
    if gpu.returncode!=0 or 'GPU 0' not in gpu.stdout:raise RuntimeError('Kaggle GPU not detected: '+gpu.stderr)
    print('GPU available:',gpu.stdout.strip().splitlines()[0])
    print('Protocol B candidates:',len(b),'(worker pins hashes)')
    return True


def _one_job(arm,seed,max_seconds):
    folder=BASE/'final'/arm/f'seed_{seed}'
    local_done=_completed_local(folder,arm,seed) if folder.exists() else False
    archive=Path('/kaggle/working')/f'M4_IP_final_{arm}_seed{seed}_REPORT.zip'
    if local_done:
        if not archive.exists():_bundle_job(folder,archive)
        return {'arm':arm,'seed':seed,'status':'COMPLETE_LOCAL','archive':str(archive),'c_updates':8000}
    prior=_discover_prior(arm,seed)
    if prior and prior['type']=='COMPLETE':
        print('SKIP COMPLETE INPUT:',arm,seed,prior['path'],flush=True)
        return {'arm':arm,'seed':seed,'status':'COMPLETE_INPUT','archive_input':prior['path'],'c_updates':8000}
    folder.mkdir(parents=True,exist_ok=True)
    resume_dir=None
    if prior is not None and not (folder/'state'/'latest.json').exists():
        resume_dir=_import_resume_zip(prior['path'],GROUP_DIR/f'resume_{arm}_{seed}')
        print('RESUME INPUT',arm,seed,'from',prior['progress'],'C updates',flush=True)
    worker_file=BASE/'m4_training_worker.py'
    worker_file.parent.mkdir(parents=True,exist_ok=True)
    worker_file.write_text(WORKER_SOURCE,encoding='utf-8')
    actual_sha=hashlib.sha256(worker_file.read_bytes()).hexdigest()
    if actual_sha!=WORKER_EXPECTED_SHA256:raise RuntimeError('Original M4 worker SHA mismatch')
    cfg={'data_mode':'auto_real_B','b_run':B_RUN,'stage':'final','arm':arm,'seed':seed,
         'target_c_updates':TARGET_C_UPDATES,'checkpoint_every_c_updates':CHECKPOINT_EVERY_C_UPDATES,
         'log_every_loops':LOG_EVERY_LOOPS,'max_worker_seconds':int(max_seconds),
         'checkpoint_safety_seconds':CHECKPOINT_SAFETY_SECONDS,
         'resume_from':str(resume_dir) if resume_dir else None,'attempt_dir':str(folder),
         'worker_sha256':actual_sha,'start_monotonic':time.monotonic(),
         'deadline_monotonic':time.monotonic()+max_seconds+300}
    config_file=GROUP_DIR/f'config_{arm}_seed{seed}.json'
    config_file.write_text(json.dumps(cfg,indent=2))
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(GPU_INDEX);env['TF_CPP_MIN_LOG_LEVEL']='2'
    log=folder/'launch.log'
    print('START',arm,seed,'max_seconds=',max_seconds,flush=True)
    start=time.monotonic()
    launch_error=None
    with log.open('w',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,'-u',str(worker_file),str(config_file)],
                               cwd='/kaggle/working',env=env,stdout=stream,
                               stderr=subprocess.STDOUT,start_new_session=True)
        shown=''
        try:
            while child.poll() is None:
                if time.monotonic()-start>max_seconds+330:
                    raise RuntimeError('HARD TIMEOUT '+arm+' '+str(seed))
                time.sleep(20)
                lines=log.read_text(encoding='utf-8',errors='replace').splitlines()
                selections=[line for line in lines if line.startswith(('CHECKPOINT','TRAIN','RESTORED','JOB COMPLETE'))]
                if selections and selections[-1]!=shown:
                    shown=selections[-1];print(shown,flush=True)
        except BaseException as e:
            launch_error=str(e)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    report_file=folder/'worker_report.json'
    report=json.loads(report_file.read_text()) if report_file.exists() else {'status':'NO_REPORT'}
    _bundle_job(folder,archive)  # Preserve last committed checkpoint even on watchdog failure.
    status=report['status']
    print('FINISH',arm,seed,status,round(time.monotonic()-start,1),'sec,',archive,flush=True)
    if launch_error is not None or child.returncode!=0 or status not in ('COMPLETE','PAUSED'):
        print('LAST LOG:\n'+'\n'.join(log.read_text(errors='replace').splitlines()[-35:]),flush=True)
        return {'arm':arm,'seed':seed,'status':'FAILED','error':launch_error or report.get('error',status),
                'archive':str(archive),'c_updates':report.get('c_updates',0)}
    return {'arm':arm,'seed':seed,'status':status,'archive':str(archive),
            'c_updates':report.get('c_updates',8000 if status=='COMPLETE' else None)}


def _write_summary(rows,status,error=None):
    info={'session_id':SESSION_ID,'status':status,'start_utc':SESSION_ID,
          'wall_seconds':round(time.monotonic()-GROUP_STARTED,1),'group_seeds':GROUP_SEEDS,
          'group_arms':GROUP_ARMS,'target_C_updates_per_job':TARGET_C_UPDATES,
          'worker_sha256':WORKER_EXPECTED_SHA256,'test_opened':False,
          'planned':len(GROUP_SEEDS)*len(GROUP_ARMS),'jobs':rows,'error':error}
    path=Path('/kaggle/working')/f'M4_GROUP_{SESSION_ID}_SUMMARY.json'
    path.write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print('GROUP RESULT:',status,'| summary:',path,flush=True)
    print('COMPLETED:',sum(str(r['status']).startswith('COMPLETE') for r in rows),'/',len(rows),'attempted')
    print('ZIP from this session:')
    for r in rows:
        if r.get('archive') and Path(r['archive']).is_file():
            print('  ',r['archive']);display(FileLink(r['archive']))
    display(FileLink(str(path)))
    return info


def run_group():
    rows=[]
    try:
        if not _preflight():return _write_summary(rows,'NOT_STARTED')
        for seed in GROUP_SEEDS:
            for arm in GROUP_ARMS:
                remaining=GROUP_END-time.monotonic()-GROUP_SAFETY_SECONDS
                if remaining<MIN_NEW_JOB_SECONDS:
                    return _write_summary(rows,'STOPPED_SESSION_BUDGET',
                                          'Not enough time for another job. Save Output, mount previous ZIPs, rerun.')
                try:
                    result=_one_job(arm,seed,min(MAX_WORKER_SECONDS,int(remaining)))
                    rows.append(result)
                except BaseException as e:
                    rows.append({'arm':arm,'seed':seed,'status':'EXCEPTION','error':str(e)})
                    return _write_summary(rows,'STOPPED_ERROR',traceback.format_exc())
                if result['status'] in ('FAILED','PAUSED'):
                    return _write_summary(rows,'STOPPED_'+result['status'],
                                          'Save attached ZIP, inspect logs / resume in a later Kaggle version')
        return _write_summary(rows,'ALL_REQUESTED_COMPLETE')
    except BaseException:
        return _write_summary(rows,'STOPPED_PREFLIGHT_ERROR',traceback.format_exc())

if RUN_NOW:
    GROUP_RESULT=run_group()
else:
    print('RUN_NOW=False: configuration only; no training.')
