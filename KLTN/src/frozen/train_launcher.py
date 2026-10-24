# CELL 3 — Launcher chạy một worker trong Kaggle (watchdog / checkpoint / report ZIP).
import os, sys, json, time, hashlib, shutil, zipfile, subprocess, traceback
from pathlib import Path
from datetime import datetime, timezone

BASE = Path('/kaggle/working/M4_IP_jobs')

def _safe_zip_import(path, dest):
    path=Path(path)
    if dest.exists():
        raise RuntimeError('Đích giải nén resume đã tồn tại; không ghi đè: '+str(dest))
    dest.mkdir(parents=True)
    with zipfile.ZipFile(path) as z:
        files=z.infolist()
        for item in files:
            name=Path(item.filename)
            if name.is_absolute() or '..' in name.parts or len(name.parts)<2 or name.parts[0]!='job':
                raise RuntimeError('ZIP resume chứa đường dẫn không hợp lệ')
        for item in files:
            rel=Path(*Path(item.filename).parts[1:])
            if not rel.parts:continue
            target=dest/rel
            if item.is_dir():target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(item) as src, target.open('wb') as out:shutil.copyfileobj(src,out)
    if not (dest/'state'/'latest.json').is_file():
        raise RuntimeError('ZIP chưa chứa checkpoint. Không phải bản PAUSED của M4?')
    return dest

def _bundle_job(folder, archive):
    # ZIP bao gồm checkpoint đang active, history và cả C weights khi complete.
    latest_file=folder/'state'/'latest.json'
    active=None
    if latest_file.is_file():active=json.loads(latest_file.read_text())['name']
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(folder.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(folder)
            if rel.parts[0]=='state':
                if len(rel.parts)>1 and rel.parts[1]!='latest.json' and rel.parts[1]!=active:
                    continue
            z.write(p,arcname=str(Path('job')/rel))
    return archive

def run_one_job():
    assert ACTION=='train' and STAGE in ('development','final')
    assert ARM in ('original_gan','matched_ros','random_local','learned_local') and SEED in (42,43,44)
    assert TARGET_C_UPDATES==8000
    assert isinstance(CHECKPOINT_EVERY_C_UPDATES,int) and CHECKPOINT_EVERY_C_UPDATES>0 and CHECKPOINT_EVERY_C_UPDATES%2==0
    assert MAX_WORKER_SECONDS>CHECKPOINT_SAFETY_SECONDS+600
    if STAGE=='final' and not ALLOW_FINAL_FIT:
        raise RuntimeError('FINAL train bị khóa: chỉ mở ALLOW_FINAL_FIT=True sau khi khóa phương pháp.')
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Kaggle-only training')
    folder=BASE/STAGE/ARM/f'seed_{SEED}'
    folder.mkdir(parents=True,exist_ok=True)
    old_complete=folder/'completed.json'
    if old_complete.exists():
        rec=json.loads(old_complete.read_text())
        if rec.get('complete'):
            print('JOB ALREADY COMPLETE; không huấn luyện lại:',folder)
            return
    worker_file=BASE/'m4_training_worker.py'
    worker_file.parent.mkdir(parents=True,exist_ok=True)
    worker_file.write_text(WORKER_SOURCE,encoding='utf-8')
    actual_sha=hashlib.sha256(worker_file.read_bytes()).hexdigest()
    assert actual_sha==WORKER_EXPECTED_SHA256
    resume=RESUME_FROM
    if RESUME_FROM is not None and RESUME_ZIP is not None:
        raise RuntimeError('Chỉ chọn một trong RESUME_FROM và RESUME_ZIP')
    if RESUME_ZIP is not None:
        dest=Path('/kaggle/working')/f'M4_import_{STAGE}_{ARM}_{SEED}'
        resume=str(_safe_zip_import(RESUME_ZIP,dest))
    if resume is not None:
        resume=str(Path(resume).resolve())
        assert (Path(resume)/'state'/'latest.json').is_file(),'RESUME_FROM không có checkpoint'
    cfg={'data_mode':'auto_real_B','b_run':B_RUN,'stage':STAGE,'arm':ARM,'seed':SEED,
         'target_c_updates':TARGET_C_UPDATES,'checkpoint_every_c_updates':CHECKPOINT_EVERY_C_UPDATES,
         'log_every_loops':LOG_EVERY_LOOPS,'max_worker_seconds':MAX_WORKER_SECONDS,
         'checkpoint_safety_seconds':CHECKPOINT_SAFETY_SECONDS,
         'resume_from':resume,'attempt_dir':str(folder),'worker_sha256':actual_sha,
         'start_monotonic':time.monotonic(),'deadline_monotonic':time.monotonic()+MAX_WORKER_SECONDS+300}
    cfg_file=BASE/f'job_{STAGE}_{ARM}_{SEED}_config.json'
    cfg_file.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(GPU_INDEX);env['TF_CPP_MIN_LOG_LEVEL']='2'
    log=folder/'launch.log'
    begin=time.monotonic()
    print('START job:',STAGE,ARM,SEED,'|',folder)
    with log.open('w',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,'-u',str(worker_file),str(cfg_file)],
                               cwd='/kaggle/working',env=env,
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        report=None
        try:
            last_line=''
            while child.poll() is None:
                if time.monotonic()-begin>MAX_WORKER_SECONDS+330:
                    import signal
                    os.killpg(child.pid, signal.SIGKILL)
                    raise RuntimeError('Worker vượt hard watchdog; xem checkpoint/log.')
                time.sleep(20)
                lines=log.read_text(encoding='utf-8',errors='replace').splitlines()
                newlines=[l for l in lines if l.startswith(('CHECKPOINT','TRAIN','JOB COMPLETE','RESTORED'))]
                if newlines and newlines[-1]!=last_line:
                    print(newlines[-1],flush=True);last_line=newlines[-1]
            report=json.loads((folder/'worker_report.json').read_text()) if (folder/'worker_report.json').exists() else None
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=12)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    status=report['status'] if report else f'NO_REPORT_EXIT_{child.returncode}'
    archive=Path('/kaggle/working')/f'M4_IP_{STAGE}_{ARM}_seed{SEED}_REPORT.zip'
    _bundle_job(folder,archive)
    print('RESULT:',status,'| elapsed:',round(time.monotonic()-begin,1),'s')
    print('LOG:',log,'\nREPORT ZIP:',archive)
    try:
        from IPython.display import display,FileLink
        display(FileLink(str(archive)))
    except Exception:pass
    if child.returncode!=0 or status=='FAILED':
        print('\nLAST 35 LOG LINES:\n'+'\n'.join(log.read_text(errors='replace').splitlines()[-35:]))
        raise RuntimeError('Job FAILED; giữ ZIP + checkpoint, không tự chạy lại.')
    if status=='PAUSED':
        print('Job đã PAUSED an toàn; mount output/ZIP của job này để RESUME, không reset optimizer.')
    elif status=='COMPLETE':
        print('ĐỦ 8.000 C updates cho job. C weights đã lưu; kết quả DEV nếu áp dụng đã được xuất.')
    else:raise RuntimeError('Worker chưa xác nhận COMPLETE/PAUSED: '+str(status))
    return report

if ACTION=='train':
    if RUN_NOW:
        job_report=run_one_job()
    else:
        print('CHƯA CHẠY: đổi RUN_NOW=True ở Cell 1 khi đã sẵn sàng chạy một job trên Kaggle.')
else:
    print('ACTION=evaluate_final; Cell 3 không huấn luyện.')