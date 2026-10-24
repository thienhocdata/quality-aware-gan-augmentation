# This cell runs only on Kaggle. It imports no TensorFlow in the notebook kernel.
import datetime as _dt
import hashlib as _hashlib
import json as _json
import os as _os
from pathlib import Path as _Path
import signal as _signal
import shutil as _shutil
import subprocess as _subprocess
import sys as _sys
import time as _time
import traceback as _traceback
import uuid as _uuid
import zipfile as _zipfile


def launch_smoke():
    # Deadline starts before discovery, hashes, TensorFlow import and compile.
    started = _time.monotonic()
    if not _sys.platform.startswith('linux') or not _Path('/kaggle/working').is_dir():
        raise RuntimeError('Notebook này chỉ chạy smoke trên Kaggle; không chạy ở máy cá nhân.')
    if DATA_MODE not in ('auto_real_B','fixture'):
        raise ValueError('DATA_MODE phải là auto_real_B hoặc fixture')
    if not isinstance(TIME_LIMIT_SECONDS,int) or not (1 <= TIME_LIMIT_SECONDS <= 1800):
        raise ValueError('TIME_LIMIT_SECONDS phải trong 1..1800; tối đa 30 phút.')
    if not isinstance(GPU_INDEX,int) or GPU_INDEX < 0:
        raise ValueError('GPU_INDEX phải là số nguyên không âm')
    if not isinstance(SEED,int) or SEED < 0:
        raise ValueError('SEED phải là số nguyên không âm')
    if B_RUN is not None and not isinstance(B_RUN,str):
        raise ValueError('B_RUN phải là None hoặc đường dẫn dạng chuỗi')
    deadline = started + TIME_LIMIT_SECONDS
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    attempt_id = stamp+'_'+_uuid.uuid4().hex[:8]
    folder = _Path('/kaggle/working')/('V2_IP_SMOKE_'+attempt_id)
    folder.mkdir(exist_ok=False)
    worker_path = folder/'worker.py'
    worker_path.write_text(WORKER_SOURCE,encoding='utf-8')
    worker_hash = _hashlib.sha256(worker_path.read_bytes()).hexdigest()
    config = {'schema':2,'attempt_id':attempt_id,'attempt_dir':str(folder),
              'seed':SEED,'data_mode':DATA_MODE,'b_run':B_RUN,
              'gpu_index':GPU_INDEX,'time_limit_seconds':TIME_LIMIT_SECONDS,
              'start_monotonic':started,'deadline_monotonic':deadline,
              'worker_sha256':worker_hash}
    config_hash = _hashlib.sha256(_json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    config['config_sha256'] = config_hash
    config_path = folder/'config.json'
    config_path.write_text(_json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def atomic_result(value):
        temp=folder/'result.json.tmp'
        temp.write_text(_json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        temp.replace(folder/'result.json')
    result={'attempt_id':attempt_id,'status':'RUNNING','pass':False,
            'config_sha256':config_hash,'worker_sha256':worker_hash,
            'limit_seconds':TIME_LIMIT_SECONDS,'data_mode':DATA_MODE,
            'scope':'single V2 integration smoke only',
            'accuracy_computed':False,'test_opened':False,'full_training_allowed':False,
            'watchdog':'independent GNU timeout plus notebook-parent deadline',
            'cleanup_errors':[]}
    atomic_result(result)
    process=None
    offset=0
    def read_log_tail():
        nonlocal offset
        log_path=folder/'worker.log'
        if log_path.exists():
            with log_path.open('r',encoding='utf-8',errors='replace') as log:
                log.seek(offset)
                content=log.read()
                offset=log.tell()
            if content:
                print(content,end='',flush=True)
    def kill_worker():
        if process is not None and process.poll() is None:
            # A separate process group prevents orphaned GPU work after timeout
            # or an ordinary notebook-cell interrupt. No extra training grace.
            try:
                _os.killpg(process.pid,_signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                result['cleanup_errors'].append(str(error))
            try:
                process.wait(timeout=5)
            except (_subprocess.TimeoutExpired,OSError) as error:
                result['cleanup_errors'].append(str(error))
    try:
        print('V2 smoke:',folder,flush=True)
        print(f'Chỉ GPU vật lý {GPU_INDEX}; deadline tổng {TIME_LIMIT_SECONDS}s; không tính accuracy.',flush=True)
        if _time.monotonic() >= deadline:
            raise TimeoutError('Hết thời gian trước khi khởi động worker')
        timeout_executable=_shutil.which('timeout')
        if timeout_executable is None:
            raise RuntimeError('Thiếu GNU timeout để giám sát độc lập; không khởi động GPU worker.')
        env=dict(_os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(GPU_INDEX),PYTHONUNBUFFERED='1',
                   TF_USE_LEGACY_KERAS='1',PYTHONHASHSEED=str(SEED),LC_ALL='C')
        with (folder/'worker.log').open('w',encoding='utf-8') as output:
            # Independent OS monitor still enforces the worker cutoff if the
            # notebook kernel is interrupted/killed or its UI output blocks.
            # Reserve 0.5 s for process startup/scheduling; never extend 30 min.
            remaining=deadline-_time.monotonic()-.5
            if remaining <= 0:
                raise TimeoutError('Hết thời gian trước khi khởi động watchdog')
            process=_subprocess.Popen([timeout_executable,'--verbose','--signal=KILL',f'{remaining:.6f}s',
                                       _sys.executable,'-u',str(worker_path),str(config_path)],
                                      stdout=output,stderr=_subprocess.STDOUT,env=env,
                                      start_new_session=True)
            last_heartbeat=started
            while True:
                now=_time.monotonic()
                if now >= deadline:
                    raise TimeoutError('Đã chạm deadline tổng; dừng worker GPU. TIMEOUT không được coi là PASS.')
                if process.poll() is not None:
                    break
                read_log_tail()
                if now-last_heartbeat >= 45:
                    print(f'Worker còn chạy | đã dùng {now-started:.0f}/{TIME_LIMIT_SECONDS}s',flush=True)
                    last_heartbeat=now
                _time.sleep(min(1.,max(.01,deadline-_time.monotonic())))
        read_log_tail()
        if _time.monotonic() >= deadline:
            raise TimeoutError('Worker kết thúc nhưng vượt deadline tổng; không chấp nhận PASS.')
        if process.returncode in (124,137,-9) and 'sending signal KILL' in (folder/'worker.log').read_text(encoding='utf-8',errors='replace'):
            raise TimeoutError('Watchdog hệ điều hành đã dừng worker khi hết thời gian.')
        report_path=folder/'smoke_report.json'
        if not report_path.is_file():
            raise RuntimeError('Worker không ghi được smoke_report.json; đọc worker.log')
        report=_json.loads(report_path.read_text(encoding='utf-8'))
        result['worker_report']=report
        if report.get('status') == 'TIMEOUT':
            raise TimeoutError(report.get('error','Worker reached its deadline'))
        if process.returncode != 0 or not report.get('pass') or report.get('status') != 'PASS':
            raise RuntimeError(f"Worker chưa PASS: {report.get('error',report.get('phase'))}")
        if any(report.get(k) != v for k,v in (('attempt_id',attempt_id),('config_sha256',config_hash),('worker_sha256',worker_hash))):
            raise RuntimeError('Report không khớp attempt/config/source hiện tại')
        if report.get('checkpoint',{}).get('optimizer_iterations') != 34 or len(report.get('routes',[])) != 34:
            raise RuntimeError('Report thiếu đủ 34 cập nhật hoặc checkpoint roundtrip')
        if DATA_MODE == 'auto_real_B' and not report.get('input',{}).get('real_B_cache_verified'):
            raise RuntimeError('Real B mode chưa xác thực cache nguồn')
        if _time.monotonic() >= deadline:
            raise TimeoutError('Đã vượt deadline trước khi chấp nhận PASS.')
        result.update(status='PASS',**{'pass':True})
    except BaseException as error:
        kill_worker()
        result.update(status='TIMEOUT' if isinstance(error,TimeoutError) else 'ABORTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
                      **{'pass':False},error=str(error),traceback=_traceback.format_exc())
        report_path=folder/'smoke_report.json'
        if report_path.is_file():
            try:
                result['worker_report']=_json.loads(report_path.read_text(encoding='utf-8'))
            except (OSError,ValueError):
                pass
    finally:
        kill_worker()
        read_log_tail()
        result['elapsed_seconds']=round(_time.monotonic()-started,3)
        result['worker_exit_code']=None if process is None else process.returncode
        if result['cleanup_errors']:
            result.update(status='FAILED',**{'pass':False},error='Không xác minh được cleanup worker; xem cleanup_errors.')
        atomic_result(result)
        # Small report only. No model weights / checkpoint bundle is exported.
        archive_path=folder.with_name(folder.name+'_REPORT.zip')
        with _zipfile.ZipFile(archive_path,'w',_zipfile.ZIP_DEFLATED) as archive:
            for name in ('result.json','smoke_report.json','config.json','worker.log','worker.py'):
                path=folder/name
                if path.is_file():
                    archive.write(path,folder.name+'/'+name)
        print('\nKẾT THÚC:',result['status'],'|',result['elapsed_seconds'],'s',flush=True)
        print('Báo cáo chính:',folder/'result.json',flush=True)
        print('ZIP NHỎ:',archive_path,flush=True)
        from IPython.display import display,FileLink
        display(FileLink(str(archive_path)))
    if not result['pass']:
        raise RuntimeError(f"Smoke {result['status']}; đã lưu log/báo cáo. Không tự chạy lại. {result.get('error','')}")
    return result


smoke_result = launch_smoke()
