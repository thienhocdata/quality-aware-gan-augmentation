# Portable launcher. A fresh process keeps TensorFlow isolated from old kernels.
import datetime as _m4_dt
import json as _m4_json
import os as _m4_os
from pathlib import Path as _M4Path
import signal as _m4_signal
import subprocess as _m4_subprocess
import sys as _m4_sys
import time as _m4_time
import uuid as _m4_uuid
import zipfile as _m4_zipfile


def run_m4_test():
    started=_m4_time.monotonic()
    if DEVICE not in ('auto','gpu','cpu'):
        raise ValueError('DEVICE must be auto, gpu or cpu')
    if type(GPU_INDEX) is not int or GPU_INDEX<0:
        raise ValueError('GPU_INDEX must be a nonnegative integer')
    if type(TIME_LIMIT_SECONDS) is not int or not 60<=TIME_LIMIT_SECONDS<=1800:
        raise ValueError('The total test deadline must be 60..1800 seconds')
    for key,value in (('MODEL_SOURCE',MODEL_SOURCE),('B_RUN',B_RUN),('HSI_PATH',HSI_PATH)):
        if value is not None and not isinstance(value,str):raise ValueError(key+' must be None or a path string')
    stamp=_m4_dt.datetime.now(_m4_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    base=_M4Path('/kaggle/working') if _M4Path('/kaggle/working').is_dir() else _M4Path.cwd()
    out=base/('M4_IP_TEST_'+stamp+'_'+_m4_uuid.uuid4().hex[:8]);out.mkdir(exist_ok=False)
    roots=[str(_M4Path.cwd())]
    if _M4Path('/kaggle/input').is_dir():roots.append('/kaggle/input')
    config={'model_source':MODEL_SOURCE,'b_run':B_RUN,'hsi_path':HSI_PATH,'device':DEVICE,
            'gpu_batch_size':32,'cpu_batch_size':16,'output_dir':str(out),
            'search_roots':roots,'start_monotonic':started,
            'deadline_monotonic':started+TIME_LIMIT_SECONDS}
    (out/'config.json').write_text(_m4_json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    worker=out/'test_worker.py';worker.write_text(TEST_WORKER_SOURCE,encoding='utf-8')
    log=out/'test_worker.log'
    env=dict(_m4_os.environ)
    env.update(TF_USE_LEGACY_KERAS='1',PYTHONUNBUFFERED='1',
               CUDA_VISIBLE_DEVICES='-1' if DEVICE=='cpu' else str(GPU_INDEX))
    process=None;offset=0;forced_timeout=False;terminal_error=None
    print('M4 INFERENCE-ONLY TEST:',out,flush=True)
    print('Twelve fixed classifiers; no G/D, no optimizer, no training. Total deadline:',TIME_LIMIT_SECONDS,'s',flush=True)
    try:
        with log.open('w',encoding='utf-8') as stream:
            process=_m4_subprocess.Popen([_m4_sys.executable,'-u',str(worker),str(out/'config.json')],
                stdout=stream,stderr=_m4_subprocess.STDOUT,env=env,
                start_new_session=(_m4_os.name!='nt'))
            last_heartbeat=_m4_time.monotonic()
            while process.poll() is None:
                if _m4_time.monotonic()>=config['deadline_monotonic']:
                    forced_timeout=True;break
                if log.is_file():
                    with log.open('r',encoding='utf-8',errors='replace') as reader:
                        reader.seek(offset);chunk=reader.read(65536);offset=reader.tell()
                    if chunk:print(chunk,end='',flush=True)
                if _m4_time.monotonic()-last_heartbeat>=45:
                    print(f'Test still running; total {_m4_time.monotonic()-started:.0f}s. No training.',flush=True)
                    last_heartbeat=_m4_time.monotonic()
                _m4_time.sleep(.5)
    except BaseException as error:
        terminal_error=str(error)
    finally:
        if process is not None and process.poll() is None:
            if _m4_os.name=='nt':process.kill()
            else:
                try:_m4_os.killpg(process.pid,_m4_signal.SIGKILL)
                except ProcessLookupError:pass
            process.wait(timeout=10)
        if log.is_file():
            with log.open('r',encoding='utf-8',errors='replace') as reader:
                reader.seek(offset);chunk=reader.read()
            if chunk:print(chunk,end='',flush=True)
        manifest_path=out/'manifest.json'
        manifest=_m4_json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {'status':'FAILED','complete':False}
        if forced_timeout:
            manifest.update(status='TIMEOUT',complete=False,error='Parent watchdog reached the shared deadline; partial outputs preserved')
        if terminal_error:manifest.update(status='FAILED',complete=False,error=terminal_error)
        if process is None or process.returncode!=0:
            manifest.update(complete=False)
            if manifest.get('status')=='COMPLETE':manifest['status']='FAILED'
        manifest['worker_exit_code']=None if process is None else process.returncode
        manifest['total_elapsed_seconds']=round(_m4_time.monotonic()-started,2)
        manifest_path.write_text(_m4_json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        report=out.with_name(out.name+'_REPORT.zip')
        whitelist=('manifest.json','evaluation_registry.json','config.json','test_worker.py','test_worker.log',
                   'RESULTS.md','test_by_seed.csv','test_per_class_by_seed.csv',
                   'test_summary_mean_sd.csv','test_per_class_mean_sd.csv','paired_differences.csv')
        with _m4_zipfile.ZipFile(report,'w',_m4_zipfile.ZIP_DEFLATED) as archive:
            for name in whitelist:
                path=out/name
                if path.is_file():archive.write(path,out.name+'/'+name)
            results=out/'test_results'
            if results.is_dir():
                for path in sorted(results.rglob('*')):
                    if path.is_file() and path.name in ('metrics.json','per_class.csv','predictions.npz','completed.json'):
                        archive.write(path,out.name+'/'+path.relative_to(out).as_posix())
        print('\nTEST STATUS:',manifest['status'],'| elapsed',manifest['total_elapsed_seconds'],'s',flush=True)
        print('REPORT ZIP:',report,flush=True)
        try:
            from IPython.display import display,FileLink,Markdown
            if (out/'RESULTS.md').is_file():display(Markdown((out/'RESULTS.md').read_text(encoding='utf-8')))
            display(FileLink(str(report)))
        except ImportError:pass
    if manifest.get('complete') is not True:
        raise RuntimeError('Test not complete; report ZIP was saved. '+manifest.get('error','See test_worker.log'))
    if manifest.get('completed_model_count')!=12:
        raise RuntimeError('Incomplete twelve-model registry; do not interpret it as the final comparison')
    return manifest


m4_test_result=run_m4_test()
