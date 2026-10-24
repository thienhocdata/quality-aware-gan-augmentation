# Kaggle-only Mốc 2 launcher. TensorFlow is imported only inside fresh workers.
import datetime as _m2_dt
import hashlib as _m2_hashlib
import json as _m2_json
import os as _m2_os
from pathlib import Path as _M2Path
import shutil as _m2_shutil
import signal as _m2_signal
import subprocess as _m2_subprocess
import sys as _m2_sys
import time as _m2_time
import traceback as _m2_traceback
import uuid as _m2_uuid
import zipfile as _m2_zipfile


_M2_ARMS = ('original_gan', 'matched_ros', 'random_local', 'learned_local')
_M2_GAN_ARMS = ('original_gan', 'learned_local')
_M2_ANCHOR_ARMS = ('matched_ros', 'random_local', 'learned_local')
_M2_LOCAL_ARMS = ('random_local', 'learned_local')
_M2_GATE_KEYS = ('freeze', 'trace_consumption', 'dropout_C_mask_keys',
                 'iteration_skips', 'checkpoint_roundtrip')


def _m2_json_sha(value):
    payload = _m2_json.dumps(value, sort_keys=True, separators=(',', ':'),
                            ensure_ascii=True, allow_nan=False).encode('utf-8')
    return _m2_hashlib.sha256(payload).hexdigest()


def _m2_require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _m2_is_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _m2_read_json(path):
    value = _m2_json.loads(_M2Path(path).read_text(encoding='utf-8'))
    _m2_require(isinstance(value, dict), f'JSON không phải object: {path}')
    return value


def _m2_atomic_json(path, value):
    path = _M2Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(_m2_json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _m2_verify_arm(folder, config):
    """Validate the current arm's report and actual consumed-record hashes."""
    report_path = folder / 'arm_report.json'
    _m2_require(report_path.is_file(), f'Thiếu arm_report.json: {folder.name}')
    report = _m2_read_json(report_path)
    for key in ('attempt_id', 'arm', 'config_sha256', 'worker_sha256'):
        _m2_require(report.get(key) == config[key], f'Report khác {key}: {folder.name}')
    if report.get('status') == 'TIMEOUT':
        raise TimeoutError(report.get('error', f'Worker {folder.name} hết thời gian'))
    _m2_require(report.get('status') == 'PASS' and report.get('pass') is True,
                f"Worker {folder.name} chưa PASS: {report.get('error', report.get('phase'))}")
    _m2_require(report.get('input', {}).get('real_B_cache_verified') is True,
                f'Worker {folder.name} chưa xác thực cache B gốc')
    for flag in ('accuracy_computed', 'validation_arrays_loaded',
                 'test_arrays_loaded', 'full_training_allowed'):
        _m2_require(report.get(flag) is False, f'Cờ phạm vi thiếu/sai: {folder.name}/{flag}')
    clocks = report.get('clocks', {})
    expected_calls = 68 if config['arm'] in _M2_GAN_ARMS else 4
    _m2_require(clocks.get('C_updates') == 4 and clocks.get('final_clock') == 68
                and clocks.get('real_optimizer_calls') == expected_calls,
                f'Clock/cập nhật không đúng Mốc 2: {folder.name}')
    gates = report.get('gates', {})
    for key in _M2_GATE_KEYS:
        _m2_require(gates.get(key) is True, f'Gate chưa PASS: {folder.name}/{key}')
    initialization = report.get('initialization', {})
    init_keys = ['C_trainable_sha256', 'C_nontrainable_sha256', 'C_adam_slots_sha256']
    if config['arm'] in _M2_GAN_ARMS:
        init_keys.extend(('D_trainable_sha256', 'G_trainable_sha256',
                          'head_dense_trainable_sha256'))
    for key in init_keys:
        _m2_require(_m2_is_sha(initialization.get(key)),
                    f'Thiếu hash khởi tạo: {folder.name}/{key}')

    consumed_path = folder / 'consumed_trace.json'
    _m2_require(consumed_path.is_file(), f'Thiếu consumed_trace.json: {folder.name}')
    consumed = _m2_read_json(consumed_path)
    _m2_require(consumed.get('attempt_id') == config['attempt_id']
                and consumed.get('arm') == config['arm'],
                f'Consumed trace khác attempt/arm: {folder.name}')
    trace = report.get('trace', {})
    _m2_require(trace.get('consumed_loops') == 2,
                f'Worker không tiêu thụ đúng hai loop: {folder.name}')
    for records_key, hash_key, applicable in (
        ('common_c', 'common_C', True),
        ('anchors', 'anchors_C', config['arm'] in _M2_ANCHOR_ARMS),
        ('local_geometry', 'local_geometry', config['arm'] in _M2_LOCAL_ARMS),
        ('common_gan', 'GAN_private', config['arm'] in _M2_GAN_ARMS),
    ):
        records = consumed.get(records_key)
        if applicable:
            _m2_require(isinstance(records, list) and len(records) > 0,
                        f'Thiếu bản ghi đã tiêu thụ: {folder.name}/{records_key}')
            if records_key == 'common_c':
                _m2_require(len(records) == 4,
                            f'Phải có bốn bản ghi C_real/C_fake: {folder.name}')
            if records_key == 'local_geometry':
                _m2_require(len(records) == 2,
                            f'Phải có hai bản ghi local geometry: {folder.name}')
            _m2_require(_m2_is_sha(trace.get(hash_key))
                        and trace[hash_key] == _m2_json_sha(records),
                        f'Hash không khớp actual consumed records: {folder.name}/{records_key}')
        else:
            _m2_require(records in (None, []) and trace.get(hash_key) is None,
                        f'Trace của nhánh không áp dụng phải rỗng: {folder.name}/{records_key}')
    return report


def _m2_verify_pairs(reports):
    _m2_require(set(reports) == set(_M2_ARMS), 'Chưa đủ bốn report Mốc 2')
    verified = {}

    def same(section, key, arms, label):
        values = [reports[arm][section][key] for arm in arms]
        _m2_require(all(value == values[0] for value in values),
                    f'Đối chứng không ghép cặp: {label}')
        verified[label] = True

    for key in ('C_trainable_sha256', 'C_nontrainable_sha256', 'C_adam_slots_sha256'):
        same('initialization', key, _M2_ARMS, 'all_arms_' + key)
    for key in ('D_trainable_sha256', 'G_trainable_sha256', 'head_dense_trainable_sha256'):
        same('initialization', key, _M2_GAN_ARMS, 'GAN_arms_' + key)
    same('trace', 'common_C', _M2_ARMS, 'actual_common_C_all_arms')
    same('trace', 'anchors_C', _M2_ANCHOR_ARMS, 'actual_anchors_ROS_random_learned')
    same('trace', 'local_geometry', _M2_LOCAL_ARMS, 'actual_local_geometry_random_learned')
    same('trace', 'GAN_private', _M2_GAN_ARMS, 'actual_GAN_private_original_learned')
    return verified


def _m2_verify_available_pairs(reports):
    """Fail at the first available mismatch, before spending the next arm."""
    checked = {}
    groups = (
        ('initialization', ('C_trainable_sha256','C_nontrainable_sha256','C_adam_slots_sha256'), _M2_ARMS),
        ('initialization', ('D_trainable_sha256','G_trainable_sha256','head_dense_trainable_sha256'), _M2_GAN_ARMS),
        ('trace', ('common_C',), _M2_ARMS),
        ('trace', ('anchors_C',), _M2_ANCHOR_ARMS),
        ('trace', ('local_geometry',), _M2_LOCAL_ARMS),
        ('trace', ('GAN_private',), _M2_GAN_ARMS),
    )
    for section, keys, arms in groups:
        available = [arm for arm in arms if arm in reports]
        if len(available) < 2:
            continue
        for key in keys:
            values = [reports[arm][section][key] for arm in available]
            _m2_require(all(value == values[0] for value in values),
                        f'Ghép cặp lệch ngay sau {available[-1]}: {section}/{key}; không chạy thêm nhánh.')
            checked[section+'/'+key] = available
    return checked


def launch_moc2():
    # One absolute cutoff covers ALL four workers, hashes, TF imports and compile.
    started = _m2_time.monotonic()
    if not _m2_sys.platform.startswith('linux') or not _M2Path('/kaggle/working').is_dir():
        raise RuntimeError('Mốc 2 chỉ chạy trên Kaggle; không chạy ở máy cá nhân.')
    if type(TIME_LIMIT_SECONDS) is not int or not 1 <= TIME_LIMIT_SECONDS <= 1800:
        raise ValueError('TIME_LIMIT_SECONDS phải trong 1..1800; tổng bốn nhánh tối đa 30 phút.')
    if type(GPU_INDEX) is not int or GPU_INDEX < 0:
        raise ValueError('GPU_INDEX phải là số nguyên không âm')
    if type(SEED) is not int or SEED != 42:
        raise ValueError('Mốc 2 khóa SEED=42; không tìm seed hoặc tự chạy lại')
    if B_RUN is not None and not isinstance(B_RUN, str):
        raise ValueError('B_RUN phải là None hoặc đường dẫn dạng chuỗi')
    if not isinstance(WORKER_SOURCE, str) or not WORKER_SOURCE.strip():
        raise ValueError('Thiếu WORKER_SOURCE đã đóng gói trong notebook')
    deadline = started + TIME_LIMIT_SECONDS
    stamp = _m2_dt.datetime.now(_m2_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    attempt_id = stamp + '_' + _m2_uuid.uuid4().hex[:8]
    folder = _M2Path('/kaggle/working') / ('MOC2_IP_SMOKE_' + attempt_id)
    folder.mkdir(exist_ok=False)
    worker_path = folder / 'worker.py'
    worker_path.write_text(WORKER_SOURCE, encoding='utf-8')
    worker_hash = _m2_hashlib.sha256(worker_path.read_bytes()).hexdigest()
    result = {
        'schema': 3, 'attempt_id': attempt_id, 'status': 'RUNNING', 'pass': False,
        'worker_sha256': worker_hash, 'scope': 'Moc2 four-arm two-loop integration only',
        'data_mode': 'auto_real_B', 'seed': SEED, 'gpu_index': GPU_INDEX,
        'limit_seconds': TIME_LIMIT_SECONDS, 'arm_order': list(_M2_ARMS),
        'C_updates_per_arm': 4, 'final_Adam_clock_per_arm': 68,
        'accuracy_computed': False, 'validation_arrays_loaded': False,
        'test_opened': False, 'full_training_allowed': False,
        'watchdog': 'independent GNU timeout; one shared absolute deadline',
        'arms': {}, 'paired_gates': {}, 'cleanup_errors': [], 'log_read_errors': [],
    }
    _m2_atomic_json(folder / 'result.json', result)
    process = None
    current_arm = None
    current_folder = None
    offset = 0

    def read_log_tail():
        nonlocal offset
        if current_folder is None:
            return
        log_path = current_folder / 'worker.log'
        try:
            if log_path.is_file():
                with log_path.open('r', encoding='utf-8', errors='replace') as log:
                    log.seek(offset)
                    content = log.read(65536)
                    offset = log.tell()
                if content:
                    print(content, end='', flush=True)
        except OSError as error:
            result['log_read_errors'].append({'arm': current_arm, 'error': str(error)})

    def kill_worker():
        if process is not None and process.poll() is None:
            try:
                _m2_os.killpg(process.pid, _m2_signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                result['cleanup_errors'].append({'arm': current_arm, 'error': str(error)})
            try:
                process.wait(timeout=5)
            except (_m2_subprocess.TimeoutExpired, OSError) as error:
                result['cleanup_errors'].append({'arm': current_arm, 'error': str(error)})

    def check_deadline():
        if _m2_time.monotonic() >= deadline:
            raise TimeoutError('Đã tới deadline tổng Mốc 2; không khởi động thêm nhánh.')

    reports = {}
    try:
        print('MỐC 2:', folder, flush=True)
        print(f'Bốn nhánh tuần tự trên GPU {GPU_INDEX}; tổng deadline {TIME_LIMIT_SECONDS}s; '
              'mỗi nhánh hai loop, không accuracy/validation/test.', flush=True)
        timeout_executable = _m2_shutil.which('timeout')
        _m2_require(timeout_executable is not None,
                    'Thiếu GNU timeout; không khởi động GPU worker.')
        env = dict(_m2_os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(GPU_INDEX), PYTHONUNBUFFERED='1',
                   TF_USE_LEGACY_KERAS='1', PYTHONHASHSEED=str(SEED), LC_ALL='C')
        for arm in _M2_ARMS:
            check_deadline()
            current_arm = arm
            current_folder = folder / arm
            current_folder.mkdir(exist_ok=False)
            offset = 0
            process = None
            config = {
                'schema': 3, 'attempt_id': attempt_id, 'arm': arm,
                'attempt_dir': str(current_folder), 'suite_attempt_dir': str(folder),
                'seed': SEED, 'data_mode': 'auto_real_B', 'b_run': B_RUN,
                'gpu_index': GPU_INDEX, 'time_limit_seconds': TIME_LIMIT_SECONDS,
                'start_monotonic': started, 'deadline_monotonic': deadline,
                'worker_sha256': worker_hash,
            }
            config['config_sha256'] = _m2_json_sha(config)
            config_path = current_folder / 'config.json'
            _m2_atomic_json(config_path, config)
            result['arms'][arm] = {'status': 'RUNNING', 'pass': False,
                                   'config_sha256': config['config_sha256']}
            _m2_atomic_json(folder / 'result.json', result)
            print(f'\nNHÁNH {len(reports) + 1}/4: {arm} | '
                  f'còn {max(0., deadline - _m2_time.monotonic()):.1f}s trong deadline chung',
                  flush=True)
            with (current_folder / 'worker.log').open('w', encoding='utf-8') as output:
                # No --foreground: monitor and child share this new process group.
                # The independent monitor also survives a killed notebook kernel.
                remaining = deadline - _m2_time.monotonic() - .5
                if remaining <= 0:
                    raise TimeoutError('Hết thời gian trước khi khởi động watchdog nhánh tiếp theo')
                process = _m2_subprocess.Popen(
                    [timeout_executable, '--verbose', '--signal=KILL', f'{remaining:.6f}s',
                     _m2_sys.executable, '-u', str(worker_path), str(config_path)],
                    stdout=output, stderr=_m2_subprocess.STDOUT, env=env,
                    start_new_session=True)
                last_heartbeat = _m2_time.monotonic()
                while True:
                    check_deadline()
                    if process.poll() is not None:
                        break
                    read_log_tail()
                    now = _m2_time.monotonic()
                    if now - last_heartbeat >= 45:
                        print(f'{arm}: worker còn chạy | tổng {now - started:.0f}/'
                              f'{TIME_LIMIT_SECONDS}s', flush=True)
                        last_heartbeat = now
                    _m2_time.sleep(min(1., max(.01, deadline - _m2_time.monotonic())))
            read_log_tail()
            check_deadline()
            log_text = (current_folder / 'worker.log').read_text(encoding='utf-8', errors='replace')
            if process.returncode in (124, 137, -9) and 'sending signal KILL' in log_text:
                raise TimeoutError(f'Watchdog đã dừng {arm} khi chạm deadline tổng')
            report = _m2_verify_arm(current_folder, config)
            _m2_require(process.returncode == 0,
                        f'Worker {arm} thoát lỗi: exit {process.returncode}; không chấp nhận PASS')
            reports[arm] = report
            result['arms'][arm] = {'status': 'PASS', 'pass': True,
                                   'worker_exit_code': process.returncode,
                                   'config_sha256': config['config_sha256'],
                                   'worker_report': report}
            result['paired_gates_so_far'] = _m2_verify_available_pairs(reports)
            _m2_atomic_json(folder / 'result.json', result)
            print(f'{arm}: PASS kỹ thuật; không phải kết quả accuracy.', flush=True)
        check_deadline()
        result['paired_gates'] = _m2_verify_pairs(reports)
        check_deadline()
        result.update(status='PASS', **{'pass': True})
    except BaseException as error:
        kill_worker()
        status = ('TIMEOUT' if isinstance(error, TimeoutError)
                  else 'ABORTED' if isinstance(error, KeyboardInterrupt) else 'FAILED')
        result.update(status=status, **{'pass': False}, error=str(error),
                      traceback=_m2_traceback.format_exc())
        if current_arm is not None and result['arms'].get(current_arm, {}).get('status') == 'RUNNING':
            result['arms'][current_arm].update(status=status, error=str(error))
            report_path = current_folder / 'arm_report.json'
            if report_path.is_file():
                try:
                    result['arms'][current_arm]['worker_report_unvalidated'] = _m2_read_json(report_path)
                except (OSError, ValueError, RuntimeError):
                    pass
    finally:
        kill_worker()
        read_log_tail()
        if result['cleanup_errors']:
            result.update(status='FAILED', **{'pass': False},
                          error='Không xác minh được cleanup; xem cleanup_errors.')
        result['elapsed_seconds'] = round(_m2_time.monotonic() - started, 3)
        result['last_worker_exit_code'] = None if process is None else process.returncode
        result['completed_arm_count'] = len(reports)
        _m2_atomic_json(folder / 'result.json', result)
        archive_path = folder.with_name(folder.name + '_REPORT.zip')
        temporary_archive = archive_path.with_suffix('.zip.tmp')
        with _m2_zipfile.ZipFile(temporary_archive, 'w', _m2_zipfile.ZIP_DEFLATED) as archive:
            for name in ('result.json', 'worker.py'):
                path = folder / name
                if path.is_file():
                    archive.write(path, folder.name + '/' + name)
            for arm in _M2_ARMS:
                for name in ('config.json', 'worker.log', 'arm_report.json', 'consumed_trace.json', 'schedule_spec.json'):
                    path = folder / arm / name
                    if path.is_file():
                        archive.write(path, folder.name + '/' + arm + '/' + name)
        temporary_archive.replace(archive_path)
        print('\nKẾT THÚC MỐC 2:', result['status'], '|', result['elapsed_seconds'], 's', flush=True)
        print('Báo cáo chính:', folder / 'result.json', flush=True)
        print('ZIP NHỎ:', archive_path, flush=True)
        try:
            from IPython.display import display, FileLink
            display(FileLink(str(archive_path)))
        except ImportError:
            pass
    if result['pass'] is not True:
        raise RuntimeError(f"Mốc 2 {result['status']}; đã lưu báo cáo, không tự chạy lại. "
                           f"{result.get('error', '')}")
    return result


moc2_result = launch_moc2()
