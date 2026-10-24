# Kaggle-only Mốc 3 launcher. TensorFlow is imported only inside fresh workers.
import datetime as _m3_dt
import hashlib as _m3_hashlib
import json as _m3_json
import math as _m3_math
import os as _m3_os
from pathlib import Path as _M3Path
import shutil as _m3_shutil
import signal as _m3_signal
import subprocess as _m3_subprocess
import sys as _m3_sys
import time as _m3_time
import traceback as _m3_traceback
import uuid as _m3_uuid
import zipfile as _m3_zipfile


_M3_ARMS = ('original_gan', 'matched_ros', 'random_local', 'learned_local')
_M3_GAN_ARMS = ('original_gan', 'learned_local')
_M3_ANCHOR_ARMS = ('matched_ros', 'random_local', 'learned_local')
_M3_LOCAL_ARMS = ('random_local', 'learned_local')
_M3_GATE_KEYS = ('benchmark', 'actual_tail_batch', 'checkpoint_restore',
                 'resume_trace', 'resume_numeric')


def _m3_json_sha(value):
    payload = _m3_json.dumps(value, sort_keys=True, separators=(',', ':'),
                            ensure_ascii=True, allow_nan=False).encode('utf-8')
    return _m3_hashlib.sha256(payload).hexdigest()


def _m3_require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _m3_is_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _m3_read_json(path):
    value = _m3_json.loads(_M3Path(path).read_text(encoding='utf-8'))
    _m3_require(isinstance(value, dict), f'JSON không phải object: {path}')
    return value


def _m3_atomic_json(path, value):
    path = _M3Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(_m3_json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _m3_verify_identity(report, config):
    for key in ('attempt_id', 'arm', 'config_sha256', 'worker_sha256'):
        _m3_require(report.get(key) == config[key], f'Report khác {key}: {config["arm"]}')
    if report.get('status') == 'TIMEOUT':
        raise TimeoutError(report.get('error', f'Worker {config["arm"]} hết thời gian'))
    _m3_require(report.get('input', {}).get('real_B_cache_verified') is True,
                f'Worker {config["arm"]} chưa xác thực cache B gốc')
    for flag in ('accuracy_computed', 'validation_arrays_loaded',
                 'test_arrays_loaded', 'full_training_allowed'):
        _m3_require(report.get(flag) is False, f'Cờ phạm vi thiếu/sai: {config["arm"]}/{flag}')


def _m3_verify_main_fields(folder, config, report):
    clocks = report.get('clocks', {})
    expected_main_calls = 1564 if config['arm'] in _M3_GAN_ARMS else 92
    _m3_require(clocks.get('main_C_updates') == 92 and clocks.get('main_clock') == 1564
                and clocks.get('main_optimizer_calls') == expected_main_calls,
                f'Clock/cập nhật main không đúng Mốc 3: {folder.name}')
    benchmark = report.get('benchmark', {})
    _m3_require(benchmark.get('steady_full_batch_loops') == 43
                and benchmark.get('tail_loop') == 44 and benchmark.get('tail_batch_size') == 22,
                f'Benchmark thiếu full/tail batch đã chạy thật: {folder.name}')
    for key in ('mean_seconds_per_loop', 'p50_seconds_per_loop'):
        value = benchmark.get(key)
        _m3_require(type(value) in (int, float) and _m3_math.isfinite(value) and value > 0,
                    f'Benchmark thời gian thiếu/không hợp lệ: {folder.name}/{key}')
    peak = benchmark.get('tf_allocator_peak_bytes')
    _m3_require(type(peak) is int and peak >= 0,
                f'Thiếu số đo peak allocator TensorFlow: {folder.name}')
    initialization = report.get('initialization', {})
    init_keys = ['C_trainable_sha256', 'C_nontrainable_sha256', 'C_adam_slots_sha256']
    if config['arm'] in _M3_GAN_ARMS:
        init_keys.extend(('D_trainable_sha256', 'G_trainable_sha256',
                          'head_dense_trainable_sha256'))
    for key in init_keys:
        _m3_require(_m3_is_sha(initialization.get(key)),
                    f'Thiếu hash khởi tạo: {folder.name}/{key}')
    for name in ('schedule_spec.json',):
        _m3_require((folder / name).is_file(), f'Thiếu bằng chứng main: {folder.name}/{name}')
        _m3_read_json(folder / name)
    consumed_path = folder / 'consumed_trace.json'
    _m3_require(consumed_path.is_file(), f'Thiếu consumed_trace.json: {folder.name}')
    consumed = _m3_read_json(consumed_path)
    _m3_require(consumed.get('attempt_id') == config['attempt_id']
                and consumed.get('arm') == config['arm'],
                f'Consumed trace khác attempt/arm: {folder.name}')
    trace = report.get('trace', {})
    _m3_require(trace.get('consumed_loops') == 46,
                f'Worker không tiêu thụ đúng 46 loop chính: {folder.name}')
    for records_key, hash_key, applicable, expected_rows in (
        ('common_c', 'common_C', True, 92),
        ('anchors', 'anchors_C', config['arm'] in _M3_ANCHOR_ARMS, 46),
        ('local_geometry', 'local_geometry', config['arm'] in _M3_LOCAL_ARMS, 46),
        ('common_gan', 'GAN_private', config['arm'] in _M3_GAN_ARMS, 1518),
    ):
        records = consumed.get(records_key)
        if applicable:
            _m3_require(isinstance(records, list) and len(records) == expected_rows,
                        f'Sai số actual main records: {folder.name}/{records_key}; cần {expected_rows}')
            _m3_require(_m3_is_sha(trace.get(hash_key))
                        and trace[hash_key] == _m3_json_sha(records),
                        f'Hash khác actual consumed records: {folder.name}/{records_key}')
        else:
            _m3_require(records in (None, []) and trace.get(hash_key) is None,
                        f'Trace không áp dụng phải rỗng: {folder.name}/{records_key}')


def _m3_verify_main(folder, config):
    path = folder / 'main_report.json'
    _m3_require(path.is_file(), f'Thiếu main_report.json: {folder.name}')
    report = _m3_read_json(path)
    _m3_verify_identity(report, config)
    _m3_require(report.get('status') == 'MAIN_COMPLETE' and report.get('pass') is False,
                f'Main chưa hoàn tất hoặc tự nhận PASS trước cold resume: {folder.name}')
    _m3_verify_main_fields(folder, config, report)
    return report


def _m3_verify_arm(folder, config):
    """Accept final PASS only from the separate fresh resume process."""
    path = folder / 'final_report.json'
    _m3_require(path.is_file(), f'Thiếu final_report.json: {folder.name}')
    report = _m3_read_json(path)
    _m3_verify_identity(report, config)
    _m3_require(report.get('status') == 'PASS' and report.get('pass') is True,
                f'Cold resume chưa PASS: {folder.name}: {report.get("error", report.get("phase"))}')
    _m3_require(report.get('main_config_sha256') == config['main_config_sha256'],
                f'Final report khác main config hiện tại: {folder.name}')
    _m3_verify_main_fields(folder, config, report)
    clocks = report.get('clocks', {})
    expected_replay_calls = 34 if config['arm'] in _M3_GAN_ARMS else 2
    _m3_require(clocks.get('replay_C_updates') == 2
                and clocks.get('replay_optimizer_calls') == expected_replay_calls
                and clocks.get('final_clock') == 1564,
                f'Clock/cập nhật replay không đúng Mốc 3: {folder.name}')
    for key in _M3_GATE_KEYS:
        _m3_require(report.get('gates', {}).get(key) is True,
                    f'Gate chưa PASS: {folder.name}/{key}')
    resume = report.get('resume', {})
    for key in ('trace_exact', 'state_restore_exact', 'numeric_allclose'):
        _m3_require(resume.get(key) is True, f'Resume chưa PASS: {folder.name}/{key}')
    _m3_require(type(resume.get('bitwise_equal')) is bool,
                f'Thiếu cờ bitwise_equal: {folder.name}')
    _m3_require(resume.get('rtol') == 1e-6 and resume.get('atol') == 1e-7,
                f'Sai tolerance resume đã khóa: {folder.name}')
    policy = resume.get('comparison_policy')
    _m3_require(isinstance(policy, (str, dict)) and bool(policy),
                f'Thiếu chính sách so sánh model/Adam/control: {folder.name}')
    for name in ('resume_ledger.json', 'resume_compare.json'):
        evidence_path = folder / name
        _m3_require(evidence_path.is_file(), f'Thiếu bằng chứng cold resume: {folder.name}/{name}')
        _m3_read_json(evidence_path)
    return report


def _m3_verify_pairs(reports):
    _m3_require(set(reports) == set(_M3_ARMS), 'Chưa đủ bốn report Mốc 3')
    verified = {}

    def same(section, key, arms, label):
        values = [reports[arm][section][key] for arm in arms]
        _m3_require(all(value == values[0] for value in values),
                    f'Đối chứng không ghép cặp: {label}')
        verified[label] = True

    for key in ('C_trainable_sha256', 'C_nontrainable_sha256', 'C_adam_slots_sha256'):
        same('initialization', key, _M3_ARMS, 'all_arms_' + key)
    for key in ('D_trainable_sha256', 'G_trainable_sha256', 'head_dense_trainable_sha256'):
        same('initialization', key, _M3_GAN_ARMS, 'GAN_arms_' + key)
    same('trace', 'common_C', _M3_ARMS, 'actual_common_C_all_arms')
    same('trace', 'anchors_C', _M3_ANCHOR_ARMS, 'actual_anchors_ROS_random_learned')
    same('trace', 'local_geometry', _M3_LOCAL_ARMS, 'actual_local_geometry_random_learned')
    same('trace', 'GAN_private', _M3_GAN_ARMS, 'actual_GAN_private_original_learned')
    return verified


def _m3_verify_available_pairs(reports):
    """Fail at the first available mismatch, before spending the next arm."""
    checked = {}
    groups = (
        ('initialization', ('C_trainable_sha256','C_nontrainable_sha256','C_adam_slots_sha256'), _M3_ARMS),
        ('initialization', ('D_trainable_sha256','G_trainable_sha256','head_dense_trainable_sha256'), _M3_GAN_ARMS),
        ('trace', ('common_C',), _M3_ARMS),
        ('trace', ('anchors_C',), _M3_ANCHOR_ARMS),
        ('trace', ('local_geometry',), _M3_LOCAL_ARMS),
        ('trace', ('GAN_private',), _M3_GAN_ARMS),
    )
    for section, keys, arms in groups:
        available = [arm for arm in arms if arm in reports]
        if len(available) < 2:
            continue
        for key in keys:
            values = [reports[arm][section][key] for arm in available]
            _m3_require(all(value == values[0] for value in values),
                        f'Ghép cặp lệch ngay sau {available[-1]}: {section}/{key}; không chạy thêm nhánh.')
            checked[section+'/'+key] = available
    return checked


def launch_moc3():
    # One absolute deadline covers all eight fresh child processes and TF imports.
    started = _m3_time.monotonic()
    if not _m3_sys.platform.startswith('linux') or not _M3Path('/kaggle/working').is_dir():
        raise RuntimeError('Mốc 3 chỉ chạy trên Kaggle; không chạy ở máy cá nhân.')
    if type(TIME_LIMIT_SECONDS) is not int or not 1 <= TIME_LIMIT_SECONDS <= 1800:
        raise ValueError('TIME_LIMIT_SECONDS phải trong 1..1800; toàn Mốc 3 tối đa 30 phút.')
    if type(GPU_INDEX) is not int or GPU_INDEX < 0:
        raise ValueError('GPU_INDEX phải là số nguyên không âm')
    if type(SEED) is not int or SEED != 42:
        raise ValueError('Mốc 3 khóa SEED=42; không tìm seed hoặc tự chạy lại')
    if B_RUN is not None and not isinstance(B_RUN, str):
        raise ValueError('B_RUN phải là None hoặc đường dẫn dạng chuỗi')
    if not isinstance(WORKER_SOURCE, str) or not WORKER_SOURCE.strip():
        raise ValueError('Thiếu WORKER_SOURCE đóng gói trong notebook')
    deadline = started + TIME_LIMIT_SECONDS
    stamp = _m3_dt.datetime.now(_m3_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    attempt_id = stamp + '_' + _m3_uuid.uuid4().hex[:8]
    folder = _M3Path('/kaggle/working') / ('MOC3_IP_FINAL_' + attempt_id)
    folder.mkdir(exist_ok=False)
    worker_path = folder / 'worker.py'
    worker_path.write_text(WORKER_SOURCE, encoding='utf-8')
    worker_hash = _m3_hashlib.sha256(worker_path.read_bytes()).hexdigest()
    result = {
        'schema': 4, 'attempt_id': attempt_id, 'status': 'RUNNING', 'pass': False,
        'worker_sha256': worker_hash, 'scope': 'final bounded technical gate, not accuracy study',
        'data_mode': 'auto_real_B', 'seed': SEED, 'gpu_index': GPU_INDEX,
        'limit_seconds': TIME_LIMIT_SECONDS, 'arm_order': list(_M3_ARMS),
        'execution_modes': ['main', 'resume'], 'worker_processes_planned': 8,
        'main_loops_per_arm': 46, 'replay_loops_per_arm': 1,
        'main_C_updates_per_arm': 92, 'replay_C_updates_per_arm': 2,
        'final_Adam_clock_per_arm': 1564,
        'accuracy_computed': False, 'validation_arrays_loaded': False,
        'test_opened': False, 'full_training_allowed': False,
        'watchdog': 'independent GNU timeout; one shared absolute deadline',
        'arms': {}, 'paired_gates': {}, 'cleanup_errors': [], 'log_read_errors': [],
        'worker_processes_started': 0,
    }
    _m3_atomic_json(folder / 'result.json', result)
    process = None
    current_arm = current_folder = current_mode = current_log_path = None
    offset = 0

    def read_log_tail():
        nonlocal offset
        if current_log_path is None:
            return
        try:
            if current_log_path.is_file():
                with current_log_path.open('r', encoding='utf-8', errors='replace') as log:
                    log.seek(offset)
                    content = log.read(65536)
                    offset = log.tell()
                if content:
                    print(content, end='', flush=True)
        except OSError as error:
            result['log_read_errors'].append({'arm': current_arm, 'mode': current_mode, 'error': str(error)})

    def kill_worker():
        if process is not None and process.poll() is None:
            try:
                _m3_os.killpg(process.pid, _m3_signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                result['cleanup_errors'].append({'arm': current_arm, 'mode': current_mode, 'error': str(error)})
            try:
                process.wait(timeout=5)
            except (_m3_subprocess.TimeoutExpired, OSError) as error:
                result['cleanup_errors'].append({'arm': current_arm, 'mode': current_mode, 'error': str(error)})

    def check_deadline():
        if _m3_time.monotonic() >= deadline:
            raise TimeoutError('Đã tới deadline tổng Mốc 3; không khởi động thêm process.')

    reports = {}
    main_reports = {}
    try:
        print('MỐC 3:', folder, flush=True)
        print(f'Bốn nhánh tuần tự trên GPU {GPU_INDEX}; mỗi nhánh main rồi cold resume '
              f'trong hai process mới. Tổng deadline {TIME_LIMIT_SECONDS}s; '
              '46 loop chính + replay loop45; không accuracy/validation/test.', flush=True)
        timeout_executable = _m3_shutil.which('timeout')
        _m3_require(timeout_executable is not None, 'Thiếu GNU timeout; không khởi động GPU worker.')
        env = dict(_m3_os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(GPU_INDEX), PYTHONUNBUFFERED='1',
                   TF_USE_LEGACY_KERAS='1', PYTHONHASHSEED=str(SEED), LC_ALL='C')
        for arm in _M3_ARMS:
            check_deadline()
            current_arm = arm
            current_folder = folder / arm
            current_folder.mkdir(exist_ok=False)
            arm_record = {'status': 'RUNNING', 'pass': False, 'stages': {}}
            result['arms'][arm] = arm_record
            base_config = {
                'schema': 4, 'attempt_id': attempt_id, 'arm': arm,
                'attempt_dir': str(current_folder), 'suite_attempt_dir': str(folder),
                'seed': SEED, 'data_mode': 'auto_real_B', 'b_run': B_RUN,
                'gpu_index': GPU_INDEX, 'time_limit_seconds': TIME_LIMIT_SECONDS,
                'start_monotonic': started, 'deadline_monotonic': deadline,
                'worker_sha256': worker_hash,
            }
            main_config_hash = None
            for mode in ('main', 'resume'):
                check_deadline()
                current_mode = mode
                current_log_path = current_folder / (mode + '_worker.log')
                offset = 0
                process = None
                config = dict(base_config, execution_mode=mode)
                if mode == 'resume':
                    _m3_require(_m3_is_sha(main_config_hash), 'Chưa có main config đã hoàn tất')
                    config['main_config_sha256'] = main_config_hash
                config['config_sha256'] = _m3_json_sha(config)
                _m3_atomic_json(current_folder / 'config.json', config)
                if mode == 'main':
                    main_config_hash = config['config_sha256']
                    config_path = current_folder / 'main_config.json'
                    _m3_atomic_json(config_path, config)
                else:
                    config_path = current_folder / 'config.json'
                arm_record['stages'][mode] = {'status': 'RUNNING', 'pass': False,
                                              'config_sha256': config['config_sha256']}
                _m3_atomic_json(folder / 'result.json', result)
                print(f'\n{arm} / {mode} | process {result["worker_processes_started"] + 1}/8 | '
                      f'còn {max(0., deadline - _m3_time.monotonic()):.1f}s trong deadline chung', flush=True)
                with current_log_path.open('w', encoding='utf-8') as output:
                    remaining = deadline - _m3_time.monotonic() - .5
                    if remaining <= 0:
                        raise TimeoutError('Hết thời gian trước khi khởi động watchdog tiếp theo')
                    process = _m3_subprocess.Popen(
                        [timeout_executable, '--verbose', '--signal=KILL', f'{remaining:.6f}s',
                         _m3_sys.executable, '-u', str(worker_path), str(config_path)],
                        stdout=output, stderr=_m3_subprocess.STDOUT, env=env, start_new_session=True)
                    result['worker_processes_started'] += 1
                    last_heartbeat = _m3_time.monotonic()
                    while True:
                        check_deadline()
                        if process.poll() is not None:
                            break
                        read_log_tail()
                        now = _m3_time.monotonic()
                        if now - last_heartbeat >= 45:
                            print(f'{arm}/{mode}: còn chạy | tổng {now - started:.0f}/'
                                  f'{TIME_LIMIT_SECONDS}s', flush=True)
                            last_heartbeat = now
                        _m3_time.sleep(min(1., max(.01, deadline - _m3_time.monotonic())))
                read_log_tail()
                check_deadline()
                log_text = current_log_path.read_text(encoding='utf-8', errors='replace')
                if process.returncode in (124, 137, -9) and 'sending signal KILL' in log_text:
                    raise TimeoutError(f'Watchdog đã dừng {arm}/{mode} tại deadline tổng')
                # Inspect a bound TIMEOUT report before classifying its nonzero exit.
                # A report is never accepted unless the corresponding process exited 0.
                report = (_m3_verify_main(current_folder, config) if mode == 'main'
                          else _m3_verify_arm(current_folder, config))
                _m3_require(process.returncode == 0,
                            f'Worker {arm}/{mode} thoát lỗi {process.returncode}; xem log của stage')
                if mode == 'main':
                    main_reports[arm] = report
                    # Main comparison is a fail-fast precheck, never final PASS.
                    result['paired_main_gates_so_far'] = _m3_verify_available_pairs(main_reports)
                    arm_record['stages'][mode].update(status='MAIN_COMPLETE', worker_exit_code=0,
                                                      worker_report=report)
                    arm_record['status'] = 'MAIN_COMPLETE'
                    print(f'{arm}: MAIN_COMPLETE; chưa PASS, bắt đầu process cold resume mới.', flush=True)
                else:
                    reports[arm] = report
                    result['paired_gates_so_far'] = _m3_verify_available_pairs(reports)
                    arm_record['stages'][mode].update(status='PASS', **{'pass': True}, worker_exit_code=0,
                                                      worker_report=report)
                    arm_record.update(status='PASS', **{'pass': True})
                    print(f'{arm}: PASS sau cold resume; không phải kết quả accuracy.', flush=True)
                _m3_atomic_json(folder / 'result.json', result)
        check_deadline()
        result['paired_gates'] = _m3_verify_pairs(reports)
        check_deadline()
        result.update(status='PASS', **{'pass': True})
    except BaseException as error:
        kill_worker()
        status = ('TIMEOUT' if isinstance(error, TimeoutError)
                  else 'ABORTED' if isinstance(error, KeyboardInterrupt) else 'FAILED')
        result.update(status=status, **{'pass': False}, error=str(error), traceback=_m3_traceback.format_exc())
        if current_arm is not None and current_arm in result['arms']:
            arm_record = result['arms'][current_arm]
            if arm_record.get('status') != 'PASS':
                arm_record.update(status=status, error=str(error))
                if current_mode in arm_record['stages']:
                    arm_record['stages'][current_mode].update(status=status, **{'pass': False}, error=str(error))
            report_name = 'main_report.json' if current_mode == 'main' else 'final_report.json'
            report_path = current_folder / report_name
            if report_path.is_file():
                try:
                    arm_record['worker_report_unvalidated'] = _m3_read_json(report_path)
                except (OSError, ValueError, RuntimeError):
                    pass
    finally:
        kill_worker()
        read_log_tail()
        if result['cleanup_errors']:
            result.update(status='FAILED', **{'pass': False}, error='Không xác minh được cleanup; xem cleanup_errors.')
        result['elapsed_seconds'] = round(_m3_time.monotonic() - started, 3)
        result['last_worker_exit_code'] = None if process is None else process.returncode
        result['main_completed_arm_count'] = len(main_reports)
        result['completed_arm_count'] = len(reports)
        _m3_atomic_json(folder / 'result.json', result)
        archive_path = folder.with_name(folder.name + '_REPORT.zip')
        temporary_archive = archive_path.with_suffix('.zip.tmp')
        with _m3_zipfile.ZipFile(temporary_archive, 'w', _m3_zipfile.ZIP_DEFLATED) as archive:
            for name in ('result.json', 'worker.py'):
                path = folder / name
                if path.is_file():
                    archive.write(path, folder.name + '/' + name)
            for arm in _M3_ARMS:
                for name in ('main_config.json', 'config.json', 'main_worker.log', 'resume_worker.log',
                             'main_report.json', 'final_report.json', 'consumed_trace.json',
                             'schedule_spec.json', 'resume_ledger.json', 'resume_compare.json'):
                    path = folder / arm / name
                    if path.is_file():
                        archive.write(path, folder.name + '/' + arm + '/' + name)
        temporary_archive.replace(archive_path)
        print('\nKẾT THÚC MỐC 3:', result['status'], '|', result['elapsed_seconds'], 's', flush=True)
        print('Báo cáo chính:', folder / 'result.json', flush=True)
        print('ZIP NHỎ:', archive_path, flush=True)
        try:
            from IPython.display import display, FileLink
            display(FileLink(str(archive_path)))
        except ImportError:
            pass
    if result['pass'] is not True:
        raise RuntimeError(f"Mốc 3 {result['status']}; đã lưu báo cáo, không tự chạy lại. "
                           f"{result.get('error', '')}")
    return result


moc3_result = launch_moc3()
