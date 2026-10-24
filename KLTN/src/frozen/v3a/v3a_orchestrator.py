"""One-run Kaggle V3-A orchestration. Importing this module never starts work."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
import numpy as np

PROTOCOL = '50e988fae260'
M4_SHA = '6d33ca6a0d39600f5f44c0d7e63547b1237149d625471ac73315d982ffbd8fb5'
PREP_SHA = '18167b01cbb8ac5d21ff73a3588cad758d23a2629f9b1c9e47dbaf4cf6989939'
HSI_SHA = 'ec2f8808710919d566f70f0d4aa885aae1ddfd42b734aba71c5e12ca65450939'
METRICS = ('OA', 'AA', 'Macro_F1', 'GM', 'Kappa', 'NLL', 'Brier')
COUNTS = [25,675,404,99,274,354,12,250,10,503,1065,282,80,545,99,44]
CLASS_NAMES = ['Alfalfa','Corn-notill','Corn-mintill','Corn','Grass-pasture','Grass-trees','Grass-pasture-mowed','Hay-windrowed','Oats','Soybean-notill','Soybean-mintill','Soybean-clean','Wheat','Woods','Buildings-Grass-Trees-Drives','Stone-Steel-Towers']


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    return sha_bytes(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())


def b_array_sha(a):
    a = np.ascontiguousarray(a)
    return sha_bytes(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes())


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def csv_write(path, rows):
    require(bool(rows), 'Cannot write empty result table')
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ArtifactSource:
    def __init__(self, path, prefix='', is_zip=False):
        self.path = Path(path)
        self.prefix = prefix
        self.is_zip = bool(is_zip)

    def label(self):
        return str(self.path)+('!/'+self.prefix if self.is_zip else '')

    def read(self, relative):
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Unsafe relative input')
        if self.is_zip:
            with zipfile.ZipFile(self.path) as z:
                return z.read(self.prefix+relative)
        return (self.path/relative).read_bytes()

    def exists(self, relative):
        if self.is_zip:
            with zipfile.ZipFile(self.path) as z:
                return self.prefix+relative in z.namelist()
        return (self.path/relative).is_file()


def pinned_b_files():
    # Import-safe M4 source initializes NumPy but does not import TensorFlow.
    w = import_module('_v3a_pins_only', Path(__file__).with_name('m4_training_worker_frozen.py'))
    return w.PINNED_B


def verify_b_source(source):
    actual = {}
    for name, pin in pinned_b_files().items():
        require(source.exists(name), 'Missing B artifact: '+name)
        actual[name] = sha_bytes(source.read(name))
        require(actual[name] == pin, 'Wrong pinned B artifact: '+source.label()+'/'+name)
    for relative in ('final_data/train_5528.npz','final_data/preprocessing_final.joblib'):
        require(source.exists(relative) and source.exists(relative+'.json'), 'Missing final B input: '+relative)
        side = json.loads(source.read(relative+'.json'))
        require(side['contract']['run_id'] == PROTOCOL, 'Wrong final B run')
        require(sha_bytes(source.read(relative)) == side['sha256'], 'Final B SHA mismatch: '+relative)
        actual[relative] = side['sha256']
    require(actual['final_data/preprocessing_final.joblib'] == PREP_SHA, 'Wrong final PCA/scaler')
    return actual


def discover_b(inputs_root, explicit=None):
    candidates = []
    paths = [Path(explicit)] if explicit else [Path(inputs_root)]
    for path in paths:
        if path.is_file() and path.suffix.lower() == '.zip':
            zip_paths = [path]
        elif path.is_dir():
            if (path/'development/dev_train_val.npz').is_file():
                candidates.append(ArtifactSource(path))
            for p in path.rglob('dev_train_val.npz'):
                if p.parent.name == 'development':
                    candidates.append(ArtifactSource(p.parent.parent))
            zip_paths = sorted(path.rglob('*.zip'))
        else:
            continue
        for zp in zip_paths:
            try:
                with zipfile.ZipFile(zp) as z:
                    prefixes = [n[:-len('development/dev_train_val.npz')] for n in z.namelist() if n.endswith('development/dev_train_val.npz')]
                candidates.extend(ArtifactSource(zp, prefix, True) for prefix in prefixes)
            except (OSError, zipfile.BadZipFile):
                continue
    require(candidates, 'Attach original full Protocol B output KLTN_HyperGAMO_B_50e988fae260 as Kaggle Input (folder or ZIP). No model retraining can replace this input.')
    valid, rejected, seen = [], [], set()
    for source in sorted(candidates, key=lambda s: s.label()):
        if source.label() in seen:
            continue
        seen.add(source.label())
        try:
            verify_b_source(source)
            valid.append(source)
        except (OSError, KeyError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
            rejected.append({'source': source.label(), 'error': str(exc)})
    require(valid, 'No B source passed locked hashes: '+json.dumps(rejected[:5], ensure_ascii=False))
    print('VERIFIED B SOURCE:', valid[0].label(), flush=True)
    return valid[0]


def prepare_train_source(source, destination):
    if not source.is_zip:
        return source.path
    destination = Path(destination)
    # No test arrays/master split are extracted during training preparation.
    names = list(pinned_b_files()) + ['final_data/train_5528.npz','final_data/train_5528.npz.json','final_data/preprocessing_final.joblib','final_data/preprocessing_final.joblib.json']
    for name in names:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source.read(name)
        if target.exists():
            require(sha_file(target) == sha_bytes(content), 'Conflicting materialized B artifact: '+name)
        else:
            target.write_bytes(content)
    return destination


def prepare_test_source(source, destination):
    if not source.is_zip:
        return source.path
    destination = Path(destination)
    for name in ('B_master_split.npz','final_data/test_4721_LOCKED.npz','final_data/test_4721_LOCKED.npz.json'):
        if source.exists(name):
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            content = source.read(name)
            if target.exists():
                require(sha_file(target) == sha_bytes(content), 'Conflicting test cache')
            else:
                target.write_bytes(content)
    return destination


def test_input_preflight(source, inputs_root):
    """Availability/sidecar checks only; never deserialize test labels/features."""
    require(source.exists('B_master_split.npz'), 'B master split missing: attach the complete original B output before spending GPU time')
    if source.exists('final_data/test_4721_LOCKED.npz') and source.exists('final_data/test_4721_LOCKED.npz.json'):
        side=json.loads(source.read('final_data/test_4721_LOCKED.npz.json'))
        expected='5345f4d10f221839e241623885fc495900e1d576390c637062792c86b900346c'
        require(side.get('sha256')==expected and side['contract']['run_id']==PROTOCOL,'Unrecognized locked test sidecar')
        return {'locked_test_cache_available':True,'test_arrays_loaded':False}
    paths=list(Path(inputs_root).rglob('*pines*corrected*.mat'))
    require(any(sha_file(p)==HSI_SHA for p in paths), 'Attach full B output with locked test cache, or exact Indian_pines_corrected.mat; stopped before GPU training')
    return {'locked_test_cache_available':False,'exact_raw_cube_available':True,'test_arrays_loaded':False}


def gpu_inventory():
    proc = subprocess.run(['nvidia-smi','--query-gpu=index,name','--format=csv,noheader'], capture_output=True, text=True, timeout=20)
    require(proc.returncode == 0, 'GPU is required for V3 training; CPU fallback disabled: '+proc.stderr[-500:])
    devices = [line.split(',',1) for line in proc.stdout.strip().splitlines() if ',' in line]
    require(devices, 'No GPU detected. Enable Kaggle GPU and Save & Run All.')
    return [{'index': a.strip(), 'name': b.strip()} for a,b in devices][:2]


def cpu_audit(cfg):
    require(sha_file(cfg['m4_worker_path']) == M4_SHA, 'Frozen M4 source changed')
    w = import_module('_v3a_cpu_m4', cfg['m4_worker_path'])
    w.np = np
    w.CONFIG = {'data_mode':'auto_real_B','stage':'final','b_run':cfg['b_run'],'deadline_monotonic':time.monotonic()+3600}
    X,y,source = w.get_training_data()
    graph = w.build_graph_per_class(X,y)
    pins = read_json(Path(__file__).with_name('baseline_pins.json'))
    for seed in (42,43,44):
        c = pins['seeds'][str(seed)]['contract']
        require(w.sha_array(X) == c['train_X_sha256'] and w.sha_array(y) == c['train_y_sha256'], 'Different final training data')
        require((graph['order']+1).tolist() == c['class_order'], 'Different class order')
        require([w.sha_array(g) for g in graph['neighbor_graphs']] == c['graph_sha256'], 'Different neighbor graphs')
    record = {'status':'PASS_CPU_DATA_GRAPH_METADATA', 'protocol_B_id':PROTOCOL,
        'train_shape':list(X.shape), 'train_X_sha256':w.sha_array(X), 'train_y_sha256':w.sha_array(y),
        'class_order':(graph['order']+1).tolist(), 'graph_sha256':[w.sha_array(g) for g in graph['neighbor_graphs']],
        'data_provenance':source, 'runtime_model_checks':'PENDING_KAGGLE_PILOT', 'test_loaded':False}
    write_json(Path(cfg['base'])/'cpu_audit.json', record)
    print('CPU AUDIT PASS | 5528 final train | original graph | test unopened', flush=True)
    return record


def verify_job(base, seed, worker_sha=None):
    job = Path(base)/'jobs'/f'seed_{seed}'
    done = read_json(job/'completed.json')
    contract = read_json(job/'contract.json')
    ptr = read_json(job/'state/latest.json')
    cp = job/'state'/ptr['name']
    meta = read_json(cp/'meta.json')
    require(done.get('complete') is True and done['c_updates'] == 8000 and done['adam_clock'] == 136000, f'Incomplete V3 job seed{seed}')
    require(ptr['c_updates'] == 8000 and sha_file(cp/'meta.json') == ptr['meta_sha256'], 'Bad final checkpoint pointer')
    require(meta['contract'] == contract and meta['c_updates'] == 8000 and meta['optimizer_clock'] == 136000, 'Checkpoint identity/budget mismatch')
    require(meta['audit_calls'] == 136000, 'Full GAN optimizer call count mismatch')
    require(set(meta['weights_hashes'])=={'weights.index','weights.data-00000-of-00001'}, 'Native checkpoint component inventory incomplete')
    for name, h in meta['weights_hashes'].items():
        require(sha_file(cp/name) == h, 'Corrupt final native component '+name)
    require(sha_file(job/'classifier.weights.h5') == done['model_sha256'], 'Corrupt classifier weights')
    require(sha_file(job/'contract.json') == done['job_contract_sha256'], 'Corrupt job contract')
    require(done['native_checkpoint_meta_sha256']==ptr['meta_sha256'], 'Completion marker and native checkpoint disagree')
    require(sha_file(job/'final_beta.json')==done['final_beta_sha256'], 'Final beta telemetry differs')
    beta=read_json(job/'final_beta.json')
    require(len(beta['signed_beta'])==15 and np.isfinite(beta['signed_beta']).all(), 'Final beta inventory invalid')
    require(beta['beta_gradient_seen']==[8000]*15,'Final beta GC/GD update counts invalid')
    probe_path=job/'selector_probes/probe_08000.json'
    require(sha_file(probe_path)==done['selector_probe_8000_manifest_sha256'],'Final train-only probe seal differs')
    probe=read_json(probe_path)
    require(probe.get('complete') is True and probe['c_updates']==8000 and probe['contract_sha256']==done['job_contract_sha256'],'Wrong final train-only probe identity')
    require(sha_file(probe_path.with_suffix('.npz'))==probe['npz_sha256'] and sha_file(probe_path.with_name('probe_08000_by_class.csv'))==probe['csv_sha256'],'Corrupt final selector probe payload')
    if worker_sha is not None:
        require(contract['sha_worker'] == worker_sha, 'Wrong V3 worker source')
    require(contract['seed'] == seed and contract['protocol_B_id'] == PROTOCOL, 'Wrong seed/protocol')
    require(contract['method']=='V3A_SIGNED_ANCHOR_DISTANCE_BIAS_V1' and contract['beta_count']==15 and contract['distance_epsilon']==1e-8 and contract['rho']==.2 and contract['k']==3,'Wrong V3-A method/configuration')
    return job, contract, done


def metrics(y, probabilities):
    pred = probabilities.argmax(1)
    cm = np.zeros((16,16), dtype=np.int64)
    np.add.at(cm, (y,pred), 1)
    tp = cm.diagonal(); s = cm.sum(1); p = cm.sum(0)
    recall = tp/s
    precision = np.divide(tp,p,out=np.zeros(16),where=p!=0)
    f1 = np.divide(2*tp,s+p,out=np.zeros(16),where=(s+p)!=0)
    oa = float(tp.sum()/len(y)); expected = float(np.dot(s.astype(float),p)/len(y)**2)
    prob64 = probabilities.astype(np.float64)
    values = {'OA':oa,'AA':float(recall.mean()),'Macro_F1':float(f1.mean()),
        'GM':float(np.exp(np.log(recall).mean())) if np.all(recall>0) else 0.,
        'Kappa':(oa-expected)/(1-expected),
        'NLL':float(-np.log(np.clip(prob64[np.arange(len(y)),y],1e-12,1.)).mean()),
        'Brier':float(np.square(prob64-np.eye(16)[y]).sum(1).mean())}
    per_class = [{'class_id':i+1,'class_name':CLASS_NAMES[i],'support':int(s[i]),'precision':float(precision[i]),'recall':float(recall[i]),'F1':float(f1[i])} for i in range(16)]
    return values, per_class, pred, cm


def locked_test(cfg):
    root = Path(cfg['b_run'])
    manifest = read_json(root/'manifest.json')
    require(sha_file(root/'manifest.json') == pinned_b_files()['manifest.json'], 'Unrecognized B manifest before test')
    master = root/'B_master_split.npz'
    require(sha_file(master) == manifest['master_split_sha256'], 'Changed B master split')
    with np.load(master, allow_pickle=False) as z:
        coords = z['coords_test']; gt = z['standard_gt']; split = z['split']
    require(np.array_equal(coords,np.argwhere(split==2)), 'Changed heldout coordinate order')
    y = gt[coords[:,0],coords[:,1]].astype(np.int64)-1
    require(np.bincount(y,minlength=16).tolist() == COUNTS, 'Wrong B test support counts')
    path = root/'final_data/test_4721_LOCKED.npz'
    if path.is_file():
        side = read_json(str(path)+'.json')
        expected = {'run_id':PROTOCOL,'implementation':'5e15f8214bf273bb889187b96c80407a8000e5e1a3ef30734bcfa86ced275a6f','kind':'locked_test','coords':b_array_sha(coords),'preprocessing_sha256':PREP_SHA}
        require(side['contract'] == expected and sha_file(path) == side['sha256'], 'Changed locked test cache')
        with np.load(path, allow_pickle=False) as z:
            X = z['X_test']; y2 = z['y_test']; coords2 = z['coords_test']
        require(np.array_equal(y,y2) and np.array_equal(coords,coords2), 'Test cache labels/order differ')
        provenance = {'kind':'original_B_locked_test_cache','sha256':side['sha256']}
    else:
        candidates = list(Path(cfg['inputs_root']).rglob('*pines*corrected*.mat'))
        paths = [p for p in candidates if sha_file(p) == HSI_SHA]
        require(paths, 'All training saved. Test cache missing: attach exact Indian_pines_corrected.mat to reconstruct with saved PCA/scaler; no PCA refit.')
        prep = root/'final_data/preprocessing_final.joblib'
        require(sha_file(prep) == PREP_SHA, 'Wrong original final preprocessing')
        import joblib
        from scipy.io import loadmat
        state = joblib.load(prep)
        pca,scaler = state['pca'],state['scaler']
        require(pca.n_samples_ == 5528 and pca.n_components_ == 40, 'Wrong fitted PCA source')
        cube = loadmat(paths[0])['indian_pines_corrected'].astype(np.float32)
        require(cube.shape == (145,145,200), 'Wrong raw cube')
        processed = scaler.transform(pca.transform(cube.reshape(-1,200).astype(np.float64))).reshape(145,145,40).astype(np.float32)
        padded = np.pad(processed,((5,5),(5,5),(0,0)),mode='constant')
        X = np.empty((4721,11,11,40,1),np.float32)
        for i,(r,c) in enumerate(coords):
            X[i,...,0] = padded[r:r+11,c:c+11]
        provenance = {'kind':'original_B_preprocessing_no_refit','raw_sha256':HSI_SHA,'preprocessing_sha256':PREP_SHA}
    require(X.shape == (4721,11,11,40,1) and X.dtype == np.float32 and np.isfinite(X).all(), 'Invalid heldout features')
    provenance['X_B_sha256'] = b_array_sha(X)
    return X,y,coords,provenance


def evaluate(cfg):
    base = Path(cfg['base'])
    worker_sha = sha_file(Path(__file__).with_name('v3a_worker.py'))
    registry = [verify_job(base, s, worker_sha) for s in (42,43,44)]
    out = base/'test_results'; out.mkdir(exist_ok=True)
    completed = out/'manifest.json'
    if completed.is_file():
        old = read_json(completed)
        if old.get('complete') is True:
            require(all(sha_file(out/p['file']) == p['sha256'] for p in old['predictions']), 'Existing prediction seal mismatch')
            require([sha_file(j/'classifier.weights.h5') for j,_,_ in registry] == old['classifier_sha256'], 'Existing test belongs to different classifiers')
            print('TEST ALREADY COMPLETE: saved predictions retained; no inference repeated', flush=True)
            return old
    write_json(base/'test_gate.json', {'all_three_complete':True,'contracts':[sha_file(j/'contract.json') for j,_,_ in registry], 'test_opened_after_gate':True})
    X,y,coords,provenance = locked_test(cfg)
    reference = read_json(Path(__file__).with_name('baseline_test_results.json'))
    require(array_sha(y)==reference['shared_y_array_sha256'] and array_sha(coords)==reference['shared_coords_array_sha256'],'New test labels/order differ from historical M4 comparison')
    require(b_array_sha(X)==reference['test_data_provenance']['X_test_B_sha256'],'New test feature cache differs from historical M4 comparison')
    w = import_module('_v3a_eval_m4', cfg['m4_worker_path'])
    w.load_runtime()
    rows, pcs, seals = [], [], []
    for seed,(job,contract,done) in zip((42,43,44),registry):
        w.keras.backend.clear_session(); w.keras.utils.set_random_seed(seed)
        w.PAIRED_C_DROPOUT,_ = w.recorded_dropout_factory(64,'C')
        C = w.tao_classifier(w.SHAPE,16)
        C.load_weights(str(job/'classifier.weights.h5'))
        sorted_prob = C.predict(X,batch_size=48,verbose=0)
        order = np.asarray(contract['class_order'],np.int64)-1
        require(np.array_equal(np.sort(order),np.arange(16)), 'Wrong C output rank mapping')
        prob = np.empty_like(sorted_prob); prob[:,order] = sorted_prob
        require(np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1)) and np.allclose(prob.sum(1),1,atol=2e-6), 'Invalid test probabilities')
        result,perclass,pred,cm = metrics(y,prob)
        rows.append({'method':'V3-A','seed':seed,**result})
        pcs.extend({'method':'V3-A','seed':seed,**r} for r in perclass)
        dest = out/f'V3A_seed{seed}_predictions.npz'
        np.savez_compressed(dest,y_true=y,y_pred=pred,probabilities=prob,coords=coords,confusion_matrix=cm)
        np.savetxt(out/f'V3A_seed{seed}_confusion.csv',cm,delimiter=',',fmt='%d',header=','.join(str(i) for i in range(1,17)),comments='')
        seals.append({'seed':seed,'file':dest.name,'sha256':sha_file(dest)})
        print('TEST',seed,json.dumps(result),flush=True)
    csv_write(out/'v3_metrics_by_seed.csv', rows)
    csv_write(out/'v3_per_class.csv',pcs)
    allrows = []
    for r in reference['rows']:
        allrows.append({'method':r['arm'],'seed':r['seed'],**{k:r[k] for k in ('OA','AA','Macro_F1','GM','Kappa')},
            'NLL':r['NLL_recomputed_float64_clip1e-12'],'Brier':r['Brier_recomputed_float64_sum16']})
    allrows.extend(rows)
    csv_write(out/'all_methods_historical_reference.csv',allrows)
    summary = []
    for method in sorted({r['method'] for r in allrows}):
        rr = [r for r in allrows if r['method']==method]
        summary.append({'method':method,**{key:float(v) for metric in METRICS for key,v in ((metric+'_mean',np.mean([r[metric] for r in rr])),(metric+'_sample_SD',np.std([r[metric] for r in rr],ddof=1)))}})
    csv_write(out/'summary_mean_sample_SD.csv',summary)
    deltas = []
    for r in rows:
        ref = next(p for p in allrows if p['method']=='learned_local' and p['seed']==r['seed'])
        deltas.append({'seed':r['seed'],'comparison':'V3-A minus historical full M4 V2',**{k:r[k]-ref[k] for k in METRICS}})
    csv_write(out/'v3_minus_historical_full_v2.csv',deltas)
    result = {'complete':True,'status':'COMPLETE_3_OF_3_V3_CLASSIFIERS','protocol_B_id':PROTOCOL,
        'test_samples':4721,'test_source':provenance,'test_previously_viewed':True,
        'comparison':'Historical full M4 V2; matched recipe, paired-identical not claimed',
        'NLL_clip':1e-12,'NLL_Brier_dtype':'float64','predictions':seals,
        'classifier_sha256':[sha_file(j/'classifier.weights.h5') for j,_,_ in registry],
        'mean_macro_F1_delta_vs_V2':float(np.mean([r['Macro_F1'] for r in deltas])),
        'wins_vs_V2':int(sum(r['Macro_F1']>0 for r in deltas)),
        'test_used_for_tuning':False,'independent_spatial_generalization_claimed':False}
    write_json(completed,result)
    lines = ['# V3-A — kết quả một lượt chạy','', '**COMPLETE: 3/3 V3 classifiers, mỗi seed 8.000 C updates.**','',
        '| Method | OA (%) | AA (%) | Macro-F1 (%) | GM (%) | Kappa |','|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('| '+r['method']+' | '+' | '.join(f"{100*r[m+'_mean']:.3f} ± {100*r[m+'_sample_SD']:.3f}" for m in ('OA','AA','Macro_F1','GM'))+f" | {r['Kappa_mean']:.4f} ± {r['Kappa_sample_SD']:.4f} |")
    lines += ['',f"V3-A − full M4 V2: mean Macro-F1 {100*result['mean_macro_F1_delta_vs_V2']:+.3f} điểm phần trăm; thắng {result['wins_vs_V2']}/3 seed.",'',
        'Báo đủ từng seed và cả kết quả không thuận lợi. Baseline là full M4 V2, không phải C-only endpoint. Đây là đối chiếu hậu nghiệm trong Protocol B có chồng lấn không gian; không tuyên bố paired-identical, tính mới, ý nghĩa thống kê hay lợi ích nhân quả riêng của distance bias. NLL dùng clipping 1e-12, float64 cho cả baseline và V3.', '',
        'Mỗi classifier có NPZ chứa y_true, y_pred, probabilities, coords và confusion_matrix; metrics/per-class/mean±sampleSD/deltas đều lưu riêng. Không chọn seed tốt nhất hoặc tự sửa mô hình sau test.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


class Backup:
    """Capture only committed native states, then ZIP an immutable RAM snapshot."""
    def __init__(self, base):
        self.base = Path(base)
        self.cache = {}

    def create(self):
        snapshot = {}
        for p in self.base.rglob('*'):
            if not p.is_file():
                continue
            rel = p.relative_to(self.base)
            if 'state' in rel.parts or any(x.endswith(('.tmp','.incomplete','.pyc')) for x in rel.parts) or '__pycache__' in rel.parts:
                continue
            if p.name == 'classifier.weights.h5':
                marker = p.parent/'completed.json'
                if not marker.is_file() or sha_file(p) != read_json(marker)['model_sha256']:
                    continue
            try:
                snapshot[rel.as_posix()] = p.read_bytes()
            except OSError:
                continue
        for job in sorted((self.base/'jobs').glob('seed_*')) if (self.base/'jobs').exists() else []:
            ptr = job/'state/latest.json'
            if not ptr.is_file():
                continue
            pointer={}
            try:
                pointer_bytes = ptr.read_bytes(); pointer = json.loads(pointer_bytes)
                cp = ptr.parent/pointer['name']
                meta_bytes = (cp/'meta.json').read_bytes()
                require(sha_bytes(meta_bytes)==pointer['meta_sha256'],'Checkpoint commit changed during snapshot')
                meta = json.loads(meta_bytes)
                chunk = {ptr.relative_to(self.base).as_posix():pointer_bytes,
                         (cp/'meta.json').relative_to(self.base).as_posix():meta_bytes}
                for name,pin in meta['weights_hashes'].items():
                    data=(cp/name).read_bytes()
                    require(sha_bytes(data)==pin,'Native weights changed during snapshot')
                    chunk[(cp/name).relative_to(self.base).as_posix()]=data
                self.cache[job.name]=chunk
            except (OSError,ValueError,RuntimeError,KeyError):
                if job.name not in self.cache:
                    continue
            snapshot.update(self.cache.get(job.name,{}))
            if not pointer:
                cached=self.cache.get(job.name,{}).get(ptr.relative_to(self.base).as_posix())
                pointer=json.loads(cached) if cached else {}
            # Retain pilot state and the immediately previous committed checkpoint when available.
            for cp in sorted((job/'state').glob('ckpt_*')):
                if not cp.is_dir() or cp.name.endswith('.incomplete') or cp.name==pointer.get('name'):
                    continue
                try:
                    mb=(cp/'meta.json').read_bytes(); mm=json.loads(mb)
                    extra={(cp/'meta.json').relative_to(self.base).as_posix():mb}
                    for n,pin in mm['weights_hashes'].items():
                        bb=(cp/n).read_bytes();require(sha_bytes(bb)==pin,'Previous checkpoint hash mismatch')
                        extra[(cp/n).relative_to(self.base).as_posix()]=bb
                    snapshot.update(extra)
                except (OSError,ValueError,KeyError,RuntimeError):
                    continue
        output=self.base.parent/'V3A_IP_ONERUN_FULL_BACKUP.zip';temp=output.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=2,allowZip64=True) as z:
            for name,data in sorted(snapshot.items()):
                z.writestr(name,data)
        temp.replace(output)
        print('BACKUP SAVED:',output,'| MB:',round(output.stat().st_size/1048576,1),flush=True)
        return output

    def try_create(self):
        try:
            return self.create()
        except (OSError,ValueError,KeyError,RuntimeError,zipfile.BadZipFile) as exc:
            print('BACKUP RETRY LATER:',type(exc).__name__,str(exc),'| live checkpoint files remain saved',flush=True)
            return None


def expected_run_identity(code):
    code=Path(code)
    return {'method':'V3-A_signed_distance_bias_v1','protocol_B_id':PROTOCOL,'seeds':[42,43,44],
        'target_C_updates':8000,'pilot_C_updates':200,'k':3,'rho':.2,'distance_epsilon':1e-8,
        'v3_worker_sha256':sha_file(code/'v3a_worker.py'),'orchestrator_sha256':sha_file(__file__),
        'm4_worker_sha256':M4_SHA,'baseline_pins_sha256':sha_file(code/'baseline_pins.json'),
        'baseline_test_sha256':sha_file(code/'baseline_test_results.json')}


def find_resume(inputs_root, identity):
    """Support both original ZIP and Kaggle-auto-unpacked prior backups."""
    root=Path(inputs_root);matches=[];incompatible=[]
    candidates=[ArtifactSource(p.parent) for p in root.rglob('run_identity.json')]
    for p in root.rglob('*.zip'):
        try:
            with zipfile.ZipFile(p) as z:
                prefixes=[n[:-len('run_identity.json')] for n in z.namelist() if n.endswith('run_identity.json')]
            candidates.extend(ArtifactSource(p,prefix,True) for prefix in prefixes)
        except (OSError,zipfile.BadZipFile):
            continue
    for s in candidates:
        try:
            old=json.loads(s.read('run_identity.json'))
            if old.get('method')!=identity['method']:
                continue
            if old!=identity:
                incompatible.append(s.label());continue
            score=0
            for seed in (42,43,44):
                path=f'jobs/seed_{seed}/state/latest.json'
                if s.exists(path):
                    score+=int(json.loads(s.read(path)).get('c_updates',0))
            matches.append((score,s.label(),s))
        except (OSError,ValueError,KeyError,zipfile.BadZipFile):
            continue
    if matches:
        return sorted(matches,key=lambda x:(-x[0],x[1]))[0][2]
    require(not incompatible,'A V3-A backup was found but its source/config differs. Use its original notebook to resume; no fresh run was silently started. '+str(incompatible[:3]))
    return None


def restore_backup(base, source, identity):
    base=Path(base)
    if source is None:
        return
    if not isinstance(source,ArtifactSource):
        path=Path(source)
        source=ArtifactSource(path,is_zip=path.is_file())
    prior=json.loads(source.read('run_identity.json'))
    require(prior==identity,'Resume source/config identity differs; no checkpoint is modified')
    if source.is_zip:
        with zipfile.ZipFile(source.path) as z:
            require(z.testzip() is None,'Corrupt resume archive')
            names=[n.filename[len(source.prefix):] for n in z.infolist() if n.filename.startswith(source.prefix) and not n.is_dir()]
    else:
        names=[p.relative_to(source.path).as_posix() for p in source.path.rglob('*') if p.is_file()]
    for name in names:
        relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in name and ':' not in name,'Unsafe resume member')
        if name in ('config.json','manifest.json','b_source.json'):
            continue
        target=base/relative;data=source.read(name)
        if target.exists():
            require(target.read_bytes()==data,'Conflicting resume file '+str(relative))
        else:
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    print('RESTORED compatible V3-A backup:',source.label(),flush=True)


def launch(cmd, gpu, log_path):
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),TF_USE_LEGACY_KERAS='1',TF_DETERMINISTIC_OPS='1',PYTHONUNBUFFERED='1',TF_CPP_MIN_LOG_LEVEL='2')
    log_path=Path(log_path);log_path.parent.mkdir(parents=True,exist_ok=True)
    log=log_path.open('a',encoding='utf-8')
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
    messages=queue.Queue()
    def consume():
        try:
            for line in proc.stdout:
                log.write(line);log.flush();messages.put(line)
        finally:
            log.close()
    thread=threading.Thread(target=consume,daemon=True);thread.start()
    return {'process':proc,'queue':messages,'thread':thread,'start':time.monotonic(),'log':str(log_path)}


def drain(task):
    while True:
        try:
            print(task['queue'].get_nowait(),end='',flush=True)
        except queue.Empty:
            return


def run_single(cmd,gpu,log_path,deadline,backup):
    task=launch(cmd,gpu,log_path)
    last_backup=time.monotonic()
    try:
        while task['process'].poll() is None:
            drain(task)
            if time.monotonic()>deadline:
                raise TimeoutError('Session/worker deadline reached; committed states saved')
            if time.monotonic()-last_backup>300:
                backup.try_create();last_backup=time.monotonic()
            time.sleep(.25)
        task['thread'].join(timeout=10);drain(task)
        require(task['process'].returncode==0,'Worker failed; read '+task['log'])
        backup.try_create()
    finally:
        if task['process'].poll() is None:
            task['process'].terminate()
            try:task['process'].wait(30)
            except subprocess.TimeoutExpired:
                task['process'].kill();task['process'].wait(30)
        task['thread'].join(timeout=10);drain(task)


def run_all(options):
    code=Path(__file__).resolve().parent
    base=Path(options['base']);base.mkdir(parents=True,exist_ok=True)
    backup=Backup(base)
    start=time.monotonic();deadline=start+float(options['session_hours'])*3600
    identity=expected_run_identity(code)
    status={'status':'STARTED','complete':False,'test_evaluated':False,'protocol_B_id':PROTOCOL,'method':'V3-A','test_previously_viewed':True,'paired_identical_claimed':False}
    try:
        require(sys.platform.startswith('linux') and Path('/kaggle/working').is_dir(),'This runner is for Kaggle GPU. Local execution is disabled.')
        if (base/'run_identity.json').exists():
            require(read_json(base/'run_identity.json')==identity,'Existing run has a different source/configuration')
        else:
            write_json(base/'run_identity.json',identity)
        resume=options.get('resume_zip')
        if resume is None and not (base/'jobs').exists():
            resume=find_resume(options['inputs_root'],identity)
        restore_backup(base,resume,identity)
        write_json(base/'run_identity.json',identity)
        # Keep all source and baseline evidence in every downloadable backup.
        for p in code.glob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.md'):
                dest=base/'source'/p.name;dest.parent.mkdir(exist_ok=True)
                if not dest.exists():shutil.copyfile(p,dest)
                else:require(sha_file(dest)==sha_file(p),'Saved source changed '+p.name)
        source=discover_b(options['inputs_root'],options.get('b_source'))
        write_json(base/'test_input_preflight.json',test_input_preflight(source,options['inputs_root']))
        b_root=prepare_train_source(source,base.parent/'_v3a_b_train')
        write_json(base/'b_source.json',{'path':str(source.path),'prefix':source.prefix,'is_zip':source.is_zip,'test_arrays_loaded':False})
        metas={}
        native=read_json(code/'baseline_pins.json')
        for seed in (42,43,44):
            p=code/f'original_m4_seed{seed}_meta.json'
            require(sha_file(p)==native['seeds'][str(seed)]['meta_sha256'],'Embedded original M4 metadata changed')
            metas[str(seed)]=str(p)
        cfg={'base':str(base),'b_run':str(b_root),'m4_worker_path':str(code/'m4_training_worker_frozen.py'),
            'original_checkpoint_metas':metas,'inputs_root':options['inputs_root'],
            'max_worker_seconds':min(3*3600,max(600,deadline-time.monotonic()-300)),
            'checkpoint_safety_seconds':180}
        config_path=base/'config.json';write_json(config_path,cfg)
        cpu_audit(cfg)
        devices=gpu_inventory()[:int(options['max_gpu_workers'])]
        write_json(base/'gpu_inventory.json',devices)
        print('ONE RUN: audit → pilot200 → full8000 x3 → test → full backup',flush=True)
        worker=[sys.executable,'-u',str(code/'v3a_worker.py')]
        pilot_marker=base/'jobs/seed_42/pilot_gate.json'
        restore_marker=base/'jobs/seed_42/pilot_restore_verification.json'
        if not restore_marker.is_file():
            # A gate may have been written before an interrupted checkpoint commit.
            # Finish/validate the same 200-update prefix before fresh restore.
            cfg['max_worker_seconds']=min(1800,max(60,deadline-time.monotonic()-300));write_json(config_path,cfg)
            run_single(worker+['train','--config',str(config_path),'--seed','42','--stop-updates','200'],devices[0]['index'],base/'logs/pilot42.log',min(deadline-180,time.monotonic()+cfg['max_worker_seconds']+180),backup)
        require(read_json(pilot_marker).get('pass') is True,'PILOT FAILED: full training remains locked')
        if not restore_marker.is_file():
            run_single(worker+['verify-pilot','--config',str(config_path),'--seed','42'],devices[0]['index'],base/'logs/verify_pilot42.log',min(deadline-180,time.monotonic()+1800),backup)
        require(read_json(restore_marker).get('pass') is True,'Pilot fresh restore verification failed')
        status['pilot_pass']=True;status['pilot_restore_pass']=True
        print('PILOT + NATIVE RESTORE PASS; continuing same 8000-update contracts',flush=True)
        pending=[s for s in (42,43,44) if not (base/'jobs'/f'seed_{s}'/'completed.json').is_file()]
        running={};last_backup=time.monotonic();last_heartbeat=time.monotonic()
        while pending or running:
            if time.monotonic()>deadline-300:
                status['status']='PAUSED_SESSION_LIMIT';break
            for device in devices:
                gpu=device['index']
                if gpu not in running and pending:
                    seed=pending.pop(0)
                    cfg['max_worker_seconds']=min(3*3600,max(60,deadline-time.monotonic()-480));write_json(config_path,cfg)
                    task=launch(worker+['train','--config',str(config_path),'--seed',str(seed),'--stop-updates','8000'],gpu,base/f'logs/train_seed{seed}.log')
                    progress_path=base/'jobs'/f'seed_{seed}'/'progress.json'
                    task['previous_updates']=read_json(progress_path).get('c_updates',0) if progress_path.is_file() else 0
                    task['seed']=seed;running[gpu]=task
                    print('START FULL seed',seed,'GPU',gpu,flush=True)
            for gpu,task in list(running.items()):
                drain(task)
                rc=task['process'].poll()
                if rc is not None:
                    task['thread'].join(timeout=10);drain(task)
                    require(rc==0,'Full train worker failed: '+task['log'])
                    done=base/'jobs'/f"seed_{task['seed']}"/'completed.json'
                    if not done.is_file():
                        progress_path=done.parent/'progress.json'
                        pg=read_json(progress_path) if progress_path.is_file() else {}
                        require(pg.get('status')=='PAUSED_WALLCLOCK' and pg.get('c_updates',0)>task['previous_updates'],'Worker stopped without completion or resumable progress')
                        pending.insert(0,task['seed']);del running[gpu]
                        backup.try_create()
                        print('AUTO RESUME NEXT: seed',task['seed'],'from',pg['c_updates'],'C updates; no restart from zero',flush=True)
                        continue
                    verify_job(base,task['seed'],identity['v3_worker_sha256'])
                    del running[gpu];backup.try_create();last_backup=time.monotonic()
            if time.monotonic()-last_backup>300:
                backup.try_create();last_backup=time.monotonic()
            if time.monotonic()-last_heartbeat>60:
                print('GPU workers:',{gpu:t['seed'] for gpu,t in running.items()},'| pending seeds:',pending,'| native checkpoints saved automatically',flush=True)
                last_heartbeat=time.monotonic()
            time.sleep(.25)
        if running:
            for task in running.values():
                task['process'].terminate()
                try:task['process'].wait(30)
                except subprocess.TimeoutExpired:task['process'].kill()
                drain(task)
        require(not pending and not running,'Session paused; test remains locked')
        registry=[verify_job(base,s,identity['v3_worker_sha256']) for s in (42,43,44)]
        status['completed_model_count']=len(registry)
        prepare_test_source(source,b_root)
        run_single([sys.executable,'-u',str(Path(__file__)),'evaluate','--config',str(config_path)],devices[0]['index'],base/'logs/test.log',deadline-30,backup)
        test=read_json(base/'test_results/manifest.json')
        require(test.get('complete') is True and len(test['predictions'])==3,'Test not complete')
        status.update(status='COMPLETE',complete=True,test_evaluated=True,completed_model_count=3,
            mean_macro_F1_delta_vs_full_M4_V2=test['mean_macro_F1_delta_vs_V2'],wins_vs_full_M4_V2=test['wins_vs_V2'])
        print('ALL COMPLETE: 3 trained V3 classifiers + test + historical baseline comparison',flush=True)
    except BaseException as exc:
        status.update(status=status['status'] if str(status['status']).startswith('PAUSED') else 'INCOMPLETE_FAILED_CHECK',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        # Terminate every still-running child before snapshot on any failure.
        for task in locals().get('running',{}).values():
            if task['process'].poll() is None:
                task['process'].terminate()
                try:task['process'].wait(30)
                except subprocess.TimeoutExpired:task['process'].kill()
        status['elapsed_seconds']=round(time.monotonic()-start,2)
        write_json(base/'manifest.json',status)
        try:
            output=backup.create()
            print('FINAL STATUS:',status['status'],'| complete:',status['complete'],'| DOWNLOAD:',output,flush=True)
        except Exception as pack_error:
            print('BACKUP ERROR:',repr(pack_error),'| files remain in',base,flush=True)
    return status


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['full','evaluate'])
    parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=read_json(args.config)
    if args.mode=='evaluate':
        evaluate(cfg)
    else:
        run_all(cfg)


if __name__=='__main__':
    main()
