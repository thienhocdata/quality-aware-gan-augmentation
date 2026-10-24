"""M4 classifier-only final test. Portable paths; no training or G/D models."""
from __future__ import annotations
import csv
import gc
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

import numpy as np

CONFIG=REPORT=OUT=None
START=time.monotonic()
ARMS=('original_gan','matched_ros','random_local','learned_local')
SEEDS=(42,43,44)
LABELS={'original_gan':'Original GAN','matched_ros':'Matched ROS',
        'random_local':'Random-local','learned_local':'Learned-local V2'}
CLASS_NAMES=['Alfalfa','Corn-notill','Corn-mintill','Corn','Grass-pasture',
             'Grass-trees','Grass-pasture-mowed','Hay-windrowed','Oats',
             'Soybean-notill','Soybean-mintill','Soybean-clean','Wheat','Woods',
             'Buildings-Grass-Trees-Drives','Stone-Steel-Towers']
METRICS=('OA','AA','Macro_F1','GM','Kappa')
SHAPE=(11,11,40,1)
RUN_ID='50e988fae260'
IMPLEMENTATION='5e15f8214bf273bb889187b96c80407a8000e5e1a3ef30734bcfa86ced275a6f'
WORKER_TRAIN_SHA='6d33ca6a0d39600f5f44c0d7e63547b1237149d625471ac73315d982ffbd8fb5'
MANIFEST_SHA='41486ba39177e36f32e09c61cff870f4086a94c7b55f39db9dbe4b92093d6e67'
PROTOCOL_SHA='d4260da1ba9289a3183ab10905f995d6a2e8956b5651576734683017705bd202'
PREP_SHA='18167b01cbb8ac5d21ff73a3588cad758d23a2629f9b1c9e47dbaf4cf6989939'
HSI_SHA='ec2f8808710919d566f70f0d4aa885aae1ddfd42b734aba71c5e12ca65450939'
TRAIN_X_SHA='26cb18329d396d46e9d639420f5c718ae98c9cd938de5fa6c35240bc6d6227b0'
TRAIN_Y_SHA='28d08700b81863a431bf78230196387afe755ee37c4a69a47e871068e7d25921'
TRAIN_COUNTS=[21,753,426,138,209,376,16,228,10,469,1390,311,125,720,287,49]
TEST_COUNTS=[25,675,404,99,274,354,12,250,10,503,1065,282,80,545,99,44]

MODEL_PINS = {'original_gan/seed_42': {'completed_sha256': '06bb7214b20881b4ca957a3827fcf431958414b4ea44a76be58de6380d9f7758', 'contract_sha256': '67e3455823f40b7a7a119846b8c8ca3f824c3c822670572319bf9fe2395988a2', 'classifier_sha256': 'ab80e7fe8b4b5105c0da875898acf4216a5a2adf9f3625d77e0ddfbee8287077', 'class_order': [9, 7, 1, 16, 13, 4, 5, 8, 15, 12, 6, 3, 10, 14, 2, 11]}, 'matched_ros/seed_42': {'completed_sha256': '1537aeb7608aa22aadc2a4567b10e93774ae65668eb561445f615f40043792a3', 'contract_sha256': 'ef184c983b47c5c21863c8f3109563d88e2be74aac41b22f034aa783851d6864', 'classifier_sha256': '1f46533c3134f0c47fca55fa4ed2fae1902fdaa0573b807dff519e3dfdb6d4db', 'class_order': [9, 7, 1, 16, 13, 4, 5, 8, 15, 12, 6, 3, 10, 14, 2, 11]}, 'random_local/seed_42': {'completed_sha256': '0518b97ede2442ad742a7eb969b53050ca2c0b48b4793e4e3c75341ec6461e80', 'contract_sha256': '595b6f68a9e9b1c903d33c0c366ad039856f2b2efc578734e99efcc04de5587f', 'classifier_sha256': 'dc2b66301de596896341979ae32febba1f9641ce0370bc539799a4cfcade2a3e', 'class_order': [9, 7, 1, 16, 13, 4, 5, 8, 15, 12, 6, 3, 10, 14, 2, 11]}, 'learned_local/seed_42': {'completed_sha256': '0a6d847bb5cb9a230d4e8387ad8b258f6bd773bad992bfd6dc8a51b2c7f5363f', 'contract_sha256': '2d1f8d23b0d5b4e97eabd3374bb05055024663473f3e7149ba80f30e66c8f021', 'classifier_sha256': '7c56b275e3f2115758c78bc4d9a026b4821e757a25fe9472f5b3c20c611beffd', 'class_order': [9, 7, 1, 16, 13, 4, 5, 8, 15, 12, 6, 3, 10, 14, 2, 11]}, 'original_gan/seed_43': {'completed_sha256': '1626cabd6c92ed47bc81c481b5f228666c4f876b829782250e3e005362c92fd3', 'contract_sha256': 'df2479d2ccf2d6970d88d2fd5a5062ff9d1f1ba5f749899f516e0573d6b01c7a', 'classifier_sha256': '6415ef4550d5b16adb86ae8f383fd4a1baf7a72f0d2fea7f52022e3e66f6adca', 'class_order': [9, 7, 1, 16, 13, 4, 5, 8, 15, 12, 6, 3, 10, 14, 2, 11]}, 'matched_ros/seed_43': {'completed_sha256': 'd5e8574d12fd335f0d3bbbafcbc3c7dca4c3fc1a7d9ce38a82309f80b13fead5', 'contract_sha256': '5caa2e3ff31d0bfdda50770cf5d52b605aa7490a00347737f494cac7f5658ba2', 'classifier_sha256': '2e64638157b6061c50cb6bfeee6f24500c917cff702dfc8f8c27b9a14ac9d455', 'class_order': [9, 7, 1, 16, 13, 4, 5, 8, 15, 12, 6, 3, 10, 14, 2, 11]}, 'random_local/seed_43': {'completed_sha256': 'ff93ee01f5d8c7796aa220f1ff57fd599ace1612ac5e0294a97be690ec753a4b', 'contract_sha256': '4ef525ba1e3498a9e9452fe378ace4f653f38f5136aaee9b7e781e02952a73d0', 'classifier_sha256': 'c1041d79bf28e1818b25481cb9c6b03b1e7413816ebcfbf48d5057ff71aaf949', 'class_order': [9, 7, 1, 16, 13, 4, 5, 8, 15, 12, 6, 3, 10, 14, 2, 11]}, 'learned_local/seed_43': {'completed_sha256': '7f9221ec4cb5f197b1105006cacb21ec212bafa57af2a43aa8f0376b888eb132', 'contract_sha256': '9102a883e9738f2ef9fe1cfc53e43bbb76e5199448a10825ac2e192950b1e0a7', 'classifier_sha256': '3901ceebebbbd86e1f1ecdd50e8b261f5cb5573873916d74b4a06de61ee22cd5', 'class_order': [9, 7, 1, 16, 13, 4, 5, 8, 15, 12, 6, 3, 10, 14, 2, 11]}, 'original_gan/seed_44': {'completed_sha256': 'bf643795353fcd64222e056a07e428aa5ba2ee9468947cca5c18a50dab6e3b57', 'contract_sha256': 'bc85ec61d4f53d33b314635a94a440f0d5d44bfb461d4b9acd3562272ca10a7a', 'classifier_sha256': 'dc3300445e271c85dd35c7cfbddbb1e88c52c012d58786e52b7a8ffe30390954', 'class_order': [9, 7, 1, 16, 13, 4, 5, 8, 15, 12, 6, 3, 10, 14, 2, 11]}, 'matched_ros/seed_44': {'completed_sha256': '536a2a877d0322d574430927bdef65af9660901b5a95674ca6d1e7d7616ab57c', 'contract_sha256': '21ff32854fc83330d4d84125c6b22caea07f7c5ec11658248b4a0e21cb25f571', 'classifier_sha256': 'e6d4862e55ebabc4b778a7a76e8823442ba97729e1445f8be3f62562ec34847b', 'class_order': [9, 7, 1, 16, 13, 4, 5, 8, 15, 12, 6, 3, 10, 14, 2, 11]}, 'random_local/seed_44': {'completed_sha256': 'c83412ea8cd592853acfa418af88ad8a0b11d76f4bce5bf990a6e24e2d1e94e9', 'contract_sha256': '939a0667a55ecc6314c94615f8bc89380c60fab7ea024d5aee8e486db45775da', 'classifier_sha256': '446a937693d11fd9d0c9fa871a81dd6b486590eaeb9a1569c578984068ec8f5e', 'class_order': [9, 7, 1, 16, 13, 4, 5, 8, 15, 12, 6, 3, 10, 14, 2, 11]}, 'learned_local/seed_44': {'completed_sha256': 'bd145c69980c0b1a6dc1f4025a0b5157dc28e3e92cbf3e47c8185d839066afa9', 'contract_sha256': 'bc52fead2233f9de6c5c4ad8a24a5dfd0c11269a72b77c5d869622adfa2cf037', 'classifier_sha256': 'cd3903e97842ec6705cfadbcbb8e8f2befe1a3039a3d7565dec665bea20de342', 'class_order': [9, 7, 1, 16, 13, 4, 5, 8, 15, 12, 6, 3, 10, 14, 2, 11]}}
def tao_classifier(shape, so_lop=16):
    L = keras.layers
    inp = L.Input(shape=shape)
    x = L.Conv3D(8, (1,1,7), activation="relu", padding="same")(inp)
    x = L.Conv3D(16, (3,3,5), activation="relu", padding="same")(x)
    x = L.Conv3D(32, (5,5,7), activation="relu", padding="same")(x)
    s = x.shape
    x = L.Reshape((s[1], s[2], s[3] * s[4]))(x)
    x = L.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.5)(x)
    out = L.Dense(so_lop, activation="softmax", kernel_initializer="he_normal")(x)
    return keras.Model(inp, out, name="classifier_3d_hypergamo")


def require(condition,message):
    if not condition:raise RuntimeError(message)


def deadline():
    if CONFIG and time.monotonic()>=CONFIG['deadline_monotonic']:
        raise TimeoutError('Reached the total test deadline; partial predictions/reports were saved. No training was run.')


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def sha_file(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(4*1024*1024),b''):
            deadline();result.update(chunk)
    return result.hexdigest()


def b_array_sha(a):
    a=np.ascontiguousarray(a)
    result=hashlib.sha256()
    result.update(str(a.shape).encode());result.update(a.dtype.str.encode());result.update(a.tobytes())
    return result.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def phase(message):
    deadline();REPORT['phase']=message
    REPORT['elapsed_seconds']=round(time.monotonic()-START,2)
    save_json(OUT/'manifest.json',REPORT)
    print(message,flush=True)


def search_roots():
    roots=[Path(p).expanduser() for p in CONFIG['search_roots']]
    return sorted({p.resolve() for p in roots if p.is_dir()},key=str)


def find_bundle():
    if CONFIG['model_source']:
        path=Path(CONFIG['model_source']).expanduser()
        require(path.is_file() or path.is_dir(),f'MODEL_SOURCE does not exist: {path}')
        return path
    candidates=[]
    for root in search_roots():
        candidates.extend(p for p in root.rglob('*.zip') if p.name.lower()=='m4_ip_12_classifiers_for_test.zip')
        candidates.extend(p.parent for p in root.rglob('bundle_manifest.json'))
    valid=[]
    probe='original_gan/seed_42/completed.json'
    for path in sorted(set(candidates),key=str):
        try:
            if path.is_file():
                with zipfile.ZipFile(path) as z:payload=z.read(probe)
            else:payload=(path/probe).read_bytes()
            if hashlib.sha256(payload).hexdigest()==MODEL_PINS['original_gan/seed_42']['completed_sha256']:
                valid.append(path)
        except (OSError,KeyError,zipfile.BadZipFile):continue
    require(valid,'Missing recognized classifier bundle. Upload M4_IP_12_CLASSIFIERS_FOR_TEST.zip as one Kaggle Input, or set MODEL_SOURCE to the local ZIP/folder.')
    return valid[0]


def obtain_models():
    """Copy only the pinned classifier files; no arbitrary archive extraction."""
    source=find_bundle();REPORT['model_source']=str(source)
    archive=zipfile.ZipFile(source) if source.is_file() else None
    try:
        def raw(name):return archive.read(name) if archive else (source/name).read_bytes()
        registry=[]
        for arm in ARMS:
            for seed in SEEDS:
                deadline();key=f'{arm}/seed_{seed}';prefix=key+'/'
                done_bytes=raw(prefix+'completed.json');contract_bytes=raw(prefix+'contract.json')
                done=json.loads(done_bytes);contract=json.loads(contract_bytes)
                pin=MODEL_PINS[key]
                require(hashlib.sha256(done_bytes).hexdigest()==pin['completed_sha256'],f'Unrecognized completed metadata: {key}')
                require(hashlib.sha256(contract_bytes).hexdigest()==done['job_contract_sha256']==pin['contract_sha256'],f'Contract mismatch: {key}')
                require(done['status']=='COMPLETE' and done['complete'] is True and done['stage']=='final',f'Incomplete/non-final model: {key}')
                require(done['arm']==arm and done['seed']==seed and done['c_updates']==8000 and done['adam_clock']==136000,f'Wrong identity/budget: {key}')
                require(done['worker_sha256']==contract['sha_worker']==WORKER_TRAIN_SHA,f'Wrong training worker: {key}')
                require(done['source_train_X_sha256']==contract['train_X_sha256']==TRAIN_X_SHA,f'Wrong train X: {key}')
                require(done['source_train_y_sha256']==contract['train_y_sha256']==TRAIN_Y_SHA,f'Wrong train y: {key}')
                order=np.asarray(done['class_order_original_ids'],np.int64)
                require(np.array_equal(np.sort(order),np.arange(1,17)) and order.tolist()==contract['class_order'],f'Class mapping invalid: {key}')
                require(order.tolist()==pin['class_order'],f'Unrecognized class order: {key}')
                target=OUT/'_classifiers'/arm/f'seed_{seed}'/'classifier.weights.h5'
                target.parent.mkdir(parents=True,exist_ok=True)
                digest=hashlib.sha256()
                with (archive.open(prefix+'classifier.weights.h5') if archive else (source/prefix/'classifier.weights.h5').open('rb')) as stream,target.open('wb') as sink:
                    for chunk in iter(lambda:stream.read(4*1024*1024),b''):
                        deadline();digest.update(chunk);sink.write(chunk)
                require(digest.hexdigest()==done['model_sha256']==pin['classifier_sha256'],f'Classifier weight hash mismatch: {key}')
                registry.append({'arm':arm,'seed':seed,'key':key,'classifier_sha256':digest.hexdigest(),
                                 'contract_sha256':done['job_contract_sha256'],'class_order':order.tolist(),
                                 'checkpoint_C_updates':8000,'weights_path':str(target),
                                 'training_runtime':{k:contract[k] for k in ('tf','tf_keras','numpy','GPU')}})
        require(len(registry)==12,'All twelve fixed classifiers are required')
        return registry
    finally:
        if archive:archive.close()


def find_b_run():
    if CONFIG['b_run']:
        candidates=[Path(CONFIG['b_run']).expanduser()]
    else:
        candidates=[]
        for root in search_roots():
            candidates.extend(p.parent for p in root.rglob('manifest.json') if p.parent.name=='KLTN_HyperGAMO_B_'+RUN_ID)
    valid=[]
    for p in sorted(set(candidates),key=str):
        if not (p/'manifest.json').is_file() or not (p/'protocol_spec.json').is_file():continue
        if sha_file(p/'manifest.json')==MANIFEST_SHA and sha_file(p/'protocol_spec.json')==PROTOCOL_SHA:valid.append(p)
    require(valid,'Missing original Protocol B cache. Add the same B dataset used for M4, or set B_RUN to its local KLTN_HyperGAMO_B_50e988fae260 folder. Do not use the dip_reproduction folder.')
    # Prefer an identical B copy which already has the saved locked-test cache.
    root=sorted(valid,key=lambda p:(not (p/'final_data/test_4721_LOCKED.npz').is_file(),str(p)))[0]
    manifest=read_json(root/'manifest.json');spec=read_json(root/'protocol_spec.json')
    require(manifest['run_id']==RUN_ID and manifest['protocol_spec_sha256']==PROTOCOL_SHA,'B manifest chain differs')
    require(spec['implementation_sha256']==IMPLEMENTATION and spec['patch_size']==11 and spec['pca_components']==40,'Wrong B scientific contract')
    require(hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]==RUN_ID,'Wrong B run ID')
    master=root/'B_master_split.npz'
    require(master.is_file() and sha_file(master)==manifest['master_split_sha256'],'Master split differs from original B manifest')
    prep=root/'final_data/preprocessing_final.joblib'
    require(prep.is_file() and sha_file(prep)==PREP_SHA,'Original final preprocessing missing/different from M4 training')
    side=read_json(str(prep)+'.json')
    require(side['sha256']==PREP_SHA and side['contract']['run_id']==RUN_ID and side['contract']['implementation']==IMPLEMENTATION,'Final preprocessing sidecar differs')
    REPORT['data_provenance']={'B_run':str(root),'B_manifest_sha256':MANIFEST_SHA,
        'B_protocol_spec_sha256':PROTOCOL_SHA,'B_master_split_sha256':manifest['master_split_sha256'],
        'preprocessing_final_sha256':PREP_SHA}
    return root


def locked_test_geometry(root):
    # Called only AFTER writing the immutable twelve-model evaluation registry.
    REPORT['test_opened']=True
    with np.load(root/'B_master_split.npz',allow_pickle=False) as d:
        split=d['split'];gt=d['standard_gt'];train=d['coords_train'];coords=d['coords_test']
    require(split.shape==gt.shape==(145,145),'Wrong B geometry')
    require(train.shape==(5528,2) and coords.shape==(4721,2),'Wrong B train/test counts')
    require(train.dtype==coords.dtype==np.int16,'B coordinate dtype differs')
    require(np.array_equal(train,np.argwhere(split==0)) and np.array_equal(coords,np.argwhere(split==2)),'Master center order/partition differs')
    require(np.all((coords>=0)&(coords<145)) and len(np.unique(coords,axis=0))==4721,'Invalid test centers')
    y=gt[coords[:,0],coords[:,1]].astype(np.int64)-1
    y_train=gt[train[:,0],train[:,1]].astype(np.int64)-1
    require(np.all((y>=0)&(y<16)) and np.bincount(y,minlength=16).tolist()==TEST_COUNTS,'Wrong B test classes')
    require(np.bincount(y_train,minlength=16).tolist()==TRAIN_COUNTS,'Wrong B train classes')
    REPORT['data_provenance'].update(test_coords_B_sha256=b_array_sha(coords),test_samples=4721,
                                    per_class_support=TEST_COUNTS,train_samples=5528)
    return coords,y


def load_test(root,coords,y):
    path=root/'final_data/test_4721_LOCKED.npz'
    expected={'run_id':RUN_ID,'implementation':IMPLEMENTATION,'kind':'locked_test',
              'coords':b_array_sha(coords),'preprocessing_sha256':PREP_SHA}
    if path.is_file():
        side=read_json(str(path)+'.json')
        require(side['contract']==expected and sha_file(path)==side['sha256'],'Locked test cache/sidecar contract differs')
        with np.load(path,allow_pickle=False) as d:
            x=d['X_test'];yy=d['y_test'];cc=d['coords_test']
        require(np.array_equal(yy,y) and yy.dtype==np.int64 and np.array_equal(cc,coords) and cc.dtype==coords.dtype,'Cached test targets/center order differ')
        REPORT['data_provenance'].update(test_source='original_B_locked_test_cache',test_cache_sha256=side['sha256'],preprocessing_deserialized=False)
    else:
        phase('No saved test cache: transform with the EXACT saved final PCA/scaler; no fitting')
        if CONFIG['hsi_path']:candidates=[Path(CONFIG['hsi_path']).expanduser()]
        else:candidates=[p for r in search_roots() for p in r.rglob('*.mat') if p.name.lower()=='indian_pines_corrected.mat']
        matches=[p for p in sorted(set(candidates),key=str) if p.is_file() and sha_file(p)==HSI_SHA]
        require(matches,'B has no locked-test cache. Add the original Indian Pines corrected .mat (or set HSI_PATH). No PCA refit or replacement data was used.')
        from scipy.io import loadmat
        import joblib
        # Hash was pinned to the exact artifact used by all twelve training jobs.
        prep=joblib.load(root/'final_data/preprocessing_final.joblib')
        pca,scaler=prep['pca'],prep['scaler']
        require(pca.n_samples_==5528 and pca.n_components_==40,'Wrong fitted PCA')
        cube=loadmat(matches[0])['indian_pines_corrected'].astype(np.float32)
        require(cube.shape==(145,145,200) and np.isfinite(cube).all(),'Wrong HSI cube')
        flat=cube.reshape(-1,200).astype(np.float64)
        processed=scaler.transform(pca.transform(flat)).reshape(145,145,40).astype(np.float32)
        padded=np.pad(processed,((5,5),(5,5),(0,0)),mode='constant')
        x=np.empty((4721,*SHAPE),np.float32)
        for i,(row,col) in enumerate(coords):
            deadline();x[i,...,0]=padded[row:row+11,col:col+11]
        REPORT['data_provenance'].update(test_source='exact_saved_final_transform',HSI_sha256=HSI_SHA,
                                        preprocessing_deserialized=True,fit_or_fit_transform_called=False,
                                        patch_padding='constant zero AFTER PCA/scaler transform')
    require(x.dtype==np.float32 and x.shape==(4721,*SHAPE) and np.isfinite(x).all(),'Invalid test patches')
    REPORT['data_provenance']['X_test_B_sha256']=b_array_sha(x)
    return x


def runtime():
    global keras
    import tensorflow as tf
    import tf_keras as keras
    require(tf.__version__.split('.')[:2]==keras.__version__.split('.')[:2],'TensorFlow/tf-keras minor mismatch; no auto installation/retry')
    physical=tf.config.list_physical_devices('GPU')
    use_gpu=bool(physical) and CONFIG['device']!='cpu'
    require(CONFIG['device']!='gpu' or use_gpu,'GPU requested but unavailable; choose auto/cpu deliberately')
    if use_gpu:
        tf.config.set_visible_devices(physical[:1],'GPU')
        tf.config.experimental.set_memory_growth(physical[0],True)
    else:tf.config.set_visible_devices([],'GPU')
    keras.mixed_precision.set_global_policy('float32')
    device='/GPU:0' if use_gpu else '/CPU:0'
    with tf.device(device):probe=tf.linalg.matmul(tf.ones((2,2)),tf.ones((2,2)))
    require(device[1:] in probe.device,f'Unexpected probe execution device: {probe.device}')
    versions={name:metadata.version(name) for name in ('numpy','scipy','scikit-learn','joblib')}
    REPORT['runtime']={'tensorflow':tf.__version__,'tf_keras':keras.__version__,**versions,
        'device':device,'GPU_details':tf.config.experimental.get_device_details(physical[0]) if use_gpu else None,
        'inference_only':True,'optimizer_created':False,'G_or_D_constructed':False}
    print('TEST DEVICE:',device,'| classifiers only; no optimizer/G/D',flush=True)
    return tf,device


def calculate_metrics(y,p):
    require(p.shape==(len(y),16) and np.isfinite(p).all(),'Invalid classifier output')
    require(np.all(p>=-1e-7) and np.all(p<=1+1e-7) and np.allclose(p.sum(1),1,atol=1e-4,rtol=1e-5),'Output is not a probability distribution')
    pred=np.argmax(p,axis=1).astype(np.int64)
    cm=np.bincount(16*y+pred,minlength=256).reshape(16,16)
    support=cm.sum(1);predicted=cm.sum(0);tp=np.diag(cm)
    recall=np.divide(tp,support,out=np.zeros(16,float),where=support>0)
    precision=np.divide(tp,predicted,out=np.zeros(16,float),where=predicted>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(16,float),where=(support+predicted)>0)
    oa=float(tp.sum()/len(y));expected=float((support.astype(float)*predicted).sum()/len(y)**2)
    result={'OA':oa,'AA':float(recall.mean()),'Macro_F1':float(f1.mean()),
            'GM':float(np.exp(np.log(recall).mean())) if np.all(recall>0) else 0.,
            'Kappa':float((oa-expected)/(1-expected)) if expected<1 else 1.}
    rows=[{'class_id':i+1,'class_name':CLASS_NAMES[i],'support':int(support[i]),
           'precision':float(precision[i]),'recall':float(recall[i]),'F1':float(f1[i])} for i in range(16)]
    return result,rows,pred,cm


def save_csv(path,rows):
    require(bool(rows),'No rows to save')
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(rows,per_class):
    summary=[];class_summary=[];paired=[]
    for arm in ARMS:
        subset=[r for r in rows if r['arm']==arm]
        require(sorted(r['seed'] for r in subset)==list(SEEDS),f'Incomplete test seeds: {arm}')
        row={'arm':arm,'method':LABELS[arm],'seeds':3}
        for metric in METRICS:
            values=np.asarray([r[metric] for r in subset])
            row[metric+'_mean']=float(values.mean());row[metric+'_sd']=float(values.std(ddof=1))
        summary.append(row)
        for cid in range(1,17):
            subset=[r for r in per_class if r['arm']==arm and r['class_id']==cid]
            row={'arm':arm,'class_id':cid,'class_name':CLASS_NAMES[cid-1],'support':TEST_COUNTS[cid-1]}
            for metric in ('precision','recall','F1'):
                vals=np.asarray([r[metric] for r in subset]);row[metric+'_mean']=float(vals.mean());row[metric+'_sd']=float(vals.std(ddof=1))
            class_summary.append(row)
    for baseline in ARMS[:-1]:
        for seed in SEEDS:
            a=next(r for r in rows if r['arm']=='learned_local' and r['seed']==seed)
            b=next(r for r in rows if r['arm']==baseline and r['seed']==seed)
            paired.append({'contrast':'learned_local - '+baseline,'seed':seed,**{k:a[k]-b[k] for k in METRICS}})
    return summary,class_summary,paired


def run():
    phase('Verify ALL twelve pinned classifiers before opening test data')
    registry=obtain_models()
    phase('Verify original B manifest, master split and final preprocessing hashes')
    b_root=find_b_run()
    freeze={'protocol_B_id':RUN_ID,'fixed_C_updates':8000,'models':registry,
        'data_provenance':REPORT['data_provenance'],'test_previously_used_for_B':True,
        'selection_policy':'all twelve submitted fixed-budget final classifiers, no test-based selection',
        'metrics':list(METRICS),'class_ids':list(range(1,17)),
        'test_worker_sha256':sha_file(__file__),'config_sha256':json_sha(CONFIG)}
    save_json(OUT/'evaluation_registry.json',freeze)
    REPORT['evaluation_registry_sha256']=sha_file(OUT/'evaluation_registry.json')
    phase('Load the fixed B test set; no training or fitting')
    coords,y=locked_test_geometry(b_root);x=load_test(b_root,coords,y)
    tf,device=runtime()
    batch=CONFIG['gpu_batch_size'] if device=='/GPU:0' else CONFIG['cpu_batch_size']
    rows=[];all_class=[]
    for index,item in enumerate(registry):
        phase(f'Test {index+1}/12: {item["key"]} | batch {batch}')
        started=time.monotonic();keras.backend.clear_session()
        keras.utils.set_random_seed(item['seed'])
        with tf.device(device):
            model=tao_classifier(SHAPE,16)
            model.load_weights(item['weights_path'])
            probabilities=np.empty((len(y),16),np.float32)
            for first in range(0,len(y),batch):
                deadline();last=min(first+batch,len(y))
                probabilities[first:last]=model(x[first:last],training=False).numpy()
                if first==0 or last==len(y) or (first//batch)%32==0:
                    print(f'  {item["key"]}: {last}/{len(y)} samples; {time.monotonic()-started:.1f}s',flush=True)
        order=np.asarray(item['class_order'])-1
        original=np.empty_like(probabilities);original[:,order]=probabilities
        metrics,classes,pred,cm=calculate_metrics(y,original)
        folder=OUT/'test_results'/item['arm']/f'seed_{item["seed"]}';folder.mkdir(parents=True,exist_ok=True)
        save_json(folder/'metrics.json',{'arm':item['arm'],'seed':item['seed'],'test_samples':len(y),
                                      'classifier_sha256':item['classifier_sha256'],**metrics})
        save_csv(folder/'per_class.csv',classes)
        np.savez_compressed(folder/'predictions.npz',coords=coords,y_true=y,y_pred=pred,
                            probabilities=original,confusion_matrix=cm)
        save_json(folder/'completed.json',{'arm':item['arm'],'seed':item['seed'],'status':'COMPLETE',
            'model_sha256':item['classifier_sha256'],'files':{name:sha_file(folder/name) for name in ('metrics.json','per_class.csv','predictions.npz')}})
        rows.append({'arm':item['arm'],'method':LABELS[item['arm']],'seed':item['seed'],
                     **metrics,'test_samples':len(y),'inference_seconds':round(time.monotonic()-started,3)})
        all_class.extend({'arm':item['arm'],'seed':item['seed'],**r} for r in classes)
        REPORT['completed_models']=[{'arm':r['arm'],'seed':r['seed']} for r in rows]
        save_csv(OUT/'test_by_seed.csv',rows)
        save_csv(OUT/'test_per_class_by_seed.csv',all_class)
        save_json(OUT/'manifest.json',REPORT)
        print(' ',item['key'],metrics,flush=True)
        del model,probabilities,original;gc.collect()
    summary,class_summary,paired=summarize(rows,all_class)
    save_csv(OUT/'test_summary_mean_sd.csv',summary)
    save_csv(OUT/'test_per_class_mean_sd.csv',class_summary)
    save_csv(OUT/'paired_differences.csv',paired)
    lines=['# M4 — Protocol B test, four arms × three seeds','',
           'Mean ± sample SD (ddof=1), three seeds. OA/AA/Macro-F1/GM in %, Kappa 0–1.',
           'This is the same within-scene B test previously evaluated for older B models; no leakage-free or new blind-test claim.','',
           '| Method | OA (%) | AA (%) | Macro-F1 (%) | GM (%) | Kappa |',
           '|---|---:|---:|---:|---:|---:|']
    for row in summary:
        values=[f'{100*row[k+"_mean"]:.3f} ± {100*row[k+"_sd"]:.3f}' if k!='Kappa' else f'{row[k+"_mean"]:.4f} ± {row[k+"_sd"]:.4f}' for k in METRICS]
        lines.append('| '+LABELS[row['arm']]+' | '+' | '.join(values)+' |')
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    REPORT.update(status='COMPLETE',complete=True,elapsed_seconds=round(time.monotonic()-START,2),
                  training_run=False,optimizer_created=False,G_or_D_constructed=False,
                  completed_model_count=12,per_class_count=16,metric_sd_ddof=1)
    save_json(OUT/'manifest.json',REPORT)
    print('\n'+'\n'.join(lines),flush=True)


def main():
    global CONFIG,REPORT,OUT,START
    CONFIG=read_json(sys.argv[1]);OUT=Path(CONFIG['output_dir']);START=CONFIG['start_monotonic']
    REPORT={'status':'RUNNING','complete':False,'scope':'M4 fixed classifier inference-only test',
            'test_opened':False,'training_run':False,'completed_models':[],
            'protocol_B_id':RUN_ID,'test_previously_used_for_B':True,
            'test_based_hyperparameter_selection':False,'spatial_leakage_eliminated_claim':False}
    try:run()
    except BaseException as error:
        REPORT.update(status='TIMEOUT' if isinstance(error,TimeoutError) else 'FAILED',complete=False,
                      error=str(error),traceback=traceback.format_exc(),elapsed_seconds=round(time.monotonic()-START,2))
        save_json(OUT/'manifest.json',REPORT);traceback.print_exc();sys.exit(1)


if __name__=='__main__':main()
