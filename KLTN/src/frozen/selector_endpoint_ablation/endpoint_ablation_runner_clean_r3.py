
"""Phase 6 endpoint-weight ablation for the frozen M4 Indian Pines Protocol B.

Standalone Kaggle worker emitted by selector.ipynb. NEVER reads test during training.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, importlib.util, json, os, shutil, sys, time, traceback
from pathlib import Path
import numpy as np

TRAIN_UPDATES = 8000
C_LOOPS = TRAIN_UPDATES // 2
WORKER_SHA = '6d33ca6a0d39600f5f44c0d7e63547b1237149d625471ac73315d982ffbd8fb5'
RUN_ID = '50e988fae260'
IMPLEMENTATION = '5e15f8214bf273bb889187b96c80407a8000e5e1a3ef30734bcfa86ced275a6f'
TEST_COUNTS = [25,675,404,99,274,354,12,250,10,503,1065,282,80,545,99,44]
TRAIN_COUNTS = [21,753,426,138,209,376,16,228,10,469,1390,311,125,720,287,49]
PREP_SHA = '18167b01cbb8ac5d21ff73a3588cad758d23a2629f9b1c9e47dbaf4cf6989939'
HSI_SHA = 'ec2f8808710919d566f70f0d4aa885aae1ddfd42b734aba71c5e12ca65450939'
METHODS = ('learned_weights', 'random_one_fixed')
SEEDS = (42, 43, 44)


def fail(cond, reason):
    if not cond: raise RuntimeError(reason)


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def js(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def load_json(p):return json.loads(Path(p).read_text())


def import_frozen(path):
    fail(Path(path).is_file(), f'M4 worker not found: {path}')
    fail(sha_file(path)==WORKER_SHA,'Frozen worker SHA mismatch; stop')
    spec=importlib.util.spec_from_file_location('frozen_m4_worker_ablation',str(path))
    w=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=w
    spec.loader.exec_module(w)
    return w


def find_checkpoint(seed, config):
    root=Path(config['checkpoints'][str(seed)])
    if (root/'state'/'ckpt_08000'/'meta.json').is_file():root=root/'state'/'ckpt_08000'
    if (root/'job'/'state'/'ckpt_08000'/'meta.json').is_file():root=root/'job'/'state'/'ckpt_08000'
    fail((root/'meta.json').is_file(),f'Missing native checkpoint meta for G seed {seed}: {root}')
    meta=load_json(root/'meta.json')
    c=meta['contract']
    for key,expected in [('arm','learned_local'),('stage','final'),('seed',seed),('budget_C_updates',8000),('protocol_B_id',RUN_ID),('sha_worker',WORKER_SHA),('k',3),('rho',0.2)]:
        fail(c.get(key)==expected,f'Checkpoint seed {seed} wrong contract {key}: {c.get(key)}')
    fail(meta['c_updates']==8000 and meta['optimizer_clock']==136000,'Not endpoint 8000 / 136000 Adam clock')
    for fname,expect in meta['weights_hashes'].items():
        fail((root/fname).is_file() and sha_file(root/fname)==expect,f'Native checkpoint file corrupted: {root/fname}')
    fail((root/'weights.index').is_file(),'Native checkpoint index not present')
    return root, meta


def load_train(w, config):
    w.CONFIG={'data_mode':'auto_real_B','stage':'final','b_run':config['b_run'],
              'deadline_monotonic':time.monotonic()+10**6}
    X,y,src=w.get_training_data()
    fail(X.shape==(5528,*w.SHAPE),'Incorrect final train')
    fail(np.bincount(y,minlength=16).tolist()==TRAIN_COUNTS,'Final train counts mismatch')
    info=w.build_graph_per_class(X,y)
    return X,y,src,info


def source_check(w, X, y, prepared, ckmeta):
    con=ckmeta['contract']
    fail(w.sha_array(X)==con['train_X_sha256'], 'Checkpoint trained on DIFFERENT X; stop')
    fail(w.sha_array(y)==con['train_y_sha256'], 'Checkpoint trained on DIFFERENT y; stop')
    fail((prepared['order']+1).tolist()==con['class_order'], 'Class order mismatch')
    fail([w.sha_array(g) for g in prepared['neighbor_graphs']]==con['graph_sha256'], 'Graph hash mismatch')
    fail(con['implementation_B']==IMPLEMENTATION,'Implementation hash mismatch')


def random_map(prepared, seed):
    # A PRIVATE RNG: independent of frame / real / anchor / G-noise / C-dropout.
    maps=[]
    for h,g in enumerate(prepared['neighbor_graphs']):
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261018,seed,h,713])))
        maps.append(rng.integers(0,3,size=len(g),dtype=np.int32))
    return maps


def schedule(w, prepared, seed, X):
    spec=w.make_schedule(prepared['y_sorted'],seed=seed,batch_size=100,latent_dim=100,
                         g_points=7,n_heads=15,class_order=(prepared['order']+1).tolist())
    fail(spec['samples']==len(X),'M4 schedule wrong sample count')
    return spec


def init_models(w, X, prepared, seed, ck_root, ckmeta, mode):
    tf=w.tf; k=w.keras
    k.backend.clear_session()
    k.utils.set_random_seed(seed)
    # M4 classifier AND M4 externally keyed dropout recorder.
    w.PAIRED_C_DROPOUT,ccontrol=w.recorded_dropout_factory(64,'C')
    k.utils.set_random_seed(seed)
    C=w.tao_classifier(w.SHAPE,16)
    context={}
    def observer(pairs):return context['observe'](pairs)
    opt=k.optimizers.legacy.Adam(learning_rate=2e-4,beta_1=.5,gradient_transformers=[observer])
    C.compile(loss='mean_squared_error',optimizer=opt)
    owners={'C':{'trainable':tuple(C.trainable_variables),'nontrainable':tuple(C.non_trainable_variables)}}
    opt._create_all_weights(list(C.trainable_variables))
    audit,obs=w.make_gradient_audit(owners);context['observe']=obs
    m={'C':C,'opt':opt,'audit':audit,'controls':{'C':ccontrol},'owners':owners}
    init={
        'C_trainable_sha256':w.arrays_sha(v.numpy() for v in owners['C']['trainable']),
        'C_nontrainable_sha256':w.arrays_sha(v.numpy() for v in owners['C']['nontrainable']),
        'C_adam_slots_sha256':w.arrays_sha(opt.get_slot(v,s).numpy() for v in owners['C']['trainable'] for s in ('m','v')),
        'Adam_clock':int(opt.iterations.numpy()),
    }
    fail(init['Adam_clock']==0,'Classifier Adam NOT fresh')
    # Restore G and heads from native checkpoint; NEVER use the trained M4 C.
    k.utils.set_random_seed(seed+102)
    G=w.tao_generator(100,16)
    heads=[]
    for h,(support,graph) in enumerate(zip(prepared['supports'],prepared['neighbor_graphs'])):
        k.utils.set_random_seed(seed+1000+h)
        heads.append(w.tao_head_v2(support,graph))
    models={'G':G,**{f'head_{h:02d}':head for h,head in enumerate(heads)}}
    # Snapshot the ORIGINAL semantic variable partitions BEFORE freezing. This
    # exactly matches frozen M4 meta trainable/name/i + nontrainable/name/i.
    semantic={}
    for name,net in [('G',G)]+[(f'head_{h}',head) for h,head in enumerate(heads)]:
        semantic[name]={
            'trainable':list(net.trainable_variables),
            'nontrainable':list(net.non_trainable_variables),
        }
    state=tf.train.Checkpoint(**models).read(str(ck_root/'weights'))
    state.assert_existing_objects_matched()
    state.expect_partial() # The unrequested native C/D/optimizer are intentionally NOT restored.
    for name,groups in semantic.items():
        for part,variables in groups.items():
            for i,v in enumerate(variables):
                key=f'{part}/{name}/{i}'
                fail(key in ckmeta['state_hashes'] and w.sha_array(v.numpy())==ckmeta['state_hashes'][key],
                     f'Native G/head variable mismatch: {key}')
    for net in models.values():net.trainable=False
    fail(all(len(net.trainable_variables)==0 for net in models.values()),'G/head are not frozen')
    frozen_vars={name:w.arrays_sha(v.numpy() for v in net.weights) for name,net in models.items()}
    # Assert no optimizer has G/head variables/slots.
    crefs={v.ref() for v in C.weights}
    fail(all(v.ref() not in crefs for net in models.values() for v in net.weights),'C and G/head share variable!')
    return m, G, heads, init, frozen_vars


def render(w, item, prepared, method, G, heads, maps):
    labels=item['fake_onehot']
    cls=item['fake_classes'];anc=item['local_anchors']
    fake=np.empty((len(cls),*w.SHAPE),dtype=np.float32)
    if method=='learned_weights':
        features=G([item['noise_c'],labels],training=False).numpy()
    for h,support in enumerate(prepared['supports']):
        ids=np.flatnonzero(cls==h)
        if not len(ids):continue
        a=anc[ids]
        if method=='learned_weights':
            fake[ids]=heads[h]([features[ids],a[:,None]],training=False).numpy()
        else:
            chosen=prepared['neighbor_graphs'][h][a,maps[h][a]]
            fake[ids]=0.8*support[a]+0.2*support[chosen]
    fail(np.isfinite(fake).all(),'Invalid fake patch')
    return fake


def state_snap(w, m):
    C=m['C'];opt=m['opt'];audit=m['audit'];control=m['controls']['C']
    return {'C_weights':w.arrays_sha(v.numpy() for v in C.weights),
            'C_slots':w.arrays_sha(opt.get_slot(v,s).numpy() for v in C.trainable_variables for s in ('m','v')),
            'clock':int(opt.iterations.numpy()),'audit_calls':int(audit['calls'].numpy()),
            'dropout_calls':int(control['calls'].numpy())}


def ckpt_store(path,w,m,contract,next_loop,trace,paired_trace,loss_tail):
    path=Path(path);base=path/'state';base.mkdir(parents=True,exist_ok=True)
    fail(int(m['opt'].iterations.numpy())==next_loop*34,'Clock wrong at checkpoint')
    key=f'ckpt_{next_loop*2:05d}'
    dest=base/key;tmp=base/(key+'.incomplete')
    if tmp.exists():shutil.rmtree(tmp)
    fail(not dest.exists(),f'Checkpoint exists unexpectedly: {dest}')
    tmp.mkdir()
    control=m['controls']['C']
    chk=w.tf.train.Checkpoint(C=m['C'],optimizer=m['opt'],
        audit=w.tf.train.Checkpoint(**m['audit']),
        dropout_controls=w.tf.train.Checkpoint(C=w.tf.train.Checkpoint(**control)))
    chk.write(str(tmp/'weights'))
    snap={'schema':1,'contract':contract,'next_loop':next_loop,
          'c_updates':next_loop*2,'clock':next_loop*34,'trace':trace,'paired_trace':paired_trace,
          'last_loss':loss_tail,'state':state_snap(w,m),
          'weights_hashes':{p.name:sha_file(p) for p in tmp.glob('weights*')}}
    write_json(tmp/'meta.json',snap)
    tmp.rename(dest)
    write_json(base/'latest.json',{'name':key,'c_updates':next_loop*2,'meta_sha256':sha_file(dest/'meta.json')})
    old=sorted(p for p in base.glob('ckpt_*') if p.is_dir())
    for p in old[:-2]:shutil.rmtree(p)
    print('CHECKPOINT',next_loop*2,'/',TRAIN_UPDATES,flush=True)


def ckpt_load(path,w,m,contract):
    base=Path(path)/'state';ptr=base/'latest.json'
    if not ptr.is_file():return 0,hashlib.sha256(b'P6_TRACE_V1').hexdigest(),hashlib.sha256(b'P6_PAIR_V1').hexdigest(),[]
    pointer=load_json(ptr);folder=base/pointer['name'];meta=folder/'meta.json'
    fail(meta.is_file() and sha_file(meta)==pointer['meta_sha256'],'Corrupt checkpoint metadata')
    record=load_json(meta)
    fail(record['contract']==contract,'Different contract on resume: refusing')
    fail(record['c_updates']==pointer['c_updates'],'Corrupt checkpoint pointer updates')
    for fname,h in record['weights_hashes'].items():fail(sha_file(folder/fname)==h,'Corrupt checkpoint weights')
    chk=w.tf.train.Checkpoint(C=m['C'],optimizer=m['opt'],
        audit=w.tf.train.Checkpoint(**m['audit']),
        dropout_controls=w.tf.train.Checkpoint(C=w.tf.train.Checkpoint(**m['controls']['C'])))
    chk.read(str(folder/'weights')).assert_consumed()
    fail(state_snap(w,m)==record['state'],'Restored classifier/Adam/dropout state mismatch')
    fail(record['clock']==34*record['next_loop'],'Restored Adam clock mismatch')
    print('RESUMED',record['c_updates'],'/',TRAIN_UPDATES,flush=True)
    return record['next_loop'],record['trace'],record['paired_trace'],record['last_loss']


def run_train(cfg,job,seed,method):
    start=time.monotonic();job=Path(job)
    fail(method in METHODS and seed in SEEDS,'Wrong job parameters')
    if (job/'completed.json').is_file():
        record=load_json(job/'completed.json')
        fail(record['status']=='COMPLETE' and record['c_updates']==8000,'Malformed existing complete marker')
        print('ALREADY COMPLETE:',seed,method);return
    job.mkdir(parents=True,exist_ok=True)
    w=import_frozen(cfg['worker_path']);w.REPORT={'route_counts':{}}
    X,y,src,prepared=load_train(w,cfg)
    ck_root,ckmeta=find_checkpoint(seed,cfg);source_check(w,X,y,prepared,ckmeta)
    spec=schedule(w,prepared,seed,X);maps=random_map(prepared,seed)
    map_hashes=[w.sha_array(m) for m in maps]
    runtime=w.load_runtime()
    m,G,heads,init,frozen=init_models(w,X,prepared,seed,ck_root,ckmeta,method)
    # Check matched classifier/Adam init across two arms before first optimizer call.
    pair_file=Path(cfg['base'])/f'init_seed_{seed}.json'
    if pair_file.is_file():
        old=load_json(pair_file)
        fail(old==init, 'Paired C initialization/Adam mismatch: STOP')
    else:write_json(pair_file,init)
    require_spec_hash=w.json_sha(spec)
    contract={'runner_sha':sha_file(__file__),'m4_worker_sha':WORKER_SHA,'source_checkpoint_meta_sha':sha_file(ck_root/'meta.json'),
          'source_native_hashes':ckmeta['weights_hashes'],
          'stage':'final','G_seed':seed,'C_seed':seed,'method':method,'train_X_sha':w.sha_array(X),'train_y_sha':w.sha_array(y),
          'schedule_sha':require_spec_hash,'graph_sha':[w.sha_array(g) for g in prepared['neighbor_graphs']],
          'C_init':init,'random_map_sha':map_hashes,'frozen_G_head_sha':frozen,
          'tf':runtime['tensorflow'],'tf_keras':runtime['tf_keras'],'numpy':np.__version__,
          'k':3,'rho':.2,'C_updates':8000,'adam_clock_end':136000,
          'C_optimizer':'tf_keras.optimizers.legacy.Adam(lr=0.0002,beta1=0.5)',
          'C_loss':'mean_squared_error','training_test_access':False}
    if (job/'contract.json').is_file():fail(load_json(job/'contract.json')==contract,'Existing job different contract')
    else:write_json(job/'contract.json',contract)
    write_json(job/'startup_audit.json',{'source':src,'init':init,'checkpoint_dir':str(ck_root),'verified_native_8000':True,
        'frozen_G_head':True,'test_opened':False,'rng_isolation':True,'initialization_paired':True})
    loop,trace,paired_trace,loss_tail=ckpt_load(job,w,m,contract)
    w.CONFIG['seed']=seed;w.CONFIG['arm']='random_local' # only for controlled_update seed/route, no arm-specific training here
    w.CONFIG['deadline_monotonic']=time.monotonic()+float(cfg['max_worker_seconds'])+500
    stop_at=time.monotonic()+float(cfg['max_worker_seconds'])
    while loop<C_LOOPS:
        if loop>0 and time.monotonic()+max(60.,float(cfg['checkpoint_safety_seconds']))>=stop_at:
            latest=(job/'state'/'latest.json')
            if (not latest.is_file()) or load_json(latest)['c_updates']!=loop*2:
                ckpt_store(job,w,m,contract,loop,trace,paired_trace,loss_tail)
            write_json(job/'progress.json',{'status':'PAUSED','C_updates':loop*2,'time_seconds':round(time.monotonic()-start,2)})
            print('PAUSED safely; re-run TRAIN to resume',seed,method,flush=True);return
        item=w.frame(spec,loop)
        fail(int(m['opt'].iterations.numpy())==34*loop,f'Adam clock wrong loop {loop}')
        real=X[item['real_ids']].copy();lab=item['real_onehot'].copy()
        loss1,record1=w.controlled_update(m,'C','C_real',loop,-1,
                            lambda:m['C'].train_on_batch(real,lab),{'C'},'C')
        w.skip_iterations(m,1,'skip_D_real')
        fake=render(w,item,prepared,method,G,heads,maps)
        loss2,record2=w.controlled_update(m,'C','C_fake',loop,-1,
                            lambda:m['C'].train_on_batch(fake,item['fake_onehot']),{'C'},'C')
        w.skip_iterations(m,31,'skip_D_fake_GC_GD')
        fail(int(m['opt'].iterations.numpy())==34*(loop+1),'Wrong optimizer step count')
        # Trace reflects actual consumed schedule, dropout masks and synthetic image.
        consumed={'real_ids':w.sha_array(item['real_ids']),
                  'real_onehot':w.sha_array(lab),'fake_classes':w.sha_array(item['fake_classes']),
                  'anchors':w.sha_array(item['local_anchors']),
                  'noise':w.sha_array(item['noise_c']),
                  'real_dropout':record1['dropout_mask_sha256'],
                  'fake_dropout':record2['dropout_mask_sha256'],
                  'fake_image':w.sha_array(fake)}
        trace=hashlib.sha256(bytes.fromhex(trace)+bytes.fromhex(js(consumed))).hexdigest()
        paired_consumed={k:v for k,v in consumed.items() if k!='fake_image'}
        paired_trace=hashlib.sha256(bytes.fromhex(paired_trace)+bytes.fromhex(js(paired_consumed))).hexdigest()
        loss_tail=[float(loss1['loss']),float(loss2['loss'])]
        loop+=1
        if loop%100==0:
            write_json(job/'progress.json',{'status':'TRAINING','C_updates':loop*2,'Adam_clock':loop*34,
                                           'C_real_loss':loss_tail[0],'C_fake_loss':loss_tail[1],'elapsed_seconds':round(time.monotonic()-start,2)})
            print('TRAIN',method,seed,'C_updates',loop*2,'C_real',round(loss_tail[0],6),'C_fake',round(loss_tail[1],6),flush=True)
        if loop%250==0 or loop==C_LOOPS:
            # Native G/head must remain untouched throughout C updates.
            current={name:w.arrays_sha(v.numpy() for v in net.weights) for name,net in {'G':G,**{f'head_{h:02d}':head for h,head in enumerate(heads)}}.items()}
            fail(current==frozen,'FROZEN G/head weights changed during C training')
            ckpt_store(job,w,m,contract,loop,trace,paired_trace,loss_tail)
    snap=state_snap(w,m)
    fail(snap['clock']==136000 and snap['audit_calls']==8000 and snap['dropout_calls']==8000,
         'Final C/Adam/Dropout budget mismatch')
    path=job/'classifier.weights.h5';m['C'].save_weights(str(path))
    write_json(job/'completed.json',{'status':'COMPLETE','c_updates':8000,'adam_clock':136000,'seed':seed,
        'method':method,'classifier_sha':sha_file(path),'contract_sha':sha_file(job/'contract.json'),
        'checkpoint_meta_sha':sha_file(job/'state'/'ckpt_08000'/'meta.json'),
        'frozen_weights_unchanged':True,'paired_consumption_trace':paired_trace,'test_opened':False,'elapsed_seconds':round(time.monotonic()-start,2)})
    print('COMPLETE',seed,method,'C=8000; test NOT opened',flush=True)


def validate_six(cfg):
    records=[]
    for seed in SEEDS:
        for method in METHODS:
            job=Path(cfg['base'])/'jobs'/f'G{seed}_C{seed}'/method
            marker=job/'completed.json'
            fail(marker.is_file(),f'Test LOCKED: missing completed job {seed}/{method}')
            completed=load_json(marker)
            fail(completed['status']=='COMPLETE' and completed['c_updates']==8000 and completed['adam_clock']==136000,
                 f'Test LOCKED: incomplete job {seed}/{method}')
            fail(sha_file(job/'classifier.weights.h5')==completed['classifier_sha'],'Saved C classifier hash differs')
            fail(sha_file(job/'contract.json')==completed['contract_sha'],'Job contract hash differs')
            fail(sha_file(job/'state'/'ckpt_08000'/'meta.json')==completed['checkpoint_meta_sha'],'Checkpoint final hash differs')
            co=load_json(job/'contract.json')
            fail(co['G_seed']==seed and co['C_seed']==seed and co['method']==method and co['k']==3 and co['rho']==.2,
                 'Job contract wrong')
            records.append((seed,method,job,co))
        a=records[-2][3];b=records[-1][3]
        for key in ('C_init','schedule_sha','train_X_sha','train_y_sha','graph_sha','random_map_sha','source_checkpoint_meta_sha'):
            fail(a[key]==b[key],f'Pair {seed} mismatch: {key}')
        left=load_json(records[-2][2]/'completed.json')
        right=load_json(records[-1][2]/'completed.json')
        fail(left['paired_consumption_trace']==right['paired_consumption_trace'],
             f'Pair {seed} consumed DIFFERENT real/anchor/noise/dropout inputs')
    return records


def b_array_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes());return h.hexdigest()


def get_locked_test(broot,cfg):
    # This function is called ONLY in --mode finalize, AFTER all 6 jobs are verified.
    root=Path(broot)
    manifest=load_json(root/'manifest.json')
    master=root/'B_master_split.npz'
    fail(master.is_file() and sha_file(master)==manifest['master_split_sha256'],'Original B master split missing/changed')
    with np.load(master,allow_pickle=False) as data:
        split=data['split'];gt=data['standard_gt'];train_coords=data['coords_train'];coords=data['coords_test']
    fail(split.shape==gt.shape==(145,145) and len(coords)==4721 and len(train_coords)==5528,'Master test partition invalid')
    fail(np.array_equal(coords,np.argwhere(split==2)) and np.array_equal(train_coords,np.argwhere(split==0)), 'B split differs')
    y=gt[coords[:,0],coords[:,1]].astype(np.int64)-1
    fail(np.bincount(y,minlength=16).tolist()==TEST_COUNTS,'Test support counts mismatch')
    path=root/'final_data'/'test_4721_LOCKED.npz'
    if path.is_file():
        side=load_json(str(path)+'.json')
        expect={'run_id':RUN_ID,'implementation':IMPLEMENTATION,'kind':'locked_test',
                'coords':b_array_sha(coords),'preprocessing_sha256':PREP_SHA}
        fail(side['contract']==expect and sha_file(path)==side['sha256'],'Locked test cache corrupted/changed')
        with np.load(path,allow_pickle=False) as z:
            X=z['X_test'];yy=z['y_test'];cc=z['coords_test']
        fail(np.array_equal(yy,y) and np.array_equal(cc,coords),'Locked test labels/coords mismatch')
        src={'source':'B_locked_test_cache','sha256':sha_file(path)}
    else:
        hsi=cfg.get('indian_pines_mat')
        if hsi is None:
            candidates=list(Path('/kaggle/input').rglob('Indian_pines_corrected.mat'))
            candidates+=list(Path('/kaggle/input').rglob('indian_pines_corrected.mat'))
            matches=[p for p in candidates if sha_file(p)==HSI_SHA]
            fail(len(matches)>=1,'Locked test cache missing; attach exact Indian_pines_corrected.mat (or provide indian_pines_mat). NO PCA refit.')
            hsi=matches[0]
        hsi=Path(hsi)
        fail(sha_file(hsi)==HSI_SHA,'Indian Pines raw .mat SHA mismatch')
        prep=root/'final_data'/'preprocessing_final.joblib'
        fail(prep.is_file() and sha_file(prep)==PREP_SHA,'Saved original final PCA/scaler missing')
        import joblib
        from scipy.io import loadmat
        model=joblib.load(prep)  # Original trusted B input, hash checked.
        pca,scaler=model['pca'],model['scaler']
        fail(pca.n_samples_==5528 and pca.n_components_==40,'PCA fit/source mismatch')
        cube=loadmat(hsi)['indian_pines_corrected'].astype(np.float32)
        fail(cube.shape==(145,145,200),'HSI shape mismatch')
        processed=scaler.transform(pca.transform(cube.reshape(-1,200).astype(np.float64))).reshape(145,145,40).astype(np.float32)
        padded=np.pad(processed,((5,5),(5,5),(0,0)),mode='constant')
        X=np.empty((4721,11,11,40,1),dtype=np.float32)
        for i,(row,col) in enumerate(coords):X[i,...,0]=padded[row:row+11,col:col+11]
        src={'source':'original_B_final_preprocessing_without_refit','raw_sha256':sha_file(hsi),'prep_sha256':PREP_SHA}
    fail(X.dtype==np.float32 and X.shape==(4721,11,11,40,1) and np.isfinite(X).all(),'Invalid locked test features')
    src['x_B_sha256']=b_array_sha(X)
    return X,y,coords,src


def metrics(y,prob):
    fail(prob.shape==(len(y),16) and np.isfinite(prob).all(),'Bad predictions')
    from sklearn.metrics import accuracy_score,precision_recall_fscore_support,cohen_kappa_score
    pred=prob.argmax(axis=1)
    p,r,f1,s=precision_recall_fscore_support(y,pred,labels=np.arange(16),zero_division=0)
    return {'OA':float(accuracy_score(y,pred)),'AA':float(r.mean()),'Macro_F1':float(f1.mean()),
            'Kappa':float(cohen_kappa_score(y,pred)),
            'NLL':float(-np.log(np.maximum(prob[np.arange(len(y)),y],1e-12)).mean()),
            'Brier':float(np.square(prob-np.eye(16,dtype=np.float32)[y]).sum(axis=1).mean())},\
           [{'class_id':int(i+1),'support':int(s[i]),'precision':float(p[i]),'recall':float(r[i]),'F1':float(f1[i])} for i in range(16)],pred


def finalize(cfg):
    records=validate_six(cfg)   # NO test files opened prior to this successful gate.
    flag=Path(cfg['base'])/'final_test_gate.json'
    write_json(flag,{'all_six_complete':True,'test_opened_after_gate':True,'contracts':[sha_file(job/'contract.json') for _,_,job,_ in records]})
    X,y,coords,test_src=get_locked_test(cfg['b_run'],cfg)
    w=import_frozen(cfg['worker_path'])
    import tensorflow as tf
    import tf_keras as keras
    fail(tf.__version__.split('.')[:2]==keras.__version__.split('.')[:2], 'TF/keras minor mismatch')
    # Classifier inference only: no GAN and no optimizer.
    keras.mixed_precision.set_global_policy('float32')
    resultdir=Path(cfg['base'])/'final_evaluation';resultdir.mkdir(parents=True,exist_ok=True)
    rows=[];byclass=[];predictions=[]
    for seed,method,job,contract in records:
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        # Frozen M4 C architecture with ordinary Dropout (same weight-bearing layers).
        w.keras=keras; w.tf=tf
        w.PAIRED_C_DROPOUT,_eval_control=w.recorded_dropout_factory(64,'C')
        C=w.tao_classifier((11,11,40,1),16)
        C.load_weights(str(job/'classifier.weights.h5'))
        sorted_prob=C.predict(X,batch_size=48,verbose=0)
        orig=np.empty_like(sorted_prob)
        # M4 C output uses sorted class ranks. Map back to original labels 0..15.
        ck_root,ckmeta=find_checkpoint(seed,cfg)
        order=np.asarray(ckmeta['contract']['class_order'],dtype=np.int64)-1
        orig[:,order]=sorted_prob
        result,pc,pred=metrics(y,orig)
        record={'G_seed':seed,'C_seed':seed,'method':method,**result}
        rows.append(record)
        byclass.extend([{'G_seed':seed,'C_seed':seed,'method':method,**v} for v in pc])
        path=resultdir/f'G{seed}_{method}_predictions.npz'
        np.savez_compressed(path,y_true=y,y_pred=pred,probabilities=orig,coords=coords)
        predictions.append({'seed':seed,'method':method,'sha256':sha_file(path)})
        print('TEST',record,flush=True)
        del C;gc.collect()
    def csvwrite(path,data):
        with Path(path).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    csvwrite(resultdir/'results.csv',rows)
    csvwrite(resultdir/'per_class.csv',byclass)
    pairs=[]
    for seed in SEEDS:
        l=next(x for x in rows if x['G_seed']==seed and x['method']=='learned_weights')
        r=next(x for x in rows if x['G_seed']==seed and x['method']=='random_one_fixed')
        pairs.append({'G_seed':seed,'C_seed':seed,**{f'learned_minus_random_{key}':l[key]-r[key] for key in ('Macro_F1','OA','AA','Kappa','NLL','Brier')}})
    csvwrite(resultdir/'paired_differences.csv',pairs)
    write_json(resultdir/'manifest.json',{'status':'COMPLETE_6_OF_6','protocol_B_id':RUN_ID,
        'retrospective':True,'test_already_seen_in_M4':True,'spatial_overlap_limit':True,
        'tuning_performed':False,'test_source':test_src,'jobs':[{'seed':s,'method':m,'C_sha':sha_file(j/'classifier.weights.h5')} for s,m,j,_ in records],
        'predictions':predictions,'pairs':pairs})
    print('FINAL COMPLETE: all three paired differences recorded',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--mode',choices=['train','finalize'],required=True)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--method',choices=METHODS)
    opts=parser.parse_args()
    cfg=load_json(opts.config)
    fail(cfg['protocol_B_id']==RUN_ID,'Wrong protocol')
    if opts.mode=='train':
        fail(opts.seed in SEEDS and opts.method in METHODS,'Must specify seed/method')
        run_train(cfg,Path(cfg['base'])/'jobs'/f'G{opts.seed}_C{opts.seed}'/opts.method,opts.seed,opts.method)
    else:finalize(cfg)

if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();sys.exit(1)

