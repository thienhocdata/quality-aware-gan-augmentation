# CELL 4 — Chỉ đọc test khi chủ động bật cờ và đủ 12 job đã hoàn tất.
def evaluate_all_final():
    from pathlib import Path
    import csv,hashlib,json,zipfile,shutil
    import numpy as np
    from sklearn.metrics import accuracy_score,precision_recall_fscore_support,cohen_kappa_score
    import pandas as pd
    if ACTION!='evaluate_final' or not ALLOW_REUSED_B_TEST:
        raise RuntimeError('Test đang khóa; ACTION=evaluate_final và ALLOW_REUSED_B_TEST=True mới được mở')
    # All classifier artifacts must exist BEFORE reading the test cache.
    expected={(a,s) for a in ('original_gan','matched_ros','random_local','learned_local') for s in (42,43,44)}
    candidates={}
    # Support attached M4 ZIPs directly: extract only the three classifier metadata
    # files, NOT the large GAN/Adam checkpoints. No re-upload of Protocol B.
    zip_extracted=Path('/kaggle/working/M4_IP_FINAL_ZIP_IMPORTED')
    if zip_extracted.exists():shutil.rmtree(zip_extracted)
    for i,zpath in enumerate(sorted(Path('/kaggle/input').rglob('M4_IP_final_*_REPORT.zip'))):
        destination=zip_extracted/f'zip_{i:02d}'
        destination.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(zpath) as zp:
            for name in ('completed.json','classifier.weights.h5','contract.json'):
                member='job/'+name
                if member not in zp.namelist():continue
                assert zp.getinfo(member).file_size < 100*1024*1024, 'Suspicious classifier file size'
                with zp.open(member) as f, (destination/name).open('wb') as out:
                    shutil.copyfileobj(f,out)
    for base in (Path('/kaggle/working/M4_IP_jobs'),Path('/kaggle/input'),zip_extracted):
        if not base.exists():continue
        for f in base.rglob('completed.json'):
            try:
                d=json.loads(f.read_text());key=(d['arm'],int(d['seed']))
                if d.get('stage')!='final' or key not in expected:continue
                p=f.parent/'classifier.weights.h5';contract=f.parent/'contract.json'
                if not d.get('complete') or d.get('c_updates')!=8000 or not p.is_file() or not contract.is_file():continue
                def sha(path):
                    h=hashlib.sha256()
                    with path.open('rb') as r:
                        for b in iter(lambda:r.read(1024*1024),b''):h.update(b)
                    return h.hexdigest()
                if sha(p)!=d['model_sha256'] or sha(contract)!=d['job_contract_sha256']:continue
                if d.get('worker_sha256')!=WORKER_EXPECTED_SHA256:continue
                if key in candidates:
                    if candidates[key][0]['model_sha256']!=d['model_sha256']:
                        raise RuntimeError('Nhiều mô hình không giống nhau cho '+str(key))
                else:candidates[key]=(d,p)
            except (ValueError,KeyError,TypeError,UnicodeDecodeError):continue
    absent=sorted(expected-set(candidates))
    if absent:raise RuntimeError('CHƯA MỞ TEST. Thiếu '+str(len(absent))+' FINAL job: '+str(absent))
    hashes={(v['source_train_X_sha256'],v['source_train_y_sha256']) for v,p in candidates.values()}
    assert len(hashes)==1,'Final train provenance khác nhau giữa jobs'
    # Search B in mounted Input. Verify manifest/hash + locked test sidecar before reading arrays.
    b_runs=sorted({p.parent.parent for p in Path('/kaggle/input').rglob('test_4721_LOCKED.npz')
                   if p.parent.name=='final_data' and p.parent.parent.name=='KLTN_HyperGAMO_B_50e988fae260'})
    if B_RUN is not None:b_runs=[Path(B_RUN)]
    assert b_runs,'Thiếu Input B test_4721_LOCKED.npz'
    root=None
    for candidate in b_runs:
        if not (candidate/'manifest.json').is_file():continue
        digest=hashlib.sha256((candidate/'manifest.json').read_bytes()).hexdigest()
        if digest=='41486ba39177e36f32e09c61cff870f4086a94c7b55f39db9dbe4b92093d6e67':
            root=candidate;break
    assert root is not None,'Không thấy đúng manifest Protocol B gốc'
    test=root/'final_data/test_4721_LOCKED.npz'
    side=root/'final_data/test_4721_LOCKED.npz.json'
    assert side.is_file(),'Thiếu B test sidecar'
    test_side=json.loads(side.read_text())
    assert test_side['contract']['run_id']=='50e988fae260','B test run identity changed'
    assert test_side['sha256']==hashlib.sha256(test.read_bytes()).hexdigest(),'B test hash không khớp sidecar'
    with np.load(test,allow_pickle=False) as d:
        x=d['X_test'];y=d['y_test'];coords=d['coords_test']
    assert x.shape==(4721,11,11,40,1) and y.shape==(4721,) and coords.shape==(4721,2),'B test arrays shape changed'
    assert np.isfinite(x).all() and np.all((y>=0)&(y<16)),'Invalid B test'
    # Actual inference only after 12-file gate.
    import tensorflow as tf
    import tf_keras as keras
    from sklearn.metrics import confusion_matrix
    PAIRED_C_DROPOUT=keras.layers.Dropout
    def make_c():
        L=keras.layers;inp=L.Input(shape=(11,11,40,1))
        z=L.Conv3D(8,(1,1,7),activation='relu',padding='same')(inp)
        z=L.Conv3D(16,(3,3,5),activation='relu',padding='same')(z)
        z=L.Conv3D(32,(5,5,7),activation='relu',padding='same')(z)
        s=z.shape;z=L.Reshape((s[1],s[2],s[3]*s[4]))(z)
        z=L.Conv2D(64,(3,3),activation='relu',padding='same')(z)
        z=L.GlobalAveragePooling2D()(z)
        z=L.Dropout(0.5)(z)
        out=L.Dense(16,activation='softmax',kernel_initializer='he_normal')(z)
        return keras.Model(inp,out,name='classifier_3d_hypergamo')
    target=Path('/kaggle/working/M4_IP_FINAL_TEST_RESULTS')
    if target.exists():raise RuntimeError('Thư mục test đã tồn tại; không ghi đè kết quả cũ: '+str(target))
    target.mkdir(parents=True)
    metrics=[];perclass=[]
    for a,s in sorted(expected):
        info,weights=candidates[a,s]
        keras.backend.clear_session();keras.utils.set_random_seed(s)
        model=make_c();model.load_weights(str(weights))
        p_sorted=model.predict(x,batch_size=64,verbose=0)
        order=np.asarray(info['class_order_original_ids'],np.int64)-1
        assert np.array_equal(np.sort(order),np.arange(16))
        prob=np.empty_like(p_sorted);prob[:,order]=p_sorted
        pred=np.argmax(prob,axis=1)
        prec,recall,f1,support=precision_recall_fscore_support(y,pred,labels=np.arange(16),zero_division=0)
        row={'arm':a,'seed':s,'OA':float(accuracy_score(y,pred)),'AA':float(recall.mean()),
             'Macro_F1':float(f1.mean()),
             'GM':float(np.exp(np.log(recall).mean())) if np.all(recall>0) else 0.0,
             'Kappa':float(cohen_kappa_score(y,pred))}
        metrics.append(row)
        for j in range(16):perclass.append({'arm':a,'seed':s,'class_id':j+1,'precision':prec[j],
                                            'recall':recall[j],'f1':f1[j],'support':support[j]})
        np.savez_compressed(target/f'{a}_seed{s}_predictions.npz',y_true=y,y_pred=pred,
                            coords=coords,probabilities=prob,
                            confusion=confusion_matrix(y,pred,labels=np.arange(16)))
        print('EVAL',a,s,{k:round(v,4) for k,v in row.items() if k not in ('arm','seed')})
    df=pd.DataFrame(metrics);df.to_csv(target/'metrics_by_seed.csv',index=False)
    pd.DataFrame(perclass).to_csv(target/'per_class.csv',index=False)
    rows=[]
    for a,group in df.groupby('arm',sort=True):
        line={'arm':a}
        for col in ('OA','AA','Macro_F1','GM','Kappa'):
            line[col+'_mean']=float(group[col].mean())
            line[col+'_sd']=float(group[col].std(ddof=1))
        rows.append(line)
    summary=pd.DataFrame(rows);summary.to_csv(target/'summary_mean_sd.csv',index=False)
    comparison=df.pivot(index='seed',columns='arm',values='Macro_F1')
    comparison['D_minus_A']=comparison['learned_local']-comparison['original_gan']
    comparison['D_minus_B']=comparison['learned_local']-comparison['matched_ros']
    comparison['D_minus_C']=comparison['learned_local']-comparison['random_local']
    comparison.to_csv(target/'paired_macro_f1_deltas.csv')
    limitation={'test':'Protocol B original reused test 4721','independent_untouched_holdout':False,
                'training_budget_C_updates_each_job':8000,
                'B_historical_HyperGAMO_Macro_F1_mean_pct':81.428,
                'B_historical_HyperGAMO_OA_mean_pct':88.322,
                'historical_B_bitwise_equal_to_matched_A':False,
                'warning':'Do not tune V2 based on this test; compare A vs D matched and show historical B as context.'}
    (target/'evaluation_scope.json').write_text(json.dumps(limitation,indent=2))
    archive=Path('/kaggle/working/M4_IP_FINAL_TEST_RESULTS.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(target.rglob('*')):
            if p.is_file():z.write(p,arcname=str(p.relative_to(target)))
    print('\n=== Tổng hợp mean ± SD (fraction, ×100 để ra %) ===')
    display(summary)
    print('Paired Macro-F1 deltas (fraction):');display(comparison)
    from IPython.display import FileLink,display
    display(FileLink(str(archive)))
    return summary

if ACTION=='evaluate_final':
    final_metrics=evaluate_all_final()
else:
    print('Test đang khóa. Cell này không đọc test khi ACTION=train.')