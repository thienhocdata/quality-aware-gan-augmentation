"""Read saved research outputs on CPU. No TensorFlow import or model execution.

API
---
discover_bundles(roots) -> dict[role, SavedBundle]; roots may be one path or a list.
load_bundle(path, expected_role=None) -> SavedBundle; accepts ZIP, directory, or ZIP!prefix.
verify_sha_manifest(bundle) -> serializable verification record.
run_saved_review(input_root, out_root) -> serializable dict with statuses, tables,
figures, paths, bundles. `tables` maps stable names to saved CSV paths.

Input may be six section ZIPs, automatically unpacked Kaggle directories, or the
combined upload ZIP with one bundle manifest in each role subdirectory.
All scientific arrays are read with allow_pickle=False. Original files are never
modified. CSV metrics are recomputed only from saved prediction arrays.
"""
from pathlib import Path, PurePosixPath
import hashlib, io, json, zipfile, itertools, datetime
import numpy as np
import pandas as pd

ROLES = ('protocol_b','m4_training','m4_test','classification_diagnostics','weighting_8000','endgame_status')
ARMS = ('original_gan','matched_ros','random_local','learned_local')
SEEDS = (42,43,44)
METRICS = ('OA','AA','Macro_F1','GM','Kappa')
CLASS_NAMES = ('Alfalfa','Corn-notill','Corn-mintill','Corn','Grass-pasture',
    'Grass-trees','Grass-pasture-mowed','Hay-windrowed','Oats','Soybean-notill',
    'Soybean-mintill','Soybean-clean','Wheat','Woods',
    'Buildings-Grass-Trees-Drives','Stone-Steel-Towers')

def require(ok,message):
    if not ok: raise RuntimeError(message)

def sha_bytes(b): return hashlib.sha256(b).hexdigest()

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

class SavedBundle:
    def __init__(self,path,prefix=''):
        self.path=Path(path); self.prefix=prefix.strip('/')
        if self.prefix:self.prefix+='/'
        self.manifest=json.loads(self.read('bundle_manifest.json').decode('utf-8-sig'))
        require(self.manifest.get('schema')=='kltn.saved_outputs.v1','Unsupported saved-output manifest')
        self.role=self.manifest['role']; self.status=self.manifest['status']
        require(self.role in ROLES,'Unknown bundle role '+str(self.role))

    def read(self,name):
        rel=PurePosixPath(name)
        require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe relative member '+name)
        if self.path.is_dir():
            return (self.path/Path(*rel.parts)).read_bytes()
        with zipfile.ZipFile(self.path) as z:return z.read(self.prefix+str(rel))

    def names(self):return sorted(self.manifest['members'])
    def find(self,suffix):
        matches=[n for n in self.names() if n.endswith(suffix)]
        require(len(matches)==1,f'{self.role}: expected exactly one {suffix}, found {len(matches)}')
        return matches[0]
    def json(self,name):return json.loads(self.read(name).decode('utf-8-sig'))
    def csv(self,name):return pd.read_csv(io.BytesIO(self.read(name)))
    def reference(self):return str(self.path)+('!'+self.prefix if self.prefix else '')

def load_bundle(path,expected_role=None):
    if isinstance(path,SavedBundle):b=path
    else:
        parts=str(path).split('!',1); p=Path(parts[0]); prefix=parts[1] if len(parts)==2 else ''
        if p.is_file() and p.name=='bundle_manifest.json':p=p.parent
        b=SavedBundle(p,prefix)
    require(expected_role is None or b.role==expected_role,f'Expected {expected_role}, got {b.role}')
    return b

def discover_bundles(roots):
    if isinstance(roots,dict):return {role:load_bundle(path,role) for role,path in roots.items()}
    if isinstance(roots,(str,Path)):roots=[roots]
    found={}; candidates=[]
    for raw in roots:
        p=Path(raw)
        if '!' in str(raw):candidates.append(load_bundle(raw));continue
        if p.is_dir():
            manifests=sorted(p.rglob('bundle_manifest.json'))
            candidates.extend(SavedBundle(m.parent) for m in manifests)
            archives=sorted(p.rglob('*.zip'))
        else:archives=[p]
        for archive in archives:
            try:
                with zipfile.ZipFile(archive) as z:
                    names=[n for n in z.namelist() if n=='bundle_manifest.json' or n.endswith('/bundle_manifest.json')]
                for n in names:candidates.append(SavedBundle(archive,n[:-len('bundle_manifest.json')]))
            except (zipfile.BadZipFile,OSError):continue
    for b in candidates:
        if b.role in found:
            prior=found[b.role]
            require(prior.manifest['members']==b.manifest['members'] and prior.status==b.status,
                f'Conflicting saved bundles for {b.role}: {prior.reference()} and {b.reference()}')
        else:found[b.role]=b
    require(all(r in found for r in ROLES),'Missing section bundles: '+', '.join(r for r in ROLES if r not in found))
    return found

def verify_sha_manifest(bundle):
    b=load_bundle(bundle); checked=0; total=0
    for name,pin in b.manifest['members'].items():
        data=b.read(name)
        require(len(data)==pin['bytes'] and sha_bytes(data)==pin['sha256'],f'Integrity mismatch: {b.role}/{name}')
        checked+=1;total+=len(data)
    return {'role':b.role,'status':'ALL_MEMBER_SHA256_PASS','checked_members':checked,'uncompressed_bytes':total}

def _save_csv(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding='utf-8-sig');return str(path)

def _aggregate(frame,keycols,metrics):
    rows=[]
    for keys,group in frame.groupby(keycols,sort=False):
        if not isinstance(keys,tuple):keys=(keys,)
        row=dict(zip(keycols,keys));row['seeds']=len(group)
        require(sorted(group['seed'].tolist())==list(SEEDS),'Expected all three trained seeds in '+str(keys))
        for metric in metrics:
            vals=group[metric].to_numpy(float)
            row[metric+'_mean']=float(vals.mean());row[metric+'_sd']=float(vals.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_protocol_b(bundle,out):
    b=load_bundle(bundle,'protocol_b');out=Path(out)
    by=b.csv(b.find('/tables/test_by_seed.csv'))
    require(len(by)==15 and by.groupby('method').size().eq(3).all(),'Historical B registry is not five x three')
    summary=_aggregate(by,['method'],METRICS)
    return {'protocol_b_by_seed':_save_csv(by,out/'protocol_b_by_seed.csv'),
            'protocol_b_summary':_save_csv(summary,out/'protocol_b_summary_mean_sd.csv'),
            'protocol_b_original_formatted_summary':_save_csv(b.csv(b.find('/tables/test_summary_mean_sd.csv')),out/'original_formatted_summary.csv')}

def summarize_m4_training(bundle,out):
    b=load_bundle(bundle,'m4_training'); registry=b.json('training_registry.json')
    require({(j['arm'],int(j['seed'])) for j in registry}==set(itertools.product(ARMS,SEEDS)) and len(registry)==12,
            'Training registry must contain exactly four arms x three seeds')
    rows=[]
    for job in registry:
        completed=job['completed']; arm=job['arm'];seed=int(job['seed'])
        require(completed['complete'] is True and completed['c_updates']==8000,'M4 job incomplete')
        prefix=f'jobs/{arm}/seed_{seed}/'
        actual=b.json(prefix+'completed.json');contract=b.json(prefix+'contract.json')
        require(actual==completed,'Registry/completed metadata mismatch')
        require(contract.get('protocol_B_id')=='50e988fae260','Wrong protocol in job contract')
        rows.append({'arm':arm,'seed':seed,'stage':completed['stage'],'status':completed['status'],
            'c_updates':completed['c_updates'],'adam_clock':completed['adam_clock'],
            'protocol_B_id':job['protocol_B_id'],'worker_sha256':job['worker_sha256'],
            'classifier_sha256':completed['model_sha256'],'source_id':job['source'],'source_member_prefix':job['prefix']})
    return {'training_registry':_save_csv(pd.DataFrame(rows),Path(out)/'training_registry.csv')}

def _metrics(y,pred):
    cm=np.bincount(y*16+pred,minlength=256).reshape(16,16)
    support=cm.sum(1);predicted=cm.sum(0);tp=cm.diagonal()
    recall=np.divide(tp,support,out=np.zeros(16,float),where=support>0)
    precision=np.divide(tp,predicted,out=np.zeros(16,float),where=predicted>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(16,float),where=(support+predicted)>0)
    oa=float(tp.sum()/len(y));expected=float((support.astype(float)*predicted).sum()/len(y)**2)
    metrics={'OA':oa,'AA':float(recall.mean()),'Macro_F1':float(f1.mean()),
        'GM':float(np.exp(np.log(recall).mean())) if np.all(recall>0) else 0.,
        'Kappa':float((oa-expected)/(1-expected)) if expected<1 else 1.}
    rows=[{'class_id':i+1,'class_name':CLASS_NAMES[i],'support':int(support[i]),
        'precision':float(precision[i]),'recall':float(recall[i]),'F1':float(f1[i])} for i in range(16)]
    return metrics,rows,cm

def analyze_saved_test(bundle,out):
    b=load_bundle(bundle,'m4_test');out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=b.json(b.find('/manifest.json'));registry=b.json(b.find('/evaluation_registry.json'))
    require(manifest.get('complete') is True and manifest.get('completed_model_count')==12,'Original test manifest incomplete')
    models=registry['models']; require(len(models)==12 and {(r['arm'],r['seed']) for r in models}==set(itertools.product(ARMS,SEEDS)),
        'Original saved test registry is not exactly twelve')
    require(registry['protocol_B_id']=='50e988fae260' and registry['fixed_C_updates']==8000,'Wrong test protocol/budget')
    by=[];perclass=[];reference=None;cms={}
    for model in models:
        arm,seed=model['arm'],int(model['seed']);suffix=f'/test_results/{arm}/seed_{seed}/'
        with np.load(io.BytesIO(b.read(b.find(suffix+'predictions.npz'))),allow_pickle=False) as a:
            y=np.asarray(a['y_true'],np.int64); pred=np.asarray(a['y_pred'],np.int64)
            prob=np.asarray(a['probabilities']);coords=np.asarray(a['coords']);saved_cm=np.asarray(a['confusion_matrix'])
        require(y.shape==pred.shape==(4721,) and prob.shape==(4721,16) and coords.shape==(4721,2),'Unexpected saved test shapes')
        require(np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-4,rtol=1e-5),'Invalid saved probabilities')
        require(np.all((y>=0)&(y<16)) and np.all((pred>=0)&(pred<16)),'Expected original test worker zero-based labels 0..15')
        require(np.array_equal(np.argmax(prob,axis=1),pred),'Saved predictions disagree with saved probabilities')
        if reference is None:reference=(y.copy(),coords.copy())
        require(np.array_equal(y,reference[0]) and np.array_equal(coords,reference[1]),'Test sample order differs between classifiers')
        scores,classes,cm=_metrics(y,pred);saved=b.json(b.find(suffix+'metrics.json'))
        require(np.array_equal(cm,saved_cm),'Saved confusion matrix mismatch')
        for key,val in scores.items():require(np.isclose(val,saved[key],rtol=0,atol=1e-12),'Recomputed '+key+' differs from original')
        require(saved['classifier_sha256']==model['classifier_sha256'],'Saved classifier provenance mismatch')
        by.append({'arm':arm,'seed':seed,'test_samples':len(y),**scores})
        perclass.extend({'arm':arm,'seed':seed,**row} for row in classes);cms[(arm,seed)]=cm
        _save_csv(pd.DataFrame(cm,index=range(1,17),columns=[f'pred_{i}' for i in range(1,17)]).rename_axis('true_class').reset_index(),out/f'confusion_{arm}_seed{seed}.csv')
    by=pd.DataFrame(by);perclass=pd.DataFrame(perclass)
    summary=_aggregate(by,['arm'],METRICS); pcs=_aggregate(perclass,['arm','class_id','class_name','support'],('precision','recall','F1'))
    paired=[]
    for other in ARMS[:-1]:
        for seed in SEEDS:
            left=by[(by.arm=='learned_local')&(by.seed==seed)].iloc[0]
            right=by[(by.arm==other)&(by.seed==seed)].iloc[0]
            paired.append({'contrast':'learned_local - '+other,'seed':seed,**{m:float(left[m]-right[m]) for m in METRICS}})
    paired=pd.DataFrame(paired)
    tables={'test_by_seed':_save_csv(by,out/'test_by_seed.csv'),
        'test_summary_mean_sd':_save_csv(summary,out/'test_summary_mean_sd.csv'),
        'test_per_class_by_seed':_save_csv(perclass,out/'test_per_class_by_seed.csv'),
        'test_per_class_mean_sd':_save_csv(pcs,out/'test_per_class_mean_sd.csv'),
        'paired_differences':_save_csv(paired,out/'paired_differences.csv')}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=[]
    fig,ax=plt.subplots(figsize=(9,4.8))
    for arm in ARMS:
        group=by[by.arm==arm].sort_values('seed');ax.plot(group.seed,100*group.Macro_F1,'o-',label=arm)
    ax.set(xticks=list(SEEDS),xlabel='Trained seed',ylabel='Saved-test Macro-F1 (%)',title='M4: three trained seeds per arm')
    ax.grid(alpha=.25);ax.legend(fontsize=9);fig.tight_layout();p=out/'macro_f1_by_seed.png';fig.savefig(p,dpi=160);plt.close(fig);figures.append(str(p))
    grid=pcs.pivot(index='class_id',columns='arm',values='F1_mean').reindex(columns=ARMS)
    fig,ax=plt.subplots(figsize=(8,8));im=ax.imshow(100*grid.to_numpy(),vmin=0,vmax=100,cmap='viridis',aspect='auto')
    ax.set_xticks(range(4),ARMS,rotation=25,ha='right');ax.set_yticks(range(16),[f'{i+1} {n}' for i,n in enumerate(CLASS_NAMES)],fontsize=8)
    ax.set_title('Per-class F1: mean of three trained seeds (%)');fig.colorbar(im,ax=ax);fig.tight_layout()
    p=out/'per_class_mean_f1.png';fig.savefig(p,dpi=160);plt.close(fig);figures.append(str(p))
    return tables,figures

def summarize_weighting(bundle,out):
    b=load_bundle(bundle,'weighting_8000');out=Path(out);manifest=b.json(b.find('/analysis_manifest.json'))
    require(manifest.get('complete') is True and manifest['completed_seeds']==list(SEEDS) and manifest['c_updates']==8000,
        'Weighting report is not complete for all three final checkpoints')
    require(manifest['arm']=='learned_local' and manifest['protocol_B_id']=='50e988fae260','Wrong weighting source')
    rows=[]
    minority=[c for c in range(1,17) if c!=11]
    for seed in SEEDS:
        for cid in minority:
            name=b.find(f'/seed_{seed}/weights/class_{cid:02d}.npz')
            with np.load(io.BytesIO(b.read(name)),allow_pickle=False) as a:
                w=np.asarray(a['weights'],np.float64);logits=np.asarray(a['actual_dense_logits'],np.float64)
                neighbors=np.asarray(a['neighbor_local_ids'],np.int64)
                require(int(a['model_seed'])==seed and int(a['class_id'])==cid and float(a['rho'])==.2,'Wrong saved weighting identity')
            require(w.ndim==3 and w.shape[1:]==(64,3) and np.isfinite(w).all(),'Invalid saved weight shape/value')
            require(np.all(w>=0) and np.allclose(w.sum(-1),1,atol=1e-6),'Weights not probabilities')
            selected=logits[:,neighbors].transpose(1,0,2)
            shifted=selected-selected.max(-1,keepdims=True);ew=np.exp(shifted);restored=ew/ew.sum(-1,keepdims=True)
            require(np.allclose(w,restored,atol=2e-6,rtol=1e-5),'Actual Dense logits do not reproduce saved masked support weights')
            entropy=-np.sum(w*np.log(np.maximum(w,np.finfo(float).tiny)),axis=-1)
            dominant=w.argmax(-1);changed=(dominant!=dominant[:,0:1]).any(axis=1)
            rows.append({'model_seed':seed,'class_id':cid,'class_name':CLASS_NAMES[cid-1],
                'anchors':len(w),'noises_per_anchor':64,'weight_triplets':len(w)*64,
                'entropy_mean_nats':float(entropy.mean()),'effective_neighbors_mean':float(np.exp(entropy).mean()),
                'maximum_weight_mean':float(w.max(-1).mean()),'fraction_max_weight_gt_0_999':float((w.max(-1)>.999).mean()),
                'mean_weight_std_over_noise':float(w.std(axis=1).mean()),
                'anchors_with_argmax_change':int(changed.sum()),'fraction_anchors_with_argmax_change':float(changed.mean()),
                'rank_1_weight_mean':float(w[:,:,0].mean()),'rank_2_weight_mean':float(w[:,:,1].mean()),'rank_3_weight_mean':float(w[:,:,2].mean()),
                'actual_dense_softmax_check_max_abs':float(np.max(np.abs(w-restored)))})
    table=pd.DataFrame(rows);require(len(table)==45,'Missing minority head/seed weighting arrays')
    eq=[]
    for seed,group in table.groupby('model_seed'):
        eq.append({'model_seed':int(seed),'minority_classes':len(group),'anchors':int(group.anchors.sum()),
            'weight_triplets':int(group.weight_triplets.sum()),'anchors_with_argmax_change':int(group.anchors_with_argmax_change.sum()),
            **{m:float(group[m].mean()) for m in ('entropy_mean_nats','effective_neighbors_mean','maximum_weight_mean',
                'fraction_max_weight_gt_0_999','mean_weight_std_over_noise','fraction_anchors_with_argmax_change')}})
    tables={'weighting_recomputed_by_class':_save_csv(table,out/'weighting_recomputed_from_actual_arrays.csv'),
        'weighting_recomputed_seed_summary':_save_csv(pd.DataFrame(eq),out/'weighting_equal_class_summary_recomputed.csv'),
        'weighting_seed_summary':_save_csv(b.csv(b.find('/per_seed_equal_class_summary.csv')),out/'original_equal_class_seed_summary.csv'),
        'weighting_per_class_saved':_save_csv(b.csv('original/per_class_metrics.csv'),out/'original_per_class_metrics.csv')}
    figures=[]
    for name in b.names():
        if '/figures/' in name and name.endswith('.png'):
            target=out/'original_figures'/Path(name).name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b.read(name));figures.append(str(target))
    return tables,figures

def run_saved_review(input_root,out_root):
    """Verify all six bundles and regenerate saved-result tables/figures, on CPU."""
    bundles=discover_bundles(input_root);out=Path(out_root);out.mkdir(parents=True,exist_ok=True)
    verification=[verify_sha_manifest(bundles[r]) for r in ROLES]
    tables={};figures=[]
    tables.update(summarize_protocol_b(bundles['protocol_b'],out/'01_PROTOCOL_B'))
    tables.update(summarize_m4_training(bundles['m4_training'],out/'02_M4_TRAINING'))
    new,fig=analyze_saved_test(bundles['m4_test'],out/'03_M4_SAVED_TEST');tables.update(new);figures+=fig
    new,fig=summarize_weighting(bundles['weighting_8000'],out/'05_V2_WEIGHTING_8000');tables.update(new);figures+=fig
    pending=bundles['endgame_status'].json('ENDGAME_STATUS.json')
    (out/'06_ENDGAME_STATUS.json').write_text(json.dumps(pending,ensure_ascii=False,indent=2),encoding='utf-8')
    info={r:{'path':b.reference(),'status':b.status,'sources':b.manifest['sources'],
        'archive_sha256':sha_file(b.path) if b.path.is_file() else None} for r,b in bundles.items()}
    result={'mode':'READ_SAVED_RESULTS_CPU','training_performed':False,'model_inference_performed':False,
        'test_inference_performed':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'statuses':{r:bundles[r].status for r in ROLES},'verification':verification,'tables':tables,'figures':figures,'bundles':info,
        'limitations':['B has cross-split spatial patch reuse; this consolidation does not eliminate spatial leakage.',
            'The Indian Pines test was viewed previously. Further design is post-hoc motivated.',
            'Three trained seeds are the replication units; test pixels, anchors and noise draws are not independent training runs.',
            'The weighting report is a final frozen endpoint observation; it does not establish a cause for seed 43 test performance.',
            'No completed 7500-to-8000 forward report was found. Its six checkpoints exist but the temporal diagnostic remains pending.',
            'M3 standalone report not located. This does not invalidate or recreate the existing M4 artifacts.']}
    summary=pd.read_csv(tables['test_summary_mean_sd'])
    lines=['# Consolidated saved-results review','',
        'CPU read-only review of saved outputs. No training, model construction, checkpoint modification or test inference.','',
        '## M4 saved-test results','', '| Arm | Macro-F1 mean ± sample SD (%) | OA mean ± sample SD (%) |',
        '|---|---:|---:|']
    for row in summary.to_dict('records'):
        lines.append(f"| {row['arm']} | {100*row['Macro_F1_mean']:.3f} ± {100*row['Macro_F1_sd']:.3f} | {100*row['OA_mean']:.3f} ± {100*row['OA_sd']:.3f} |")
    lines+=['','## Boundaries','']+['- '+v for v in result['limitations']]
    report=out/'REPORT.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    result['paths']={'report_md':str(report),'manifest':str(out/'review_manifest.json'),'report_zip':str(out.parent/(out.name+'.zip'))}
    (out/'SOURCES.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'review_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(result['paths']['report_zip'],'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(out)).replace('\\','/'))
    return result
