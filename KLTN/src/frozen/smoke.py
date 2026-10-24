# -*- coding: utf-8 -*-
"""Kaggle-only V2 integration smoke. No accuracy, no study training, no test.

The notebook embeds this complete source; no separate .py upload is needed.
"""
from __future__ import annotations
import gc
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

SHAPE = (11, 11, 40, 1)
FLAT = 4840
N_CLASSES = 16
K = 3
RHO = 0.2
RUN_ID = '50e988fae260'
IMPLEMENTATION = '5e15f8214bf273bb889187b96c80407a8000e5e1a3ef30734bcfa86ced275a6f'
EXPECTED_COUNTS = [17,602,341,110,167,301,13,182,8,375,1112,249,100,576,230,39]
# Pinned from the Kaggle pilot_contract.json previously received from the user.
# These are ORIGINAL B file hashes, not hashes of locally reconstructed caches.
PINNED_B = {
    'manifest.json':'41486ba39177e36f32e09c61cff870f4086a94c7b55f39db9dbe4b92093d6e67',
    'protocol_spec.json':'d4260da1ba9289a3183ab10905f995d6a2e8956b5651576734683017705bd202',
    'development/dev_split.npz':'1dcb4dac430f756d3bacc68d4e741bb5e34175ca018b744a6f3867e91b2cf608',
    'development/dev_split.npz.json':'72babb8a6056e3e298051a3581f03ae45256b556f11251a6377224ab354711b0',
    'development/dev_train_val.npz':'bf97ea110113ffd94afe1ba8a87d89a7ee3f5ad35efd3e3e214d5e300849fcc6',
    'development/dev_train_val.npz.json':'726cfc2135e316dd93f301dcc3babcd65e98ce0565d836470ede04c40ec02326',
    'development/preprocessing_dev.joblib':'d143a6d5f972fb07328e6c137d42a8332f91e06cbbbd96f41c8c5ea452a2dde1',
    'development/preprocessing_dev.joblib.json':'f5c80cf8a3e4f204c9caf15df948933466e6ca9028606c11cec8da9e0ebb6458',
}
np = tf = keras = None
CONFIG = REPORT = FOLDER = None
START = None


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            deadline()
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def deadline():
    if CONFIG is not None and time.monotonic() >= CONFIG['deadline_monotonic']:
        raise TimeoutError('Smoke exceeded the total 30-minute deadline, including startup/compile.')


def phase(name):
    deadline()
    REPORT['phase'] = name
    REPORT['elapsed_seconds'] = round(time.monotonic()-START, 3)
    atomic_json(FOLDER/'smoke_report.json', REPORT)
    print(f"[{REPORT['elapsed_seconds']:.1f}s] {name}", flush=True)


def b_array_sha(value):
    # This is B's hash format, different from the smoke sha_array helper below.
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.shape).encode())
    h.update(value.dtype.str.encode())
    h.update(value.tobytes())
    return h.hexdigest()


def load_input_verified():
    mode = CONFIG['data_mode']
    require(mode in ('auto_real_B', 'fixture'), 'Unknown data_mode')
    if mode == 'fixture':
        require(CONFIG['b_run'] is None, 'fixture mode must not also select a B run')
        rng = np.random.default_rng(19)
        y = np.concatenate([np.repeat(np.arange(15),4),np.repeat(15,6)]).astype(np.int64)
        X = rng.normal(0,.25,size=(len(y),*SHAPE)).astype(np.float32)
        return X,y,{'source':'synthetic_fixture','real_B_cache_verified':False,
                    'note':'Shape/wiring test only; not an Indian Pines data integration result.'}
    if CONFIG['b_run'] is not None:
        roots = [Path(CONFIG['b_run'])]
    else:
        preferred = Path('/kaggle/input/notebooks/minhlongg/notebook708cde343b')/f'KLTN_HyperGAMO_B_{RUN_ID}'
        roots = sorted(({preferred} if preferred.is_dir() else set()) | {
            p.parent.parent for p in Path('/kaggle/input').rglob('dev_train_val.npz')
            if p.parent.name == 'development' and p.parent.parent.name == f'KLTN_HyperGAMO_B_{RUN_ID}'
        })
    require(roots, 'Missing B output input. Add the original notebook Output containing KLTN_HyperGAMO_B_50e988fae260. No synthetic fallback was made.')
    valid, rejected = [], []
    for root in roots:
        deadline()
        try:
            actual = {}
            for relative, expected in PINNED_B.items():
                path = root/relative
                require(path.is_file(), f'Missing original B artifact: {path}')
                actual[relative] = file_sha(path)
                require(actual[relative] == expected, f'Original B hash mismatch: {path}')
            valid.append((root, actual))
        except RuntimeError as error:
            rejected.append(str(error))
    require(valid, 'No original B cache passed pinned hashes. '+ '; '.join(rejected))
    # Every eligible copy has the same eight pinned hashes, so sorted first is
    # deterministic even when multiple identical Kaggle input copies exist.
    root, actual = sorted(valid, key=lambda item: str(item[0]))[0]
    read_json = lambda relative: json.loads((root/relative).read_text(encoding='utf-8'))
    manifest = read_json('manifest.json')
    spec = read_json('protocol_spec.json')
    require(manifest['run_id'] == RUN_ID, 'Wrong B run_id')
    require(manifest['protocol_spec_sha256'] == actual['protocol_spec.json'], 'Broken B manifest/spec link')
    require(hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12] == RUN_ID, 'Protocol/run-id hash mismatch')
    require(spec['implementation_sha256'] == IMPLEMENTATION and spec['patch_size'] == 11 and spec['pca_components'] == 40, 'Wrong B implementation/preprocessing config')
    for relative in ('development/dev_train_val.npz','development/dev_split.npz','development/preprocessing_dev.joblib'):
        sidecar = read_json(relative+'.json')
        require(sidecar['sha256'] == actual[relative], f'Broken sidecar hash: {relative}')
        require(sidecar['contract']['run_id'] == RUN_ID and sidecar['contract']['implementation'] == IMPLEMENTATION, f'Broken contract: {relative}')
    cache_contract = read_json('development/dev_train_val.npz.json')['contract']
    prep_contract = read_json('development/preprocessing_dev.joblib.json')['contract']
    require(cache_contract['kind'] == 'dev_cache', 'Not a development cache')
    require(cache_contract['preprocessing_sha256'] == actual['development/preprocessing_dev.joblib'], 'Cache/preprocessing hash mismatch')
    require(prep_contract['pca_components'] == 40 and prep_contract['pca_solver'] == 'full', 'Wrong PCA metadata')
    # Read only these three train arrays. Do not deserialize validation/test,
    # and never execute joblib/pickle preprocessing objects.
    with np.load(root/'development/dev_train_val.npz',allow_pickle=False) as data:
        X = data['X_train']
        y = data['y_train']
        coords = data['coords_train']
    require(X.dtype == np.float32 and X.shape == (4422,*SHAPE), 'Wrong development train feature dtype/shape')
    require(y.dtype == np.int64 and y.shape == (4422,), 'Wrong train label dtype/shape')
    require(coords.shape == (4422,2) and np.issubdtype(coords.dtype,np.integer), 'Wrong train center coordinates')
    require(np.all((coords>=0)&(coords<145)) and len(np.unique(coords,axis=0)) == 4422, 'Invalid train centers')
    require(np.all((y>=0)&(y<16)) and np.bincount(y,minlength=16).tolist() == EXPECTED_COUNTS, 'Wrong B per-class development counts')
    require(np.isfinite(X).all(), 'Nonfinite train features')
    require(cache_contract['train_coords'] == b_array_sha(coords), 'Train coordinate contract mismatch')
    require(prep_contract['fit_coords_sha256'] == b_array_sha(coords), 'PCA was not fit on these development train centers')
    return X,y,{'source':'original_B_development_train','run_path':str(root),
                'real_B_cache_verified':True,'pinned_file_hashes':actual,
                'eligible_identical_copies':len(valid),'rejected_candidates':rejected,
                'train_coords_B_hash':b_array_sha(coords),
                'validation_arrays_loaded':False,'test_arrays_loaded':False,
                'joblib_deserialized':False}


def load_runtime():
    global tf, keras
    tfv,kv = metadata.version('tensorflow'),metadata.version('tf-keras')
    require(tfv.split('.')[:2] == kv.split('.')[:2], f'TensorFlow/tf-keras minor mismatch: {tfv}/{kv}. No automatic install/retry.')
    import tensorflow as tensorflow
    import tf_keras as oldkeras
    tf,keras = tensorflow,oldkeras
    physical = tf.config.list_physical_devices('GPU')
    require(len(physical) == 1, f'Expected exactly one visible GPU; got {len(physical)}. CPU fallback is disabled.')
    tf.config.set_visible_devices(physical[:1], 'GPU')
    tf.config.experimental.set_memory_growth(physical[0], True)
    keras.mixed_precision.set_global_policy('float32')
    require(hasattr(keras.optimizers,'legacy') and hasattr(keras.optimizers.legacy,'Adam'), 'tf_keras legacy Adam is required')
    logical = tf.config.list_logical_devices('GPU')
    require(len(logical) == 1, 'Expected one logical GPU')
    with tf.device('/GPU:0'):
        probe = tf.linalg.matmul(tf.ones((2,2)),tf.ones((2,2)))
    require('GPU:0' in probe.device and np.isfinite(probe.numpy()).all(), 'GPU execution probe failed')
    return {'tensorflow':tf.__version__,'tf_keras':keras.__version__,
            'physical_visible':[p.name for p in physical], 'logical_visible':[p.name for p in logical],
            'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
            'gpu_details':tf.config.experimental.get_device_details(physical[0]),
            'gpu_probe_device':probe.device,'cpu_fallback_allowed':False}


# Same C/D/G and V2 head functions as the submitted notebook.

def sha_array(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def math_gate():
    """Independent NumPy/FP64 finite-difference check. Not a GAN training test."""
    rng = np.random.default_rng(812)
    xs = rng.normal(size=(3, 9))
    anchor = rng.normal(size=9)
    logits = rng.normal(size=3)
    def f(l):
        x = np.exp(l - l.max()); w = x/x.sum()
        return (1-RHO)*anchor + RHO*(w @ xs),w
    y,w = f(logits); mu = w @ xs
    analytic = np.stack([RHO*w[j]*(xs[j] - mu) for j in range(3)],axis=1)
    eps = 1e-6
    numeric = np.stack([(f(logits + np.eye(3)[j]*eps)[0]
                       -f(logits - np.eye(3)[j]*eps)[0])/(2*eps) for j in range(3)],axis=1)
    error = float(np.max(np.abs(analytic-numeric)))
    assert error < 1e-8, error
    assert np.linalg.norm(y-anchor) <= RHO * np.linalg.norm(xs-anchor,axis=1).max() + 1e-12
    assert np.linalg.matrix_rank(analytic,tol=1e-9) == 2
    # Explicit k=1 finite-difference check: the only softmax weight is 1.
    one = lambda ell: (1-RHO)*anchor + RHO*(np.exp(ell-np.max(ell))/np.exp(ell-np.max(ell)).sum() @ xs[:1])
    k1 = (one(np.array([0.3+eps]))-one(np.array([0.3-eps])))/(2*eps)
    assert np.array_equal(k1,np.zeros_like(k1)), 'k=1 gradient was not zero'
    return {'numpy_jacobian_max_abs_error':error,
            'num_neighbor_mixing_degrees_of_freedom':2,
            'zero_gradient_if_k_is_1':True,
            'real_training_performed':False}


def build_graph_per_class(X, y):
    """Exact full-patch train-only kNN. Local supports sorted by global row order."""
    from scipy.spatial.distance import cdist
    counts = np.bincount(y, minlength=16)
    order = np.lexsort((np.arange(16),counts))
    if not np.all(counts[order[-1]] > counts[order[:-1]]) or np.any(counts[order[:-1]] < K+1):
        raise ValueError('Requires a strict single majority and >=4 train samples for each of 15 heads')
    inverse = np.empty(16, np.int64)
    inverse[order] = np.arange(16)
    y_sorted = inverse[y]
    supports, graph = [], []
    for h in range(15):
        support = np.asarray(X[y_sorted == h],np.float32)
        flat = support.reshape(len(support), FLAT).astype(np.float64)
        dist = cdist(flat, flat, metric='euclidean')
        np.fill_diagonal(dist,np.inf)
        # Tie-break by original train-row order (which is support-row order).
        nearest = np.empty((len(support),K),np.int32)
        for a in range(len(support)):
            nearest[a] = np.lexsort((np.arange(len(support)),dist[a]))[:K]
        assert not np.any(nearest == np.arange(len(support))[:,None])
        supports.append(support)
        graph.append(nearest)
    return {'supports':supports,'neighbor_graphs':graph,
            'counts':counts,'order':order,'inverse':inverse,'y_sorted':y_sorted}


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


def tao_generator(latent_dim, so_lop):
    L = keras.layers
    noise = L.Input(shape=(latent_dim,))
    labels = L.Input(shape=(so_lop,))
    x = L.Concatenate()([noise, labels])
    x = L.Dense(7*7*256, use_bias=False)(x)
    x = L.BatchNormalization(momentum=0.9)(x)
    x = L.LeakyReLU()(x)
    x = L.Reshape((7,7,256))(x)
    x = L.Conv2DTranspose(128, (5,5), strides=(1,1), padding="same", use_bias=False)(x)
    x = L.BatchNormalization(momentum=0.9)(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2DTranspose(64, (5,5), strides=(2,2), padding="same", use_bias=False)(x)
    x = L.BatchNormalization(momentum=0.9)(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2DTranspose(1, (5,5), strides=(2,2), padding="same", use_bias=False)(x)
    x = L.BatchNormalization(momentum=0.9)(x)
    out = L.Flatten()(x)  # 28×28 = 784
    return keras.Model([noise, labels], out, name="G")


def tao_discriminator(shape, so_lop):
    L = keras.layers
    inp = L.Input(shape=shape)
    x = L.Conv3D(8,(1,1,7),activation="relu",padding="same")(inp)
    x = L.Conv3D(16,(3,3,5),activation="relu",padding="same")(x)
    x = L.Conv3D(32,(5,5,7),activation="relu",padding="same")(x)
    s = x.shape
    x = L.Reshape((s[1],s[2],s[3]*s[4]))(x)
    x = L.Conv2D(64,(3,3),activation="relu",padding="same")(x)
    x = L.GlobalAveragePooling2D()(x)
    label = L.Input(shape=(so_lop,))
    x = L.Concatenate()([x,label])
    x = L.Dropout(.5)(x)
    out = L.Dense(1,activation="sigmoid",kernel_initializer="he_normal")(x)
    return keras.Model([inp,label],out,name="D")


def tao_head_v2(support, neighbors, rho=RHO):
    """B Dense(Nc) head, but local kNN masked softmax and a sampled anchor.

    Input 1: G output (batch,784). Input 2: head-local anchor row id (batch,1).
    The full Dense(Nc) remains trainable; no learned anchor choice and no
    learned rho. Supports/graph are fixed, non-trainable tensors.
    """
    assert support.ndim == 5 and support.shape[1:] == SHAPE
    n = len(support)
    assert neighbors.shape == (n,K)
    assert np.all((neighbors>=0) & (neighbors<n))
    assert not np.any(neighbors == np.arange(n)[:,None])
    L=keras.layers
    support_const=tf.constant(np.asarray(support.reshape(n,FLAT),np.float32))
    graph_const=tf.constant(neighbors,dtype=tf.int32)
    features=L.Input(shape=(784,),name='generator_features')
    anchor_ids=L.Input(shape=(1,),dtype='int32',name='head_local_anchor_id')
    # Crucial: Dense(Nc) has the SAME output size as the original B head.
    logits=L.Dense(n,activation=None,name='support_logits')(features)
    def local_mix(inputs):
        all_logits, ids=inputs
        ids=tf.reshape(tf.cast(ids,tf.int32),[-1])
        allowed=tf.gather(graph_const,ids)                    # (batch,3)
        allowed_logits=tf.gather(all_logits,allowed,axis=1,batch_dims=1)
        w=tf.nn.softmax(allowed_logits,axis=1)
        x0=tf.gather(support_const,ids)                       # (batch,4840)
        xs=tf.gather(support_const,allowed)                   # (batch,3,4840)
        return (1.0-rho)*x0 + rho*tf.einsum('bk,bkp->bp',w,xs)
    mixed=L.Lambda(local_mix,output_shape=(FLAT,),name='local_convex_mix')([logits,anchor_ids])
    output=L.Reshape(SHAPE,name='synthetic_patch')(mixed)
    return keras.Model([features,anchor_ids],output,name=f'head_v2_{n}')


def make_gradient_audit(owners):
    # Separate observational state: not an optimizer slot and not model weights.
    # Identity gradient_transformer observes actual compiled train_on_batch loss.
    names = list(owners)
    lookup = {}
    for i,name in enumerate(names):
        for variable in owners[name]['trainable']:
            require(variable.ref() not in lookup, 'A trainable variable has multiple owners')
            lookup[variable.ref()] = i
    with tf.device('/CPU:0'):
        audit = {'calls':tf.Variable(0,dtype=tf.int64,trainable=False),
                 'norm_sq':tf.Variable(np.zeros(len(names)),dtype=tf.float64,trainable=False),
                 'counts':tf.Variable(np.zeros(len(names),np.int32),trainable=False)}
    def observe(grads_and_vars):
        pairs = list(grads_and_vars)
        sums = [[] for _ in names]
        counts = [0]*len(names)
        finite_ops = []
        for gradient,variable in pairs:
            require(variable.ref() in lookup, 'Optimizer saw a variable with no registered owner')
            owner = lookup[variable.ref()]
            if gradient is None:
                continue
            require(not isinstance(gradient,tf.IndexedSlices), 'Unexpected sparse gradient in these Dense/Conv networks; stop rather than silently change sparse update semantics')
            finite_op = tf.debugging.assert_all_finite(gradient,'Nonfinite actual loss gradient')
            if finite_op is not None:
                finite_ops.append(finite_op)
            sums[owner].append(tf.reduce_sum(tf.square(tf.cast(gradient,tf.float64))))
            counts[owner] += 1
        with tf.control_dependencies(finite_ops):
            record_ops = [audit['norm_sq'].assign(tf.stack([tf.add_n(s) if s else tf.constant(0.,tf.float64) for s in sums])),
                          audit['counts'].assign(tf.constant(counts,tf.int32)),
                          audit['calls'].assign_add(1)]
        # Numerical identity; no clipping, resampling, extra forward/backward or
        # optimizer step. Control edges ensure the observation executes each call.
        with tf.control_dependencies(record_ops):
            return [(None if g is None else tf.identity(g),v) for g,v in pairs]
    return audit,observe


def build_v2_models(X,y,seed):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    prepared = build_graph_per_class(X,y)
    context = {}
    def identity_observer(grads_and_vars):
        require('observe' in context, 'Gradient observer was called before initialization')
        return context['observe'](grads_and_vars)
    # Same optimizer algorithm/hyperparameters as B; transformer only records
    # unchanged gradients actually supplied by the compiled Keras routes.
    opt = keras.optimizers.legacy.Adam(learning_rate=2e-4,beta_1=.5,
                                      gradient_transformers=[identity_observer])
    C,D,G = tao_classifier(SHAPE,16),tao_discriminator(SHAPE,16),tao_generator(100,16)
    C.compile(loss='mean_squared_error',optimizer=opt)
    D.compile(loss='mean_squared_error',optimizer=opt)
    heads = [tao_head_v2(s,g) for s,g in zip(prepared['supports'],prepared['neighbor_graphs'])]
    C.trainable=False; D.trainable=False
    vs_c,vs_d = [],[]
    for h,head in enumerate(heads):
        z=keras.layers.Input(shape=(100,));lab=keras.layers.Input(shape=(16,))
        anc=keras.layers.Input(shape=(1,),dtype='int32')
        mc=keras.Model([z,lab,anc],C(head([G([z,lab]),anc])),name=f'G_vs_C_{h}')
        mc.compile(loss='mean_squared_error',optimizer=opt)
        vs_c.append(mc)
        z2=keras.layers.Input(shape=(100,));lab2=keras.layers.Input(shape=(16,))
        lab3=keras.layers.Input(shape=(16,));anc2=keras.layers.Input(shape=(1,),dtype='int32')
        md=keras.Model([z2,lab2,lab3,anc2],D([head([G([z2,lab2]),anc2]),lab3]),name=f'G_vs_D_{h}')
        md.compile(loss='mean_squared_error',optimizer=opt)
        vs_d.append(md)
    # Preserve B's compile/freeze pattern; the smoke tests the resulting actual
    # updates instead of assuming compile captured the desired trainable set.
    C.trainable=True;D.trainable=True
    networks={'C':C,'D':D,'G':G,**{f'head_{h}':head for h,head in enumerate(heads)}}
    owners={name:{'trainable':tuple(net.trainable_variables),
                  'nontrainable':tuple(net.non_trainable_variables)} for name,net in networks.items()}
    all_variables=[v for owner in owners.values() for v in owner['trainable']]
    require(all('GPU:0' in v.device for v in all_variables), 'Model trainable variables were not all placed on the selected GPU')
    require(hasattr(opt,'_create_all_weights') and hasattr(opt,'get_slot'), 'Legacy Adam slot API unavailable; no optimizer fallback')
    opt._create_all_weights(all_variables)
    require(set(opt.get_slot_names()) == {'m','v'}, 'Unexpected Adam slot layout')
    audit,observe = make_gradient_audit(owners)
    context['observe'] = observe
    return dict(C=C,D=D,G=G,heads=heads,vs_c=vs_c,vs_d=vs_d,opt=opt,
                owners=owners,audit=audit,**prepared)


def state_snapshot(m):
    state = {}
    for name,owner in m['owners'].items():
        parts={part:[v.numpy().copy() for v in owner[part]] for part in ('trainable','nontrainable')}
        for slot in ('m','v'):
            values=[]
            for variable in owner['trainable']:
                value=m['opt'].get_slot(variable,slot)
                require(value is not None, f'Missing {slot} slot for {name}/{variable.name}')
                values.append(value.numpy().copy())
            parts[slot]=values
        for arrays in parts.values():
            require(all(np.isfinite(a).all() for a in arrays),f'Nonfinite state for {name}')
        state[name]=parts
    return state


def state_diff(before,after):
    require(before.keys() == after.keys(), 'Owner list changed')
    result={}
    for name in before:
        result[name]={}
        for part,arrays in before[name].items():
            other=after[name][part]
            require(len(arrays) == len(other),'Variable/slot count changed')
            changed=0;max_abs=0.
            for a,b in zip(arrays,other):
                require(a.shape == b.shape and a.dtype == b.dtype, 'Variable/slot signature changed')
                if not np.array_equal(a,b):
                    changed+=1
                    max_abs=max(max_abs,float(np.max(np.abs(a.astype(np.float64)-b))))
            result[name][part]={'changed_tensors':changed,'max_abs_delta':max_abs}
    return result


def fingerprint(m):
    h=hashlib.sha256()
    for name,parts in state_snapshot(m).items():
        for part,arrays in parts.items():
            h.update(f'{name}/{part}'.encode())
            for value in arrays:h.update(sha_array(value).encode())
    for value in m['opt'].get_weights():h.update(sha_array(value).encode())
    return h.hexdigest()


def inference_gate(m,z,condition,anchors):
    phase('forward / NumPy oracle / non-intrusive inference')
    before=state_snapshot(m);clock=int(m['opt'].iterations.numpy())
    observer_calls=int(m['audit']['calls'].numpy())
    features=m['G'].predict_on_batch([z,condition])
    fake=m['heads'][0].predict_on_batch([features,anchors])
    require(features.shape == (2,784) and fake.shape == (2,*SHAPE),'Wrong forward shape')
    require(np.isfinite(fake).all(),'Nonfinite generated patch')
    kernel,bias=m['heads'][0].get_layer('support_logits').get_weights()
    logits=features@kernel+bias
    support=m['supports'][0].reshape(-1,FLAT)
    expected=[]
    for i,anchor in enumerate(anchors.ravel()):
        neighbors=m['neighbor_graphs'][0][anchor]
        probs=np.exp(logits[i,neighbors]-logits[i,neighbors].max());probs/=probs.sum()
        expected.append(((1-RHO)*support[anchor]+RHO*(probs@support[neighbors])).reshape(SHAPE))
    np.testing.assert_allclose(fake,np.asarray(expected),rtol=2e-5,atol=4e-6)
    changes=state_diff(before,state_snapshot(m))
    require(all(v['changed_tensors']==0 for owner in changes.values() for v in owner.values()),'Inference changed weights/BN/Adam slots')
    require(clock == int(m['opt'].iterations.numpy()) and observer_calls == int(m['audit']['calls'].numpy()),'Inference performed an optimizer update')
    return fake,{'pass':True,'shape':list(fake.shape),'numpy_max_abs_error':float(np.max(np.abs(fake-np.asarray(expected))))}


def update_gate(m,label,fn,active,family):
    phase(label)
    active=set(active)
    before=state_snapshot(m)
    clock=int(m['opt'].iterations.numpy())
    audit_calls=int(m['audit']['calls'].numpy())
    value=fn()
    loss=float(np.asarray(value).reshape(-1)[0])
    require(np.isfinite(loss),f'{label}: nonfinite loss')
    require(int(m['opt'].iterations.numpy()) == clock+1,f'{label}: shared Adam clock did not advance exactly once')
    require(int(m['audit']['calls'].numpy()) == audit_calls+1,f'{label}: actual compiled-loss gradient observer did not execute exactly once')
    counts=m['audit']['counts'].numpy().tolist()
    norm_sq=m['audit']['norm_sq'].numpy().tolist()
    gradients={}
    for i,(name,owner) in enumerate(m['owners'].items()):
        wanted=len(owner['trainable']) if name in active else 0
        require(counts[i] == wanted,f'{label}: optimizer gradient variable count for {name}: got {counts[i]}, expected {wanted}. Freeze/connectivity gate not met.')
        require(math.isfinite(norm_sq[i]) and norm_sq[i]>=0,f'{label}: invalid gradient norm')
        gradients[name]={'variable_count':counts[i],'l2_norm':math.sqrt(norm_sq[i]),
                         'numerically_active':bool(norm_sq[i]>0)}
    delta=state_diff(before,state_snapshot(m))
    for name,parts in delta.items():
        if name not in active:
            require(all(v['changed_tensors']==0 for v in parts.values()),f'{label}: frozen {name} weights/buffers/Adam slots changed')
        elif name != 'G':
            require(parts['nontrainable']['changed_tensors']==0,f'{label}: unexpected nontrainable buffer change for {name}')
    if family in ('C','D'):
        require(all(delta[name]['trainable']['changed_tensors']>0 and gradients[name]['numerically_active'] for name in active),f'{label}: no verified active classifier/discriminator update')
    evidence=all(gradients[name]['numerically_active'] and delta[name]['trainable']['changed_tensors']>0 for name in active)
    result={'route':label,'family':family,'loss':loss,'clock':clock+1,
            'actual_loss_gradients':gradients,'state_delta':delta,
            'active_trainable_update_observed':evidence,
            'frozen_weights_buffers_slots_unchanged':True}
    REPORT['routes'].append(result)
    phase(label+' complete')
    return result


def checkpoint_object(m):
    values={key:m[key] for key in ('C','D','G')}
    values.update({f'head_{h:02d}':head for h,head in enumerate(m['heads'])})
    values['optimizer']=m['opt']
    return tf.train.Checkpoint(**values)


def smoke():
    phase('input provenance: train only, before TensorFlow import')
    X,y,source=load_input_verified()
    REPORT['input']=source
    REPORT['input'].update(shape=list(X.shape),X_sha256=sha_array(X),y_sha256=sha_array(y),class_counts=np.bincount(y,minlength=16).tolist())
    phase('NumPy math checks')
    REPORT['math']=math_gate()
    phase('TensorFlow / exactly one GPU')
    REPORT['runtime']=load_runtime()
    phase('build V2 / train-only neighbors / allocate shared Adam slots')
    with tf.device('/GPU:0'):
        m=build_v2_models(X,y,CONFIG['seed'])
    require(len(m['heads']) == 15 and int(m['opt'].iterations.numpy()) == 0,'Wrong initial state')
    REPORT['class_order_original_ids_1based']=(m['order']+1).tolist()
    REPORT['graph_sha256']=[sha_array(g) for g in m['neighbor_graphs']]
    REPORT['model_trainable_variable_counts']={name:len(owner['trainable']) for name,owner in m['owners'].items()}
    rng=np.random.default_rng(912)
    z=rng.normal(size=(2,100)).astype(np.float32)
    condition=np.zeros((2,16),np.float32);condition[:,0]=1
    anchors=np.array([[0],[1]],np.int32)
    fake,REPORT['inference']=inference_gate(m,z,condition,anchors)
    real=X[:2].copy()
    real_label=np.eye(16,dtype=np.float32)[m['y_sorted'][:2]]
    update_gate(m,'C_real',lambda:m['C'].train_on_batch(real,real_label),{'C'},'C')
    update_gate(m,'D_real',lambda:m['D'].train_on_batch([real,real_label],np.ones((2,1),np.float32)),{'D'},'D')
    update_gate(m,'C_fake',lambda:m['C'].train_on_batch(fake,condition),{'C'},'C')
    update_gate(m,'D_fake',lambda:m['D'].train_on_batch([fake,condition],np.zeros((2,1),np.float32)),{'D'},'D')
    for head in range(15):
        condition_h=np.zeros((2,16),np.float32);condition_h[:,head]=1
        update_gate(m,f'G_vs_C_{head}',lambda h=head,c=condition_h:m['vs_c'][h].train_on_batch([z,c,anchors],1-c),{'G',f'head_{head}'},'GC')
        update_gate(m,f'G_vs_D_{head}',lambda h=head,c=condition_h:m['vs_d'][h].train_on_batch([z,c,c,anchors],np.ones((2,1),np.float32)),{'G',f'head_{head}'},'GD')
    require(len(REPORT['routes']) == 34 and int(m['opt'].iterations.numpy()) == 34,'Wrong total update count')
    # Zero gradient in a particular tensor/route is not automatically a code
    # defect. Still require actual G+head trainable updates in both loss families
    # and at least one of GC/GD for every head before claiming integration PASS.
    learning={family:sum(r['active_trainable_update_observed'] for r in REPORT['routes'] if r['family']==family) for family in ('GC','GD')}
    covered_heads=[h for h in range(15) if any(r['route'] in (f'G_vs_C_{h}',f'G_vs_D_{h}') and r['active_trainable_update_observed'] for r in REPORT['routes'])]
    REPORT['learning_evidence']={'verified_routes_by_loss':learning,'heads_with_G_and_head_trainable_updates':covered_heads}
    require(all(n>0 for n in learning.values()) and len(covered_heads)==15,'Gradient/update gate unresolved: insufficient numerically active G/head routes. This alone is not proof the V2 formula is wrong.')
    phase('checkpoint weights / BN / all Adam slots / clock roundtrip')
    before=fingerprint(m)
    # Helper return values only; no retained checkpoint/model references when
    # rebuilding. The temporary checkpoint is removed after roundtrip.
    with tempfile.TemporaryDirectory(prefix='v2_ckpt_') as temp:
        prefix=str(Path(temp)/'checkpoint')
        checkpoint_object(m).write(prefix)
        del m
        keras.backend.clear_session();gc.collect()
        deadline()
        with tf.device('/GPU:0'):
            m=build_v2_models(X,y,CONFIG['seed'])
        checkpoint_object(m).read(prefix).assert_consumed()
        require(int(m['opt'].iterations.numpy()) == 34,'Restored Adam clock mismatch')
        require(fingerprint(m) == before,'Checkpoint did not restore all model/BN/optimizer state')
    REPORT['checkpoint']={'weights_BN_m_v_and_clock_roundtrip':True,'optimizer_iterations':34,
                          'fingerprint':before,'exact_dropout_resume_tested':False,
                          'training_continuation_tested':False}
    del m
    keras.backend.clear_session();gc.collect()
    deadline()
    REPORT.update(status='PASS',pass_=True,phase='complete',elapsed_seconds=round(time.monotonic()-START,3))
    # JSON uses the usual 'pass' key, not Python's reserved keyword.
    REPORT['pass']=REPORT.pop('pass_')
    atomic_json(FOLDER/'smoke_report.json',REPORT)
    print('SMOKE PASS — V2 engineering only; 2 C updates, 34 Adam calls; no accuracy/test.',flush=True)


def main():
    global CONFIG,REPORT,FOLDER,START,np
    CONFIG=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    FOLDER=Path(CONFIG['attempt_dir'])
    START=CONFIG['start_monotonic']
    REPORT={'attempt_id':CONFIG['attempt_id'],'config_sha256':CONFIG['config_sha256'],
            'worker_sha256':CONFIG['worker_sha256'],'status':'RUNNING','pass':False,
            'scope':'single V2 integration smoke; no control-arm study',
            'seed':CONFIG['seed'],'rho':RHO,'k':K,'routes':[],
            'C_updates_planned':2,'Adam_calls_planned':34,'batch_size':2,
            'paired_control_schedule_tested':False,'exact_dropout_resume_tested':False,
            'full_training_allowed':False,'accuracy_computed':False,
            'validation_arrays_loaded':False,'test_arrays_loaded':False,
            'no_local_experiment_required':True}
    try:
        unsigned = {key:value for key,value in CONFIG.items() if key != 'config_sha256'}
        require(hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':')).encode()).hexdigest() == CONFIG['config_sha256'], 'Config integrity check failed')
        require(file_sha(__file__) == CONFIG['worker_sha256'], 'Worker source integrity check failed')
        phase('start worker / import NumPy')
        import numpy as numpy
        np=numpy
        smoke()
    except BaseException as error:
        REPORT.update(status='TIMEOUT' if isinstance(error,TimeoutError) else 'FAILED',
                      **{'pass':False},error_type=type(error).__name__,error=str(error),
                      traceback=traceback.format_exc(),elapsed_seconds=round(time.monotonic()-START,3))
        atomic_json(FOLDER/'smoke_report.json',REPORT)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
