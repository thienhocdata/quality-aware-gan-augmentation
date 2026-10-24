"""Final bounded Kaggle technical gate, never a classification experiment.

Passed MOC2 model, schedule, and consumed training routes are embedded by the
packager. Dense state snapshots and per-route report I/O are omitted from the
benchmark. Identity gradient observation and actual Dropout-mask recording stay.
"""
from __future__ import annotations
import gc
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

np = tf = keras = None
CONFIG = REPORT = FOLDER = START = None
PAIRED_C_DROPOUT = PAIRED_D_DROPOUT = None
GAN_ARMS = ('original_gan','learned_local')
LOCAL_ARMS = ('random_local','learned_local')
ANCHOR_ARMS = ('matched_ros','random_local','learned_local')
BATCH = 100
MAIN_LOOPS = 46
COMPARISON_POLICY = {
    'weights_BN': {'rtol':1e-6,'atol':1e-7},
    'Adam_m_v': {'rtol':1e-6,'atol':1e-12},
    'audit_float': {'rtol':1e-6,'atol':1e-12},
    'integer_and_dropout_control':'bitwise exact',
    'boundary_restore':'all arrays bitwise exact',
    'meaning':'allclose is numerical continuation; exact trajectory only if bitwise_equal',
}

SHAPE = (11, 11, 40, 1)


FLAT = 4840


N_CLASSES = 16


K = 3


RHO = 0.2


RUN_ID = '50e988fae260'


IMPLEMENTATION = '5e15f8214bf273bb889187b96c80407a8000e5e1a3ef30734bcfa86ced275a6f'


EXPECTED_COUNTS = [17,602,341,110,167,301,13,182,8,375,1112,249,100,576,230,39]


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
    tf.config.experimental.enable_op_determinism()
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


def sha_array(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


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
    x = PAIRED_C_DROPOUT(0.5)(x)
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
    x = PAIRED_D_DROPOUT(.5)(x)
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


def tao_head_convex(support, shape):
    """Tương đương denseGenProcessCreate: softmax weights × support matrix."""
    L = keras.layers
    support_flat = support.reshape(len(support), -1).astype(np.float32)
    support_const = tf.constant(support_flat)
    inp = L.Input(shape=(784,))
    weights = L.Dense(len(support), activation="softmax", name="support_weights")(inp)
    flat = L.Lambda(lambda w: tf.matmul(w, support_const), name="convex_sum")(weights)
    out = L.Reshape(shape)(flat)
    return keras.Model(inp, out, name=f"head_{len(support)}")


N_CLASSES = 16


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def arrays_sha(arrays):
    return json_sha([sha_array(np.asarray(a)) for a in arrays])


def dropout_key(seed,loop,route,head=-1):
    payload=json.dumps([int(seed),int(loop),str(route),int(head)],separators=(',',':')).encode()
    return np.frombuffer(hashlib.sha256(payload).digest()[:8],dtype='<i4').copy()


def recorded_dropout_factory(width,owner):
    """Rate-0.5 Dropout with external stateless seed and actual mask recorder.

    Control/audit variables are intentionally separate from frozen network
    weights and Adam state. They are reconstructed from route keys, not treated
    as model parameters. The layer must have no weights of its own.
    """
    with tf.device('/CPU:0'):
        control={'seed':tf.Variable([0,1],dtype=tf.int32,trainable=False),
                 'used_seed':tf.Variable([0,0],dtype=tf.int32,trainable=False),
                 'calls':tf.Variable(0,dtype=tf.int64,trainable=False),
                 'rows':tf.Variable(0,dtype=tf.int32,trainable=False),
                 'mask':tf.Variable(np.zeros((BATCH,width),np.int32),trainable=False)}
    class RecordedDropout(keras.layers.Layer):
        def __init__(self,rate=.5,**kwargs):
            super().__init__(**kwargs)
            require(float(rate)==.5,'Matched runner keeps Dropout rate 0.5')
            self.rate=float(rate)
        def call(self,inputs,training=None):
            def dropped():
                seed=control['seed'].read_value()
                rows=tf.shape(inputs)[0]
                checks=[tf.debugging.assert_less_equal(rows,BATCH),
                        tf.debugging.assert_equal(tf.shape(inputs)[1],width)]
                with tf.control_dependencies([c for c in checks if c is not None]):
                    mask=tf.cast(tf.random.stateless_uniform(tf.shape(inputs),seed=seed,dtype=tf.float32)>=self.rate,tf.int32)
                operations=[control['mask'].assign(tf.pad(mask,[[0,BATCH-rows],[0,0]])),
                            control['rows'].assign(rows),control['used_seed'].assign(seed),
                            control['calls'].assign_add(1)]
                with tf.control_dependencies(operations):
                    return inputs*tf.cast(mask,inputs.dtype)/(1.-self.rate)
            return keras.backend.in_train_phase(dropped,inputs,training=training)
        def get_config(self):
            return {**super().get_config(),'rate':self.rate}
    RecordedDropout.__name__='RecordedDropout_'+owner
    return RecordedDropout,control


def build_models(X,y,arm,seed):
    global PAIRED_C_DROPOUT,PAIRED_D_DROPOUT
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    prepared=build_graph_per_class(X,y)
    context={}
    def observer(pairs):
        require('observe' in context,'Optimizer observation not initialized')
        return context['observe'](pairs)
    opt=keras.optimizers.legacy.Adam(learning_rate=2e-4,beta_1=.5,gradient_transformers=[observer])
    PAIRED_C_DROPOUT,c_control=recorded_dropout_factory(64,'C')
    keras.utils.set_random_seed(seed)
    C=tao_classifier(SHAPE,16)
    C.compile(loss='mean_squared_error',optimizer=opt)
    controls={'C':c_control}
    networks={'C':C}
    heads=[];vs_c=[];vs_d=[]
    if arm in GAN_ARMS:
        PAIRED_D_DROPOUT,d_control=recorded_dropout_factory(80,'D')
        controls['D']=d_control
        keras.utils.set_random_seed(seed+101)
        D=tao_discriminator(SHAPE,16)
        D.compile(loss='mean_squared_error',optimizer=opt)
        keras.utils.set_random_seed(seed+102)
        G=tao_generator(100,16)
        for h,(support,graph) in enumerate(zip(prepared['supports'],prepared['neighbor_graphs'])):
            keras.utils.set_random_seed(seed+1000+h)
            heads.append(tao_head_convex(support,SHAPE) if arm=='original_gan' else tao_head_v2(support,graph))
        C.trainable=False;D.trainable=False
        for h,head in enumerate(heads):
            z=keras.layers.Input(shape=(100,));condition=keras.layers.Input(shape=(16,))
            if arm=='original_gan':
                fake=head(G([z,condition]));inputs=[z,condition]
            else:
                anchor=keras.layers.Input(shape=(1,),dtype='int32')
                fake=head([G([z,condition]),anchor]);inputs=[z,condition,anchor]
            mc=keras.Model(inputs,C(fake),name=f'GC_{h}')
            mc.compile(loss='mean_squared_error',optimizer=opt)
            vs_c.append(mc)
            z2=keras.layers.Input(shape=(100,));cond2=keras.layers.Input(shape=(16,));cond3=keras.layers.Input(shape=(16,))
            if arm=='original_gan':
                fake2=head(G([z2,cond2]));inputs2=[z2,cond2,cond3]
            else:
                anchor2=keras.layers.Input(shape=(1,),dtype='int32')
                fake2=head([G([z2,cond2]),anchor2]);inputs2=[z2,cond2,cond3,anchor2]
            md=keras.Model(inputs2,D([fake2,cond3]),name=f'GD_{h}')
            md.compile(loss='mean_squared_error',optimizer=opt)
            vs_d.append(md)
        C.trainable=True;D.trainable=True
        networks.update(D=D,G=G)
        networks.update({f'head_{h}':head for h,head in enumerate(heads)})
    owners={name:{'trainable':tuple(model.trainable_variables),'nontrainable':tuple(model.non_trainable_variables)} for name,model in networks.items()}
    all_variables=[v for owner in owners.values() for v in owner['trainable']]
    require(all('GPU:0' in v.device for v in all_variables),'Unexpected non-GPU model variables')
    opt._create_all_weights(all_variables)
    require(set(opt.get_slot_names())=={'m','v'},'Unexpected shared Adam slots')
    model_refs={v.ref() for model in networks.values() for v in model.weights}
    for name,control in controls.items():
        require(not any(v.ref() in model_refs for v in control.values()),f'{name} dropout control was accidentally tracked as network weights')
        for layer in networks[name].layers:
            if layer.__class__.__name__.startswith('RecordedDropout'):
                require(not layer.weights,'RecordedDropout must not own trainable/frozen model state')
    audit,observe=make_gradient_audit(owners)
    context['observe']=observe
    return {**networks,'C':C,'heads':heads,'vs_c':vs_c,'vs_d':vs_d,
            'opt':opt,'owners':owners,'audit':audit,'controls':controls,**prepared}


def initial_fingerprints(m):
    def trainable(name):return arrays_sha(v.numpy() for v in m['owners'][name]['trainable'])
    result={'C_trainable_sha256':trainable('C'),
            'C_nontrainable_sha256':arrays_sha(v.numpy() for v in m['owners']['C']['nontrainable']),
            'C_adam_slots_sha256':arrays_sha(m['opt'].get_slot(v,s).numpy() for v in m['owners']['C']['trainable'] for s in ('m','v'))}
    if CONFIG['arm'] in GAN_ARMS:
        result.update(D_trainable_sha256=trainable('D'),G_trainable_sha256=trainable('G'),
                      head_dense_trainable_sha256=arrays_sha(v.numpy() for head in m['heads'] for v in head.trainable_variables))
    return result


def controlled_update(m,model_owner,route,loop,head,fn,active,family):
    control=m['controls'][model_owner]
    seed=dropout_key(CONFIG['seed'],loop,route,head)
    control['seed'].assign(seed)
    calls=int(control['calls'].numpy())
    before=int(m['opt'].iterations.numpy())
    result=update_gate(m,f'loop{loop}/{route}'+(f'_{head}' if head>=0 else ''),fn,active,family)
    require(int(control['calls'].numpy())==calls+1,f'{route}: actual Dropout recorder did not execute once')
    require(np.array_equal(control['used_seed'].numpy(),seed),f'{route}: stale Dropout seed captured by train function')
    rows=int(control['rows'].numpy())
    record={'loop':loop,'route':route,'head':head,'clock_before':before,'clock_after':int(m['opt'].iterations.numpy()),
            'dropout_seed':seed.tolist(),'dropout_mask_sha256':sha_array(control['mask'].numpy()[:rows]),'rows':rows}
    return result,record


def infer_fake(m,item,arm,spec):
    """Return patches and evidence derived from arrays actually used to render."""
    clock=int(m['opt'].iterations.numpy())
    labels=item['fake_onehot'].copy()
    actual_classes=labels.argmax(1).astype(np.int32)
    anchors=item['local_anchors'].copy()
    fake=np.empty((len(labels),*SHAPE),np.float32)
    source_rows=np.empty(len(labels),np.int64)
    neighbors_local=np.empty((len(labels),3),np.int32)
    neighbors_global=np.empty((len(labels),3),np.int64)
    used_anchors=np.empty(len(labels),np.int32)
    used_weights=np.empty((len(labels),3),np.float32)
    features=None
    if arm in GAN_ARMS:
        noise=item['noise_c'].copy()
        features=m['G'].predict_on_batch([noise,labels])
    for h,support in enumerate(m['supports']):
        ii=np.flatnonzero(actual_classes==h)
        if not len(ii):continue
        if arm=='original_gan':
            fake[ii]=m['heads'][h].predict_on_batch(features[ii])
            continue
        local=anchors[ii]
        used_anchors[ii]=local
        inventory=np.asarray(spec['support_row_ids'][h],np.int64)
        source_rows[ii]=inventory[local]
        if arm=='matched_ros':
            fake[ii]=support[local]
        else:
            allowed=m['neighbor_graphs'][h][local]
            neighbors_local[ii]=allowed
            neighbors_global[ii]=inventory[allowed]
            if arm=='random_local':
                weights=item['dirichlet_weights'][ii].copy()
                used_weights[ii]=weights
                fake[ii]=(1-RHO)*support[local]+RHO*np.einsum('bk,bk...->b...',weights,support[allowed])
            else:
                fake[ii]=m['heads'][h].predict_on_batch([features[ii],local[:,None]])
            movement=np.linalg.norm((fake[ii]-support[local]).reshape(len(ii),-1).astype(np.float64),axis=1)
            neighbor_distance=np.linalg.norm((support[allowed]-support[local,None]).reshape(len(ii),3,-1).astype(np.float64),axis=2)
            bound=RHO*neighbor_distance.max(axis=1)
            require(np.all(movement<=bound+2e-5*(1+bound)),'Local mixing violated the fixed rho displacement bound')
    require(np.isfinite(fake).all(),'Nonfinite fake patches')
    require(int(m['opt'].iterations.numpy())==clock,'Fake-slot inference changed optimizer clock')
    evidence={'fake_classes':actual_classes,'fake_onehot':labels}
    if arm in ANCHOR_ARMS:
        evidence.update(local_anchors=used_anchors,support_row_ids=source_rows)
    if arm in LOCAL_ARMS:
        evidence.update(neighbors_local=neighbors_local,neighbors_global=neighbors_global)
    if arm in GAN_ARMS:evidence['noise_c']=noise
    if arm=='random_local':evidence['dirichlet_weights']=used_weights
    return fake,labels,evidence


"""Pure NumPy input schedule for the bounded Indian Pines Moc 2 smoke.

This is a NEW keyed schedule, not a replay of the historical Protocol B GAN.
No TensorFlow, filesystem access, model construction, or optimization occurs here.
The runner must pass the returned arrays to its actual update calls and record
those inputs with ``consumed_trace``; a schedule fixture alone proves no training
route consumed the schedule.
"""

import hashlib
import json
from collections.abc import Mapping

import numpy as np


SCHEMA_VERSION = 1
N_CLASSES = 16
N_HEADS = 15
K_NEIGHBORS = 3

# Numeric domain tags are fixed protocol constants. Do not use Python hash().
_DOMAINS = {
    "real_order": 101,
    "fake_classes": 201,
    "c_anchors": 202,
    "c_noise": 203,
    "d_real_targets": 301,
    "d_fake_targets": 302,
    "local_random_weights": 401,
    "g_noise": 501,
    "g_anchors": 502,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, bool),
             f"{name} must be an integer")
    value = int(value)
    _require(value >= minimum, f"{name} must be >= {minimum}")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def array_sha(value):
    """Schedule-format digest, intentionally distinct from B's array hash."""
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(_canonical(list(value.shape)))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _spec_sha(spec):
    return hashlib.sha256(_canonical({k: v for k, v in spec.items()
                                     if k != "schedule_sha256"})).hexdigest()


def _rng(spec, loop, domain, head=0):
    sequence = np.random.SeedSequence([
        SCHEMA_VERSION, int(spec["seed"]), int(loop), _DOMAINS[domain], int(head)
    ])
    # Pin the bit generator instead of depending on default_rng's future default.
    return np.random.Generator(np.random.PCG64(sequence))


def make_schedule(y_sorted, seed=42, batch_size=100, latent_dim=100,
                  g_points=7, n_heads=15, *, class_order):
    """Return a JSON-serializable schedule specification.

    ``y_sorted`` contains class-rank labels of the actual X_train row order.
    ``class_order`` is the corresponding 16 original 1-based class IDs, e.g.
    ``(prepared['order'] + 1).tolist()``. Support-row inventories are global
    X_train row indices in exactly the order used by ``X[y_sorted == h]``.
    """
    seed = _integer(seed, "seed")
    batch_size = _integer(batch_size, "batch_size", 1)
    latent_dim = _integer(latent_dim, "latent_dim", 1)
    g_points = _integer(g_points, "g_points", 1)
    n_heads = _integer(n_heads, "n_heads", 1)
    _require(n_heads == N_HEADS, "This smoke schedule is for 15 Indian Pines heads")
    labels = np.asarray(y_sorted)
    _require(labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer),
             "y_sorted must be a one-dimensional integer array")
    _require(len(labels) > 0 and np.all((labels >= 0) & (labels < N_CLASSES)),
             "Invalid class-rank labels")
    labels = labels.astype(np.int64, copy=False)
    counts = np.bincount(labels, minlength=N_CLASSES)
    _require(np.all(counts > 0), "All 16 classes must be present")
    _require(np.all(np.diff(counts) >= 0), "y_sorted class counts must be sorted")
    _require(np.all(counts[-1] > counts[:-1]), "A strict single majority is required")
    _require(np.all(counts[:-1] >= K_NEIGHBORS + 1),
             "Every minority head needs at least four supports")
    order = np.asarray(class_order)
    _require(order.shape == (N_CLASSES,) and np.issubdtype(order.dtype, np.integer),
             "class_order must contain 16 original integer class IDs")
    _require(np.array_equal(np.sort(order), np.arange(1, N_CLASSES + 1)),
             "class_order must be a permutation of original IDs 1..16")
    spec = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fresh_keyed_integration_schedule_not_historical_B_replay",
        "numpy_version": np.__version__,
        "bit_generator": "PCG64",
        "seed": seed,
        "samples": int(len(labels)),
        "batch_size": batch_size,
        "n_batches": int((len(labels) + batch_size - 1) // batch_size),
        "tail_batch_size": int((len(labels) - 1) % batch_size + 1),
        "latent_dim": latent_dim,
        "g_points": g_points,
        "n_classes": N_CLASSES,
        "n_heads": n_heads,
        "class_order_original_ids_1based": order.astype(np.int64).tolist(),
        "class_counts_sorted": counts.astype(np.int64).tolist(),
        "y_sorted": labels.tolist(),
        "support_row_ids": [np.flatnonzero(labels == h).tolist()
                            for h in range(N_CLASSES)],
        "domain_tags": dict(_DOMAINS),
        "real_order_policy": "one permutation reused at every wrap; no reshuffle",
        "g_pairing_policy": "GC and GD use the same per-head noise and anchors",
        "fake_class_policy": "minority probability proportional to majority-count deficit",
    }
    spec["real_order"] = _rng(spec, 0, "real_order").permutation(len(labels)).tolist()
    spec["schedule_sha256"] = _spec_sha(spec)
    validate_schedule(spec)
    return spec


def validate_schedule(spec):
    """Check a deserialized specification without generating or training a model."""
    _require(spec["schema_version"] == SCHEMA_VERSION, "Wrong schedule schema")
    _require(spec["numpy_version"] == np.__version__,
             "NumPy version changed; do not silently claim exact schedule replay")
    _require(spec["bit_generator"] == "PCG64" and spec["domain_tags"] == _DOMAINS,
             "RNG protocol changed")
    _require(spec["schedule_sha256"] == _spec_sha(spec), "Schedule digest mismatch")
    n = _integer(spec["samples"], "samples", 1)
    bs = _integer(spec["batch_size"], "batch_size", 1)
    _integer(spec["seed"], "seed")
    _integer(spec["latent_dim"], "latent_dim", 1)
    _integer(spec["g_points"], "g_points", 1)
    _require(spec["n_heads"] == 15 and spec["n_classes"] == 16,
             "Wrong class/head layout")
    y = np.asarray(spec["y_sorted"], dtype=np.int64)
    counts = np.asarray(spec["class_counts_sorted"], dtype=np.int64)
    _require(y.shape == (n,) and np.all((y >= 0) & (y < 16)), "Wrong label inventory")
    _require(counts.shape == (16,) and np.array_equal(np.bincount(y, minlength=16), counts),
             "Wrong support counts")
    _require(np.all(counts[:-1] >= 4) and np.all(np.diff(counts) >= 0)
             and np.all(counts[-1] > counts[:-1]), "Wrong sorted minority inventory")
    order = np.asarray(spec["real_order"], dtype=np.int64)
    _require(order.shape == (n,) and np.array_equal(np.sort(order), np.arange(n)),
             "real_order is not a complete permutation")
    _require(spec["n_batches"] == (n + bs - 1) // bs
             and spec["tail_batch_size"] == (n - 1) % bs + 1, "Wrong batch partition")
    original_ids = np.asarray(spec["class_order_original_ids_1based"])
    _require(original_ids.shape == (16,)
             and np.array_equal(np.sort(original_ids), np.arange(1, 17)),
             "Wrong original class order")
    _require(len(spec["support_row_ids"]) == 16, "Wrong support inventory length")
    for head in range(16):
        _require(np.array_equal(np.asarray(spec["support_row_ids"][head]),
                                np.flatnonzero(y == head)),
                 f"Support row order mismatch for class rank {head}")
    return True


def frame(spec, loop):
    """Generate the complete input frame for a zero-based real-batch loop.

    The arrays are generated by independent keyed domains; calls or extra draws
    in a control arm cannot advance another domain. ``g[h]`` must be used for
    BOTH GC_h and GD_h. This is a schedule contract, not evidence of consumption.
    """
    validate_schedule(spec)
    loop = _integer(loop, "loop")
    counts = np.asarray(spec["class_counts_sorted"], dtype=np.int64)
    y = np.asarray(spec["y_sorted"], dtype=np.int64)
    inventories = [np.asarray(rows, dtype=np.int64) for rows in spec["support_row_ids"]]
    batch_index = loop % spec["n_batches"]
    start = batch_index * spec["batch_size"]
    real_ids = np.asarray(spec["real_order"][start:start + spec["batch_size"]], dtype=np.int64)
    size, latent, gp = len(real_ids), spec["latent_dim"], spec["g_points"]
    deficits = counts[-1] - counts[:-1]
    cdf = np.cumsum(deficits / deficits.sum())
    cdf[-1] = 1.0
    fake_classes = np.searchsorted(cdf, _rng(spec, loop, "fake_classes").random(size),
                                   side="right").astype(np.int32)
    anchors = np.empty(size, dtype=np.int32)
    source_rows = np.empty(size, dtype=np.int64)
    for head in range(15):
        selected = np.flatnonzero(fake_classes == head)
        if len(selected):
            anchors[selected] = _rng(spec, loop, "c_anchors", head).integers(
                0, int(counts[head]), size=len(selected), dtype=np.int32)
            source_rows[selected] = inventories[head][anchors[selected]]
    fake_onehot = np.eye(16, dtype=np.float32)[fake_classes]
    g_inputs = []
    for head in range(15):
        head_anchors = _rng(spec, loop, "g_anchors", head).integers(
            0, int(counts[head]), size=(gp, 1), dtype=np.int32)
        condition = np.zeros((gp, 16), dtype=np.float32)
        condition[:, head] = 1.0
        g_inputs.append({
            "head": head,
            "noise": _rng(spec, loop, "g_noise", head).normal(size=(gp, latent)).astype(np.float32),
            "anchors": head_anchors,
            "support_row_ids": inventories[head][head_anchors[:, 0]],
            "condition": condition,
            "classifier_targets": 1.0 - condition,
            "discriminator_targets": np.ones((gp, 1), dtype=np.float32),
        })
    return {
        "loop": loop,
        "real_epoch": loop // spec["n_batches"],
        "batch_index": batch_index,
        "batch_size": size,
        "real_ids": real_ids,
        "real_classes": y[real_ids].astype(np.int32),
        "real_onehot": np.eye(16, dtype=np.float32)[y[real_ids]],
        "fake_classes": fake_classes,
        "fake_onehot": fake_onehot,
        "local_anchors": anchors,
        "support_row_ids": source_rows,
        "noise_c": _rng(spec, loop, "c_noise").normal(size=(size, latent)).astype(np.float32),
        "d_real_targets": (1.0 - _rng(spec, loop, "d_real_targets").uniform(
            0.0, 0.1, size=(size, 1))).astype(np.float32),
        "d_fake_targets": _rng(spec, loop, "d_fake_targets").uniform(
            0.0, 0.1, size=(size, 1)).astype(np.float32),
        "dirichlet_weights": _rng(spec, loop, "local_random_weights").dirichlet(
            np.ones(K_NEIGHBORS), size=size).astype(np.float32),
        "g": g_inputs,
    }


def _array_fields(value, prefix=""):
    result = {}
    if isinstance(value, np.ndarray):
        result[prefix] = value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            result.update(_array_fields(child, f"{prefix}/{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.update(_array_fields(child, f"{prefix}/{index}" if prefix else str(index)))
    return result


def _field_records(fields):
    return {name: {"dtype": value.dtype.str, "shape": list(value.shape),
                   "sha256": array_sha(value)}
            for name, value in sorted(fields.items())}


def frame_digest(value):
    """Digest a frame's input arrays plus its loop/batch metadata."""
    payload = {"loop": int(value["loop"]), "real_epoch": int(value["real_epoch"]),
               "batch_index": int(value["batch_index"]), "batch_size": int(value["batch_size"]),
               "arrays": _field_records(_array_fields(value))}
    return hashlib.sha256(_canonical(payload)).hexdigest()


def consumed_trace(spec, loop, consumed, *, required_fields=None, label=""):
    """Validate explicitly supplied ACTUAL call inputs against the keyed frame.

    ``consumed`` may be nested like a frame or use flat paths, e.g.
    ``{'real_ids': ids_used, 'fake_classes': labels_used, 'g/0/noise': z_used}``.
    Values must be NumPy arrays of their original dtype, before TF conversion.
    ``required_fields`` should be supplied by each real route's adapter. This
    helper cannot by itself prove those arrays were passed to a model; invoke
    it where the runner actually performs that route, and retain route evidence.
    Different arms can compare per-field hashes for their shared inputs; an
    arm-specific label is intentionally excluded from the digest.
    """
    expected = _array_fields(frame(spec, loop))
    observed = _array_fields(consumed)
    _require(observed, "A consumed trace must contain actual input arrays")
    required = set(observed) if required_fields is None else set(required_fields)
    _require(required and required.issubset(observed), "Required consumed inputs are missing")
    _require(set(observed).issubset(expected), "Unknown consumed schedule fields")
    for name, value in observed.items():
        target = expected[name]
        _require(value.dtype == target.dtype and value.shape == target.shape,
                 f"Consumed dtype/shape mismatch: {name}")
        _require(np.array_equal(value, target), f"Actual input differs from schedule: {name}")
    records = _field_records(observed)
    payload = {"schedule_sha256": spec["schedule_sha256"], "loop": int(loop), "arrays": records}
    return {"label": str(label), **payload,
            "verified_fields": sorted(observed),
            "required_fields": sorted(required),
            "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
            "scope": "only explicitly supplied consumed inputs; model/update evidence is separate"}


def metadata_gate(spec, loops=90):
    """Bounded CPU metadata checks across two 45-batch IP epochs by default.

    This checks the schedule implementation, not training, accuracy, optimizer
    state, or an arm's actual consumption. The real runner must record those.
    """
    validate_schedule(spec)
    loops = _integer(loops, "loops", 1)
    _require(loops >= 2 * spec["n_batches"],
             "Metadata gate must cover at least two complete real-batch wraps")
    expected_counts = np.asarray(spec["class_counts_sorted"], dtype=np.int64)
    if spec["samples"] == 4422 and spec["batch_size"] == 100:
        _require(spec["n_batches"] == 45 and spec["tail_batch_size"] == 22,
                 "Indian Pines B development tail must be 22 at batch100")
    # A JSON roundtrip is part of the specification contract.
    restored = json.loads(json.dumps(spec, allow_nan=False))
    validate_schedule(restored)
    nuisance = np.random.default_rng(99991)
    digests, tails, real_rows, fake_counts = [], [], [], np.zeros(15, dtype=np.int64)
    for loop in range(loops):
        item = frame(spec, loop)
        digest = frame_digest(item)
        nuisance.normal(size=10000)  # unrelated arm/private draws
        _require(frame_digest(frame(restored, loop)) == digest,
                 "Unrelated RNG activity changed common frame")
        size = item["batch_size"]
        _require(item["real_ids"].shape == (size,) and item["noise_c"].shape == (size, spec["latent_dim"]),
                 "Wrong common input shape")
        _require(np.all((item["fake_classes"] >= 0) & (item["fake_classes"] < 15)),
                 "Fake schedule sampled majority/out-of-range class")
        _require(np.all(item["local_anchors"] >= 0)
                 and np.all(item["local_anchors"] < expected_counts[item["fake_classes"]]),
                 "Head-local C anchors are out of range")
        actual_labels = np.asarray(spec["y_sorted"])[item["support_row_ids"]]
        _require(np.array_equal(actual_labels, item["fake_classes"]),
                 "Source-row IDs do not belong to fake classes")
        _require(item["dirichlet_weights"].shape == (size, 3)
                 and np.all(item["dirichlet_weights"] >= 0)
                 and np.allclose(item["dirichlet_weights"].sum(axis=1), 1., rtol=0, atol=2e-7),
                 "Wrong random-local weights")
        _require(np.all((item["d_real_targets"] >= .9) & (item["d_real_targets"] <= 1.))
                 and np.all((item["d_fake_targets"] >= 0.) & (item["d_fake_targets"] <= .1)),
                 "Wrong D label smoothing")
        _require(len(item["g"]) == 15, "Missing per-head G inputs")
        for head, values in enumerate(item["g"]):
            _require(values["noise"].shape == (spec["g_points"], spec["latent_dim"])
                     and values["anchors"].shape == (spec["g_points"], 1),
                     "Wrong G input shape")
            _require(np.all((values["anchors"] >= 0) & (values["anchors"] < expected_counts[head])),
                     "Head-local G anchor is out of range")
            _require(np.all(np.asarray(spec["y_sorted"])[values["support_row_ids"]] == head),
                     "G source-row inventory is wrong")
        if item["batch_index"] == spec["n_batches"] - 1:
            tails.append({"loop": loop, "size": size})
            _require(size == spec["tail_batch_size"], "Wrong real-batch tail")
        real_rows.append(item["real_ids"])
        fake_counts += np.bincount(item["fake_classes"], minlength=15)
        digests.append(digest)
    all_real = np.concatenate(real_rows)
    for epoch in range(loops // spec["n_batches"]):
        first = epoch * spec["samples"]
        _require(np.array_equal(all_real[first:first + spec["samples"]], spec["real_order"]),
                 "Real order changed or missed samples at wrap")
    return {
        "pass": True,
        "scope": "pure NumPy metadata only; no model or actual training-route evidence",
        "schema_version": SCHEMA_VERSION,
        "schedule_sha256": spec["schedule_sha256"],
        "seed": spec["seed"],
        "numpy_version": np.__version__,
        "class_order_original_ids_1based": spec["class_order_original_ids_1based"],
        "samples": spec["samples"], "n_batches": spec["n_batches"],
        "tail_batch_size": spec["tail_batch_size"], "loops_checked": loops,
        "tail_loops": tails, "real_presentations_checked": int(len(all_real)),
        "fake_presentations_by_head": fake_counts.tolist(),
        "frame_trace_sha256": hashlib.sha256(_canonical(digests)).hexdigest(),
        "json_roundtrip_checked": True,
        "unrelated_rng_invariance_checked": True,
        "historical_B_replay_claimed": False,
        "model_training_performed": False,
    }



def phase(name):
    deadline()
    REPORT['phase']=name
    REPORT['elapsed_seconds']=round(time.monotonic()-START,3)
    target='main_report.json' if CONFIG['execution_mode']=='main' else 'final_report.json'
    atomic_json(FOLDER/target,REPORT)
    print(f"[{REPORT['elapsed_seconds']:.1f}s] {CONFIG['arm']}/{CONFIG['execution_mode']}: {name}",flush=True)


def update_gate(m,label,fn,active,family):
    """Lean actual-loss audit, not a repeated MOC2 frozen-state snapshot gate."""
    deadline()
    clock=int(m['opt'].iterations.numpy())
    calls=int(m['audit']['calls'].numpy())
    value=fn()  # train_on_batch materializes the loss, synchronizing GPU work.
    loss=float(np.asarray(value).reshape(-1)[0])
    require(np.isfinite(loss),f'{label}: nonfinite loss')
    require(int(m['opt'].iterations.numpy())==clock+1,f'{label}: wrong clock')
    require(int(m['audit']['calls'].numpy())==calls+1,f'{label}: wrong gradient observer count')
    counts=m['audit']['counts'].numpy()
    norms=m['audit']['norm_sq'].numpy()
    for i,(name,owner) in enumerate(m['owners'].items()):
        wanted=len(owner['trainable']) if name in active else 0
        require(int(counts[i])==wanted,f'{label}: unexpected gradient ownership {name}')
        require(np.isfinite(norms[i]) and norms[i]>=0,f'{label}: invalid gradient norm {name}')
    REPORT['route_counts'][family]=REPORT['route_counts'].get(family,0)+1
    return {'loss':loss,'clock':clock+1}


def skip_iterations(m,amount,label):
    """Iteration-only assignment; no zero-gradient optimizer application."""
    deadline()
    calls=int(m['audit']['calls'].numpy())
    clock=int(m['opt'].iterations.numpy())
    m['opt'].iterations.assign_add(amount)
    require(int(m['opt'].iterations.numpy())==clock+amount,'Wrong iteration skip')
    require(int(m['audit']['calls'].numpy())==calls,'Skip called optimizer')


def new_ledger():
    return {'attempt_id':CONFIG['attempt_id'],'arm':CONFIG['arm'],
            'common_c':[],'anchors':[],'local_geometry':[],'common_gan':[],
            'private_G_anchors':[],'random_weights':[]}


def run_loop(m,X,spec,loop,ledger):
    arm=CONFIG['arm']
    graph_hashes=[sha_array(g) for g in m['neighbor_graphs']]
    old_anchors=len(REPORT['G_anchor_consumption'])
    old_weights=len(REPORT['random_weight_consumption'])
    deadline()
    item=frame(spec,loop)
    t=34*loop
    require(int(m['opt'].iterations.numpy())==t,'Wrong loop start clock')
    ids=item['real_ids'].copy();real=X[ids].copy();real_labels=item['real_onehot'].copy()
    consumed=consumed_trace(spec,loop,{'real_ids':ids,'real_onehot':real_labels},required_fields=['real_ids','real_onehot'],label='C_real')
    _,c_record=controlled_update(m,'C','C_real',loop,-1,lambda:m['C'].train_on_batch(real,real_labels),{'C'},'C')
    c_record.update(input_real_sha256=sha_array(real),labels_sha256=sha_array(real_labels),schedule=consumed)
    c_record['real_train_row_ids']=ids.tolist()
    ledger['common_c'].append(c_record)
    if arm in GAN_ARMS:
        targets=item['d_real_targets'].copy()
        consumed=consumed_trace(spec,loop,{'real_ids':ids,'real_onehot':real_labels,'d_real_targets':targets},required_fields=['real_ids','real_onehot','d_real_targets'],label='D_real')
        _,record=controlled_update(m,'D','D_real',loop,-1,lambda:m['D'].train_on_batch([real,real_labels],targets),{'D'},'D')
        record['schedule']=consumed;ledger['common_gan'].append(record)
    else:
        skip_iterations(m,1,f'loop{loop}/skip_D_real')
    fake,fake_labels,evidence=infer_fake(m,item,arm,spec)
    REPORT['fake_patch_hashes'].append(sha_array(fake))
    if arm in GAN_ARMS:
        consumed=consumed_trace(spec,loop,{'noise_c':evidence['noise_c'],'fake_onehot':fake_labels},required_fields=['noise_c','fake_onehot'],label='G_fake_inference')
        ledger['common_gan'].append({'loop':loop,'route':'G_fake_inference','clock':t+2,'schedule':consumed})
    if arm in ANCHOR_ARMS:
        consumed=consumed_trace(spec,loop,{key:evidence[key] for key in ('fake_classes','local_anchors','support_row_ids')},required_fields=['fake_classes','local_anchors','support_row_ids'],label='actual_C_fake_anchors')
        ledger['anchors'].append({'schedule':consumed,'local_anchor_ids':evidence['local_anchors'].tolist(),
                                  'global_train_row_ids':evidence['support_row_ids'].tolist(),
                                  'class_ranks':evidence['fake_classes'].tolist()})
    if arm in LOCAL_ARMS:
        ledger['local_geometry'].append({'loop':loop,'k':K,'rho':RHO,'graph_hashes':graph_hashes,
                                        'neighbors_local_sha256':sha_array(evidence['neighbors_local']),
                                        'neighbors_global_sha256':sha_array(evidence['neighbors_global']),
                                        'neighbors_local_ids':evidence['neighbors_local'].tolist(),
                                        'neighbors_global_train_row_ids':evidence['neighbors_global'].tolist()})
    if arm=='random_local':
        REPORT['random_weight_consumption'].append(consumed_trace(spec,loop,{'dirichlet_weights':evidence['dirichlet_weights']},required_fields=['dirichlet_weights'],label='actual_random_local_weights'))
    consumed=consumed_trace(spec,loop,{'fake_classes':evidence['fake_classes'],'fake_onehot':fake_labels},required_fields=['fake_classes','fake_onehot'],label='C_fake')
    _,c_record=controlled_update(m,'C','C_fake',loop,-1,lambda:m['C'].train_on_batch(fake,fake_labels),{'C'},'C')
    c_record.update(labels_sha256=sha_array(fake_labels),schedule=consumed)
    c_record['fake_class_ranks']=evidence['fake_classes'].tolist()
    ledger['common_c'].append(c_record)
    if arm in GAN_ARMS:
        targets=item['d_fake_targets'].copy()
        consumed=consumed_trace(spec,loop,{'fake_onehot':fake_labels,'d_fake_targets':targets},required_fields=['fake_onehot','d_fake_targets'],label='D_fake')
        _,record=controlled_update(m,'D','D_fake',loop,-1,lambda:m['D'].train_on_batch([fake,fake_labels],targets),{'D'},'D')
        record['schedule']=consumed;ledger['common_gan'].append(record)
        for h,g_inputs in enumerate(item['g']):
            noise=g_inputs['noise'].copy();cond=g_inputs['condition'].copy();anchor=g_inputs['anchors'].copy()
            gc_target=g_inputs['classifier_targets'].copy();gd_target=g_inputs['discriminator_targets'].copy()
            gc_args=[noise,cond] if arm=='original_gan' else [noise,cond,anchor]
            gd_args=[noise,cond,cond] if arm=='original_gan' else [noise,cond,cond,anchor]
            if arm=='learned_local':
                REPORT['G_anchor_consumption'].append(consumed_trace(spec,loop,{f'g/{h}/anchors':gc_args[-1]},required_fields=[f'g/{h}/anchors'],label=f'GC_GD_anchor_{h}'))
                require(np.array_equal(gc_args[-1],gd_args[-1]),'GC/GD did not consume the same private anchor IDs')
            for route,args,target,family,model in (('GC',gc_args,gc_target,'GC',m['vs_c'][h]),('GD',gd_args,gd_target,'GD',m['vs_d'][h])):
                fields={f'g/{h}/noise':args[0],f'g/{h}/condition':args[1],f'g/{h}/'+('classifier_targets' if route=='GC' else 'discriminator_targets'):target}
                consumed=consumed_trace(spec,loop,fields,required_fields=list(fields),label=f'{route}_{h}')
                _,record=controlled_update(m,'C' if route=='GC' else 'D',route,loop,h,lambda model=model,args=args,target=target:model.train_on_batch(args,target),{'G',f'head_{h}'},family)
                if route=='GD':
                    require(np.array_equal(args[1],args[2]),'G-vs-D class conditions differ')
                    record['D_condition_sha256']=sha_array(args[2])
                record['schedule']=consumed;ledger['common_gan'].append(record)
    else:
        skip_iterations(m,31,f'loop{loop}/skip_D_fake_and_all_G')
    require(int(m['opt'].iterations.numpy())==t+34,'Wrong loop-end shared Adam clock')
    # Ledger is saved outside timed loops.
    ledger['private_G_anchors'].extend(REPORT['G_anchor_consumption'][old_anchors:])
    ledger['random_weights'].extend(REPORT['random_weight_consumption'][old_weights:])



def trace_summary(ledger,loops):
    arm=CONFIG['arm']
    return {'common_C':json_sha(ledger['common_c']),
            'anchors_C':json_sha(ledger['anchors']) if arm in ANCHOR_ARMS else None,
            'local_geometry':json_sha(ledger['local_geometry']) if arm in LOCAL_ARMS else None,
            'GAN_private':json_sha(ledger['common_gan']) if arm in GAN_ARMS else None,
            'consumed_loops':loops}


def state_variables(m):
    """Stable semantic keys, independent of Keras-generated variable names."""
    result={'optimizer/iterations':m['opt'].iterations}
    for name,owner in m['owners'].items():
        for part in ('trainable','nontrainable'):
            for i,variable in enumerate(owner[part]):
                result[f'{part}/{name}/{i}']=variable
                if part=='trainable':
                    for slot in ('m','v'):
                        result[f'{slot}/{name}/{i}']=m['opt'].get_slot(variable,slot)
    for owner,control in m['controls'].items():
        for key,variable in control.items():result[f'control/{owner}/{key}']=variable
    for key,variable in m['audit'].items():result[f'audit/{key}']=variable
    for key in ('learning_rate','beta_1','beta_2','decay'):
        variable=m['opt']._get_hyper(key)
        if hasattr(variable,'numpy'):result[f'optimizer_hyper/{key}']=variable
    require(all(v is not None for v in result.values()),'Missing state variable')
    return result


def state_hashes(m):
    return {key:sha_array(variable.numpy()) for key,variable in state_variables(m).items()}


def checkpoint_object(m):
    values={'C':m['C'],'optimizer':m['opt'],
            'audit':tf.train.Checkpoint(**m['audit']),
            'dropout_controls':tf.train.Checkpoint(**{
                name:tf.train.Checkpoint(**control) for name,control in m['controls'].items()})}
    if CONFIG['arm'] in GAN_ARMS:
        values.update(D=m['D'],G=m['G'])
        values.update({f'head_{h:02d}':head for h,head in enumerate(m['heads'])})
    return tf.train.Checkpoint(**values)


def fixed_contract(m,spec):
    # Supports/neighbor constants are NOT trainable checkpoint variables.
    contract={'source_sha256':CONFIG['worker_sha256'],
            'schedule_sha256':json_sha(spec),'input_X_sha256':REPORT['input']['X_sha256'],
            'input_y_sha256':REPORT['input']['y_sha256'],
            'order_sha256':sha_array(m['order']),
            'support_sha256':[sha_array(s) for s in m['supports']],
            'neighbor_sha256':[sha_array(g) for g in m['neighbor_graphs']],
            'runtime':{k:REPORT['runtime'][k] for k in ('tensorflow','tf_keras','numpy','op_determinism_enabled','gpu_details')},
            'rho':RHO,'k':K,'batch_size':BATCH,'seed':CONFIG['seed'],
            'next_loop':45,'checkpoint_C_updates':90,'checkpoint_clock':1530}
    # GPU compute_capability is a tuple at runtime, but a list after JSON save.
    return json.loads(json.dumps(contract,sort_keys=True,allow_nan=False))


def capture_state(path,m):
    # Raw arrays are temporary continuation evidence, excluded from report ZIP.
    arrays={key:variable.numpy() for key,variable in state_variables(m).items()}
    require(all(np.isfinite(a).all() for a in arrays.values()),'Nonfinite saved state')
    np.savez(path,**arrays)
    return {key:sha_array(a) for key,a in arrays.items()}


def compare_state(path,m):
    """Compare to the saved uninterrupted reference; preset, part-specific tolerances."""
    variables=state_variables(m)
    groups={}
    with np.load(path,allow_pickle=False) as saved:
        require(set(saved.files)==set(variables),'Reference variable inventory changed')
        for key,var in variables.items():
            a=saved[key];b=var.numpy()
            require(a.shape==b.shape and a.dtype==b.dtype,f'State signature differs: {key}')
            require(np.isfinite(a).all() and np.isfinite(b).all(),f'Nonfinite reference/replay {key}')
            part=key.split('/')[0]
            if part in ('m','v'):policy=COMPARISON_POLICY['Adam_m_v']
            elif part=='audit' and np.issubdtype(a.dtype,np.floating):policy=COMPARISON_POLICY['audit_float']
            else:policy=COMPARISON_POLICY['weights_BN']
            exact=bool(np.array_equal(a,b))
            integer_or_control=(not np.issubdtype(a.dtype,np.floating)) or part=='control'
            close=exact if integer_or_control else bool(np.allclose(a,b,**policy))
            delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
            absolute=float(delta.max(initial=0.))
            relative=float((delta/np.maximum(np.maximum(np.abs(a.astype(np.float64)),np.abs(b.astype(np.float64))),1e-30)).max(initial=0.))
            item=groups.setdefault(part,{'tensors':0,'bitwise_equal':True,'allclose':True,
                                         'max_abs_delta':0.,'max_rel_delta':0.,'nonmatching_tensor_keys':[]})
            item['tensors']+=1;item['bitwise_equal']&=exact;item['allclose']&=close
            item['max_abs_delta']=max(item['max_abs_delta'],absolute)
            item['max_rel_delta']=max(item['max_rel_delta'],relative)
            if not close:item['nonmatching_tensor_keys'].append(key)
    return {'groups':groups,'bitwise_equal':all(g['bitwise_equal'] for g in groups.values()),
            'numeric_allclose':all(g['allclose'] for g in groups.values()),
            'comparison_policy':COMPARISON_POLICY}


def tail_gate(ledger):
    rows=[r for r in ledger['common_c'] if r['loop']==44]
    require([r['route'] for r in rows]==['C_real','C_fake'] and all(r['rows']==22 for r in rows),'Actual tail C routes did not consume 22')
    require([(r['clock_before'],r['clock_after']) for r in rows]==[(1496,1497),(1498,1499)],'Tail C clocks wrong')
    private=[r for r in ledger['common_gan'] if r['loop']==44 and r['route'] in ('D_real','D_fake','GC','GD')]
    if CONFIG['arm'] in GAN_ARMS:
        require(len(private)==32,'Missing actual GAN tail routes')
        require(all(r['rows']==(22 if r['route'].startswith('D_') else 7) for r in private),'Actual tail D/G batch sizes wrong')
    return {'pass':True,'loop':44,'C_routes':rows,'private_route_rows':[
        {'route':r['route'],'head':r['head'],'rows':r['rows'],'clock_after':r['clock_after']} for r in private],
        'C_updates_after_tail':90,'clock_after_tail':1530}


def benchmark_summary(timings,memory,setup_seconds):
    seconds=np.asarray([r['seconds'] for r in timings if 1<=r['loop']<=43],np.float64)
    require(len(seconds)==43 and np.isfinite(seconds).all() and np.all(seconds>0),'Missing full-batch timings')
    tail=next(r['seconds'] for r in timings if r['loop']==44)
    return {'steady_full_batch_loops':43,'full_batch_size':100,'tail_loop':44,'tail_batch_size':22,
            'mean_seconds_per_loop':float(seconds.mean()),'p50_seconds_per_loop':float(np.median(seconds)),
            'q25_seconds_per_loop':float(np.quantile(seconds,.25)),'q75_seconds_per_loop':float(np.quantile(seconds,.75)),
            'min_seconds_per_loop':float(seconds.min()),'max_seconds_per_loop':float(seconds.max()),
            'tail_seconds':tail,'warmup_loop0_seconds':timings[0]['seconds'],
            'setup_before_loop0_seconds':setup_seconds,
            'tf_allocator_peak_bytes':int(memory['peak']),'tf_allocator_current_bytes':int(memory['current']),
            'memory_scope':'TF allocator GPU:0; reset after loop0, sampled after44 BEFORE checkpoint/reference; not total VRAM',
            'timing_scope':'frame/render + train_on_batch + gradient observer + actual mask/input ledger; no per-route full-state snapshot or report disk I/O; not bare-model throughput',
            'early_subbatch_retracing_may_remain':True,'posthoc_outlier_filtering':False,
            'checkpoint_validation_test_cost_measured':False,'loop_timings':timings}


def prepare():
    phase('verify original B provenance')
    X,y,source=load_input_verified()
    require(source['real_B_cache_verified'],'Original B cache only')
    REPORT['input']={**source,'shape':list(X.shape),'X_sha256':sha_array(X),'y_sha256':sha_array(y)}
    phase('one GPU / deterministic ops')
    REPORT['runtime']=load_runtime()
    REPORT['runtime'].update(numpy=np.__version__,op_determinism_enabled=True)
    phase('build same MOC2 components and train-only neighbor graph')
    with tf.device('/GPU:0'):m=build_models(X,y,CONFIG['arm'],CONFIG['seed'])
    REPORT['initialization']=initial_fingerprints(m)
    spec=make_schedule(m['y_sorted'],seed=CONFIG['seed'],batch_size=BATCH,latent_dim=100,g_points=7,n_heads=15,class_order=(m['order']+1).tolist())
    require(spec['n_batches']==45 and spec['tail_batch_size']==22,'Wrong real B batch partition')
    return X,y,m,spec


def run_main():
    setup=time.monotonic()
    X,y,m,spec=prepare()
    setup_seconds=time.monotonic()-setup
    atomic_json(FOLDER/'schedule_spec.json',spec)
    ledger=new_ledger();timings=[]
    for loop in range(45):
        phase(f'actual loop {loop}/44; batch {frame(spec,loop)["batch_size"]}')
        start=time.perf_counter()
        run_loop(m,X,spec,loop,ledger)
        # Materialized train_on_batch loss + these reads synchronize updates.
        require(int(m['opt'].iterations.numpy())==34*(loop+1),'Unsynchronized loop clock')
        timings.append({'loop':loop,'batch_size':frame(spec,loop)['batch_size'],
                        'seconds':time.perf_counter()-start})
        if loop==0:tf.config.experimental.reset_memory_stats('GPU:0')
    memory=tf.config.experimental.get_memory_info('GPU:0')
    REPORT['tail']=tail_gate(ledger)
    REPORT['benchmark']=benchmark_summary(timings,memory,setup_seconds)
    REPORT['benchmark']['training_phase_elapsed_seconds']=sum(r['seconds'] for r in timings)
    recovery=FOLDER/'_recovery'
    recovery.mkdir(exist_ok=False)
    phase('save boundary BEFORE uninterrupted reference: next45 / C90 / clock1530')
    boundary={'contract':fixed_contract(m,spec),'state_hashes':state_hashes(m),
              'main_config_sha256':CONFIG['config_sha256'],'ledger_sha256':json_sha(ledger),
              'optimizer_calls':int(m['audit']['calls'].numpy()),
              'dropout_policy':'stateless keys (seed,loop,route,head), actual external recorders checkpointed; no general stateful RNG claim'}
    atomic_json(recovery/'boundary_ledger.json',ledger)
    checkpoint_object(m).write(str(recovery/'checkpoint'))
    atomic_json(recovery/'boundary.json',boundary)
    phase('uninterrupted reference loop45, first batch after wrap')
    reference=new_ledger()
    run_loop(m,X,spec,45,reference)
    for key in ('common_c','anchors','local_geometry','common_gan','private_G_anchors','random_weights'):
        ledger[key].extend(reference[key])
    atomic_json(FOLDER/'consumed_trace.json',ledger)
    atomic_json(recovery/'reference_ledger.json',reference)
    reference_state=capture_state(recovery/'reference_state.npz',m)
    atomic_json(recovery/'reference_state_hashes.json',reference_state)
    applications=1564 if CONFIG['arm'] in GAN_ARMS else 92
    require(int(m['audit']['calls'].numpy())==applications and int(m['opt'].iterations.numpy())==1564,'Wrong main counts')
    REPORT['trace']=trace_summary(ledger,46)
    REPORT['clocks']={'main_C_updates':92,'main_clock':1564,'main_optimizer_calls':applications,
                      'checkpoint_C_updates':90,'checkpoint_clock':1530,'next_loop':45}
    REPORT['boundary']={'contract_sha256':json_sha(boundary['contract']),
                        'all_state_fingerprint':json_sha(boundary['state_hashes']),
                        'audit_and_external_dropout_controls_in_checkpoint':True}
    REPORT.update(status='MAIN_COMPLETE',**{'pass':False},phase='main saved; fresh process continuation still required',
                  elapsed_seconds=round(time.monotonic()-START,3))
    atomic_json(FOLDER/'main_report.json',REPORT)
    print('MAIN_COMPLETE:',CONFIG['arm'],'| not PASS; launch fresh resume process',flush=True)


def run_resume():
    main_config=json.loads((FOLDER/'main_config.json').read_text(encoding='utf-8'))
    main_report=json.loads((FOLDER/'main_report.json').read_text(encoding='utf-8'))
    require(main_config['config_sha256']==CONFIG['main_config_sha256'],'Wrong main config binding')
    require(json_sha({k:v for k,v in main_config.items() if k!='config_sha256'})==main_config['config_sha256'],'Main config corrupt')
    for key in ('attempt_id','arm','worker_sha256'):
        require(main_report[key]==CONFIG[key]==main_config[key],f'Main identity differs: {key}')
    require(main_report['status']=='MAIN_COMPLETE' and main_report['pass'] is False,'Main is not ready')
    X,y,m,regenerated=prepare()
    require(REPORT['initialization']==main_report['initialization'],'Fresh initialization differs')
    spec=json.loads((FOLDER/'schedule_spec.json').read_text(encoding='utf-8'))
    require(spec==regenerated,'Saved schedule/reconstructed input contract differs')
    recovery=FOLDER/'_recovery'
    boundary=json.loads((recovery/'boundary.json').read_text(encoding='utf-8'))
    require(boundary['main_config_sha256']==CONFIG['main_config_sha256'],'Checkpoint main binding differs')
    require(boundary['contract']==fixed_contract(m,spec),'Fixed supports/graph/data/code/runtime contract differs')
    consumed=json.loads((FOLDER/'consumed_trace.json').read_text(encoding='utf-8'))
    boundary_ledger=json.loads((recovery/'boundary_ledger.json').read_text(encoding='utf-8'))
    require(json_sha(boundary_ledger)==boundary['ledger_sha256'],'Boundary input ledger corrupt')
    require(main_report['trace']==trace_summary(consumed,46),'Main consumed ledger corrupt')
    phase('restore model/BN/Adam + audit + external Dropout recorders in NEW process')
    checkpoint_object(m).read(str(recovery/'checkpoint')).assert_consumed()
    require(state_hashes(m)==boundary['state_hashes'],'Boundary restore is not bitwise exact')
    expected_boundary_calls=1530 if CONFIG['arm'] in GAN_ARMS else 90
    require(int(m['opt'].iterations.numpy())==1530 and int(m['audit']['calls'].numpy())==expected_boundary_calls,'Checkpoint boundary clock/audit differs')
    phase('replay saved next_loop45 using restored state and saved schedule')
    replay=new_ledger()
    run_loop(m,X,spec,45,replay)
    reference=json.loads((recovery/'reference_ledger.json').read_text(encoding='utf-8'))
    atomic_json(FOLDER/'resume_ledger.json',replay)
    exact_trace=(reference==replay)
    # Reference NPZ digest is checked independently before comparing to replay.
    reference_hashes=json.loads((recovery/'reference_state_hashes.json').read_text(encoding='utf-8'))
    with np.load(recovery/'reference_state.npz',allow_pickle=False) as raw:
        require({key:sha_array(raw[key]) for key in raw.files}==reference_hashes,'Reference state artifact corrupt')
    comparison=compare_state(recovery/'reference_state.npz',m)
    comparison.update(trace_exact=exact_trace,state_restore_exact=True,
                      trace_reference_sha256=json_sha(reference),trace_replay_sha256=json_sha(replay))
    atomic_json(FOLDER/'resume_compare.json',comparison)
    require(exact_trace,'Actual replay input/mask/anchor/noise trace differs')
    require(comparison['numeric_allclose'],'Resumed trajectory differs beyond preset tolerances; see resume_compare.json')
    final_calls=1564 if CONFIG['arm'] in GAN_ARMS else 92
    require(int(m['opt'].iterations.numpy())==1564 and int(m['audit']['calls'].numpy())==final_calls,'Final resumed counters differ')
    # Remove ONLY this attempt's known recovery directory, never a JSON-supplied path.
    require(recovery.resolve().parent==FOLDER.resolve() and recovery.name=='_recovery','Unsafe recovery cleanup path')
    shutil.rmtree(recovery)
    REPORT['benchmark']=main_report['benchmark'];REPORT['tail']=main_report['tail']
    REPORT['trace']=main_report['trace'];REPORT['boundary']=main_report['boundary']
    REPORT['clocks']={**main_report['clocks'],'replay_C_updates':2,
                      'replay_optimizer_calls':34 if CONFIG['arm'] in GAN_ARMS else 2,
                      'final_clock':1564,'physical_C_updates_across_main_and_replay':94,
                      'physical_optimizer_calls':1598 if CONFIG['arm'] in GAN_ARMS else 94}
    REPORT['resume']={**comparison,'rtol':1e-6,'atol':1e-7,
                       'top_level_tolerance_scope':'weights/BN only; Adam m/v and audit float use atol1e-12 in comparison_policy',
                       'temporary_recovery_removed':True,
                       'fresh_process':True,'general_stateful_RNG_resume_claimed':False,
                       'continuation_kind':'bitwise' if comparison['bitwise_equal'] else 'numerical within preset tolerances'}
    REPORT['gates']={'benchmark':True,'actual_tail_batch':True,'checkpoint_restore':True,'resume_trace':True,'resume_numeric':True}
    REPORT.update(status='PASS',**{'pass':True},phase='final bounded technical gate complete',
                  elapsed_seconds=round(time.monotonic()-START,3))
    atomic_json(FOLDER/'final_report.json',REPORT)
    print('ARM PASS:',CONFIG['arm'],'| tail22 + benchmark + fresh-process continuation | no accuracy/test',flush=True)


def main():
    global CONFIG,REPORT,FOLDER,START,np
    CONFIG=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    FOLDER=Path(CONFIG['attempt_dir']);START=CONFIG['start_monotonic']
    REPORT={'attempt_id':CONFIG['attempt_id'],'arm':CONFIG['arm'],'config_sha256':CONFIG['config_sha256'],
            'main_config_sha256':CONFIG.get('main_config_sha256'),'worker_sha256':CONFIG['worker_sha256'],
            'execution_mode':CONFIG['execution_mode'],'status':'RUNNING','pass':False,
            'scope':'final bounded technical gate; no classification study',
            'rho':RHO,'k':K,'seed':CONFIG['seed'],'batch_size':BATCH,
            'route_counts':{},'fake_patch_hashes':[],'random_weight_consumption':[],'G_anchor_consumption':[],
            'accuracy_computed':False,'validation_arrays_loaded':False,'test_arrays_loaded':False,
            'full_training_allowed':False,'historical_B_bitwise_replay':False,
            'dropout_policy':'new MOC2 matched stateless keys; actual masks recorded; rate0.5 unchanged'}
    try:
        require(sys.platform.startswith('linux') and Path('/kaggle/working').is_dir(),'Kaggle only; no local TF execution')
        require(CONFIG['arm'] in ('original_gan','matched_ros','random_local','learned_local'),'Unknown arm')
        require(CONFIG['execution_mode'] in ('main','resume'),'Unknown execution stage')
        require(CONFIG['data_mode']=='auto_real_B' and CONFIG['seed']==42,'Locked real B/seed42')
        require(json_sha({k:v for k,v in CONFIG.items() if k!='config_sha256'})==CONFIG['config_sha256'],'Config integrity failed')
        require(file_sha(__file__)==CONFIG['worker_sha256'],'Worker integrity failed')
        phase('worker start / NumPy')
        import numpy as numpy
        np=numpy
        if CONFIG['execution_mode']=='main':run_main()
        else:run_resume()
    except BaseException as error:
        REPORT.update(status='TIMEOUT' if isinstance(error,TimeoutError) else 'FAILED',**{'pass':False},
                      error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
                      elapsed_seconds=round(time.monotonic()-START,3))
        target='main_report.json' if CONFIG['execution_mode']=='main' else 'final_report.json'
        atomic_json(FOLDER/target,REPORT)
        traceback.print_exc();sys.exit(1)


if __name__=='__main__':main()
