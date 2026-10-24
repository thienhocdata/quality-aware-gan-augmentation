from pathlib import Path, PurePosixPath
import hashlib, json, sys, zipfile
import numpy as np
import pandas as pd
tf = keras = None
CONFIG = {
    'input_roots': ['/kaggle/input'],
    'model_seeds': [42,43,44],
    'updates': [7500,8000],
    'analysis_seed': 20261016,
    'noises_per_class': 64,
    'batch_size': 32,
}
# TensorFlow được import sau khi tìm đủ6checkpoint, xác minh train và graph.


"""Read-only M4 V2 input discovery. No TensorFlow, model execution or pickle.

Designed to be embedded in the Kaggle notebook. Disk writes are limited to
verified ZIP payload copies under an explicit /kaggle/working destination.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import zipfile

PROTOCOL_ID = "50e988fae260"
IMPLEMENTATION_B = "5e15f8214bf273bb889187b96c80407a8000e5e1a3ef30734bcfa86ced275a6f"
M4_WORKER_SHA = "6d33ca6a0d39600f5f44c0d7e63547b1237149d625471ac73315d982ffbd8fb5"
TRAIN_X_SHA = "26cb18329d396d46e9d639420f5c718ae98c9cd938de5fa6c35240bc6d6227b0"
TRAIN_Y_SHA = "28d08700b81863a431bf78230196387afe755ee37c4a69a47e871068e7d25921"
TRAIN_FILE_SHA = "503898ef73ebf18060813ba51be8cc7b8c57474ed3d22f2fffeb9e8c24075cb6"
PREPROCESSING_SHA = "18167b01cbb8ac5d21ff73a3588cad758d23a2629f9b1c9e47dbaf4cf6989939"
MANIFEST_SHA = "41486ba39177e36f32e09c61cff870f4086a94c7b55f39db9dbe4b92093d6e67"
PROTOCOL_SHA = "d4260da1ba9289a3183ab10905f995d6a2e8956b5651576734683017705bd202"
FINAL_COUNTS = [21,753,426,138,209,376,16,228,10,469,1390,311,125,720,287,49]
CLASS_ORDER = [9,7,1,16,13,4,5,8,15,12,6,3,10,14,2,11]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha_stream(handle):
    h = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def sha_file(path):
    with Path(path).open("rb") as f:
        return sha_stream(f)


def array_sha(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def b_array_sha(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.dtype.str.encode())
    h.update(a.tobytes())
    return h.hexdigest()


class InputRef:
    """One direct file or one unique ZIP member; extraction is never extractall."""
    def __init__(self, path, member=None):
        self.path = Path(path)
        self.member = member

    def label(self):
        return str(self.path) if self.member is None else f"{self.path}::{self.member}"

    def open(self):
        if self.member is None:
            return self.path.open("rb")
        z = zipfile.ZipFile(self.path)
        # The returned stream owns the ZIP until it closes (standard ZipExtFile).
        matches = [x for x in z.infolist() if x.filename == self.member]
        require(len(matches) == 1, f"Duplicate ZIP member: {self.label()}")
        stream = z.open(matches[0])
        # Reading this stream to EOF checks ZIP CRC. ZipExtFile holds its own
        # shared-file reference; the archive's public handle can close here.
        z.close()
        return stream

    def read_json(self):
        with self.open() as f:
            return json.load(f)

    def sha(self):
        with self.open() as f:
            return sha_stream(f)

    def sibling(self, name):
        if self.member is None:
            return InputRef(self.path.with_name(name))
        return InputRef(self.path, str(PurePosixPath(self.member).with_name(name)))

    def exists(self):
        if self.member is None:
            return self.path.is_file()
        with zipfile.ZipFile(self.path) as z:
            return sum(n == self.member for n in z.namelist()) == 1

    def materialize(self, folder, name=None):
        if self.member is None:
            return self.path
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        name = name or PurePosixPath(self.member).name
        require(Path(name).name == name, "Only fixed basename extraction is allowed")
        dest = folder / name
        if dest.is_file() and sha_file(dest) == self.sha():
            return dest
        temp = dest.with_name(dest.name + ".partial")
        with self.open() as src, temp.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        require(sha_file(temp) == self.sha(), f"Extracted byte hash mismatch: {self.label()}")
        temp.replace(dest)
        return dest


def inventory_input_refs(roots):
    """Metadata and train filenames only; never load val/test arrays or archives.

    The archive directory is inspected, but only requested members are opened.
    A ZIP that contains nested ZIPs is recorded as such and not auto-executed.
    """
    meta_refs, train_refs, issues = [], [], []
    zip_paths = set()
    for root in map(Path, roots):
        if root.is_file():
            if root.suffix.lower() == ".zip":
                zip_paths.add(root)
            elif root.name == "meta.json":
                meta_refs.append(InputRef(root))
            elif root.name == "train_5528.npz":
                train_refs.append(InputRef(root))
            continue
        if not root.is_dir():
            issues.append({"source": str(root), "issue": "Input root missing"})
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name == "meta.json":
                meta_refs.append(InputRef(path))
            elif path.name == "train_5528.npz":
                train_refs.append(InputRef(path))
            elif path.suffix.lower() == ".zip":
                zip_paths.add(path)
    for path in sorted(zip_paths):
        try:
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    name = PurePosixPath(info.filename).name
                    if name == "meta.json":
                        meta_refs.append(InputRef(path, info.filename))
                    elif name == "train_5528.npz":
                        train_refs.append(InputRef(path, info.filename))
        except (OSError, zipfile.BadZipFile) as error:
            issues.append({"source": str(path), "issue": f"Unreadable ZIP: {error}"})
    return meta_refs, train_refs, issues






def _find_protocol_refs(train_ref):
    # Ascend based on file structure. No dependency on B run/dataset root name.
    if train_ref.member is None:
        for parent in train_ref.path.parents:
            manifest = InputRef(parent / "manifest.json")
            protocol = InputRef(parent / "protocol_spec.json")
            if manifest.exists() and protocol.exists():
                return manifest,protocol
    else:
        for parent in PurePosixPath(train_ref.member).parents:
            prefix = "" if str(parent) == "." else str(parent) + "/"
            manifest = InputRef(train_ref.path, prefix+"manifest.json")
            protocol = InputRef(train_ref.path, prefix+"protocol_spec.json")
            if manifest.exists() and protocol.exists():
                return manifest,protocol
    raise RuntimeError(f"Missing B manifest/protocol ancestors: {train_ref.label()}")


def load_final_train_verified(train_refs, destination):
    import numpy as np
    candidates,rejected = [],[]
    for ref in train_refs:
        try:
            require(ref.sha() == TRAIN_FILE_SHA, "Final train NPZ differs from pinned M4 source")
            side = ref.sibling("train_5528.npz.json").read_json()
            prep = ref.sibling("preprocessing_final.joblib")
            prep_side = ref.sibling("preprocessing_final.joblib.json").read_json()
            manifest_ref,protocol_ref = _find_protocol_refs(ref)
            require(manifest_ref.sha() == MANIFEST_SHA and protocol_ref.sha() == PROTOCOL_SHA,
                    "B manifest/protocol pins differ")
            manifest,protocol = manifest_ref.read_json(),protocol_ref.read_json()
            require(manifest.get("run_id") == PROTOCOL_ID and manifest.get("protocol_spec_sha256") == PROTOCOL_SHA,
                    "B manifest identity mismatch")
            require(hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:12] == PROTOCOL_ID,
                    "B protocol/run ID link mismatch")
            require(protocol.get('implementation_sha256') == IMPLEMENTATION_B and
                    protocol.get('patch_size') == 11 and protocol.get('pca_components') == 40,
                    "B protocol preprocessing/implementation differs from M4")
            for record in (side,prep_side):
                require(record["contract"].get("run_id") == PROTOCOL_ID and record["contract"].get("implementation") == IMPLEMENTATION_B,
                        "Final artifact contract identity mismatch")
            require(side.get("sha256") == TRAIN_FILE_SHA and side["contract"].get("kind") == "final_train",
                    "Final train sidecar mismatch")
            require(prep.sha() == PREPROCESSING_SHA == prep_side.get("sha256") == side["contract"].get("preprocessing_sha256"),
                    "Final preprocessing byte hash/provenance mismatch")
            require(prep_side["contract"].get("pca_components") == 40 and prep_side["contract"].get("pca_solver") == "full",
                    "Final PCA metadata differs")
            candidates.append((ref,side,prep_side))
        except Exception as error:
            rejected.append({"source":ref.label(),"reason":str(error)})
    require(candidates, "No verified M4 final-train cache found. Add original B cache input. Rejections: "+json.dumps(rejected))
    candidates.sort(key=lambda x: (x[0].member is not None,x[0].label()))
    ref,side,prep_side = candidates[0]
    path = ref.materialize(destination,"train_5528.npz")
    with np.load(path,allow_pickle=False) as d:
        # These are the only arrays materialized. Never X_val or X_test.
        X,y,coords = d["X_train"],d["y_train"],d["coords_train"]
    require(X.dtype == np.float32 and X.shape == (5528,11,11,40,1), "Wrong final train X shape/dtype")
    require(y.dtype == np.int64 and y.shape == (5528,), "Wrong final train y shape/dtype")
    require(array_sha(X) == TRAIN_X_SHA and array_sha(y) == TRAIN_Y_SHA, "Train array hash differs from M4 job contract")
    require(np.bincount(y,minlength=16).tolist() == FINAL_COUNTS, "Final train class counts differ")
    require(coords.shape == (5528,2) and np.issubdtype(coords.dtype,np.integer), "Bad final coords shape/dtype")
    require(len(np.unique(coords,axis=0)) == 5528 and np.all((coords >= 0)&(coords < 145)), "Bad final train coordinates")
    require(b_array_sha(coords) == side["contract"]["coords"] == prep_side["contract"]["fit_coords_sha256"],
            "Final train centers differ from PCA fit centers")
    require(np.isfinite(X).all(), "Nonfinite final train features")
    return X,y,coords,{"source":ref.label(),"npz_sha256":TRAIN_FILE_SHA,
                        "preprocessing_sha256":PREPROCESSING_SHA,
                        "eligible_identical_copies":len(candidates),"rejected":rejected,
                        "validation_arrays_loaded":False,"test_arrays_loaded":False,
                        "joblib_deserialized":False}

ENDGAME_PINS = {(42, 7500): {'meta.json': 'b75a107d2c0290ac32e939e33e623af5c3e94e7c4a517ceed22e1c5dc8e4e4cc', 'weights.index': '7d5f973a82d9509617e07d9bce6fe318a0274e1b1b29907a2537d6aa483ee49c', 'weights.data-00000-of-00001': 'd0b3e70cc4de3d431642512efc8c4ce4501c14150d0d527ffc065999a21cefea'}, (42, 8000): {'meta.json': 'cc0f0b8817ea074b30ed3d044cb96f3069c6d591fc4eb345d5d0012880431e42', 'weights.index': '3767710c97ee62f2f2fb3b2470f85f8bb2587c71044b7a0a3d8babb243317a7b', 'weights.data-00000-of-00001': '5af9bd48710d98a9a883309506a8cba4365929713927b9aa78ed34842e471415'}, (43, 7500): {'meta.json': 'ca24919fa9dc7a44761efa4ff08acfc8e2ad42a6a3946a33ecb21656fdef5647', 'weights.index': 'ed1bf1440919d974bd2014431125baa90fa6695d6d441ae15e8cbe54c597fbf7', 'weights.data-00000-of-00001': '9cb94bf6e7066f090f2492ea3fb1f6e9a5a1eb5eaca1a52fb37bf4a620485d1f'}, (43, 8000): {'meta.json': '1f9478c8339266746a8218f70b99da4a1948c866fffe2197a011234d77cfc9eb', 'weights.index': 'b96d463087f8ea2dbcf6b1631a293b00d752b8313bf9bb681790ca9fb3ccb9c3', 'weights.data-00000-of-00001': '97cc8b798f58601872de02cbc3b396a4514821d40f0561b898eec5213fb32cdb'}, (44, 7500): {'meta.json': 'b25287ac7c3d79b156d87d6fc6ea90e1c7ddea259c67f22e6c1693abd4c3e961', 'weights.index': '5bc7596aab1bcfd1e187d92911ffffae4a881e7bb410078ba2c9e46151a6d515', 'weights.data-00000-of-00001': '255155d97c6a9aabbd41bb67c39d9815015cfc61ca35b7cc3307ae763b091449'}, (44, 8000): {'meta.json': '678b69d03feb3c9fabe03e2567c41abc08c961a2a4f82e1bd3c9e96b38105674', 'weights.index': '1847a904e5286a56a888f7300f06f17a73e1958fb88103a9cfe0b138aa63c976', 'weights.data-00000-of-00001': '8fa310ec1f4c33e397735afe6b13ef4602c0a08c4e696895335fee783a1b0942'}}
USER_CODE_SHA = 'd1d6026ce67201b8f43b539907242b53e5c2ed22deb0287805186c8c79e6b93d'


SHAPE=(11,11,40,1)
FLAT=4840
N_CLASSES=16
K=3
RHO=.2
EXACT_FUNCTION_SHA = {'sha_array': '2c79ec8643f9648e27b3bdfaa191c9bc3f502290ed88a894f28073fab651ae8a', 'build_graph_per_class': '580c122db5b3b9950e317e0152ddcf59895a349934fa03ea0ce4716715e9b25a', 'tao_generator': '6eb347f75e905b4a03c6f65e71667c58f6cdeb4b91cd9682d7beb37795428248', 'tao_head_v2': 'c8349ddb058e8c5fd7ecd82d6ba2b24748a76399c4bc9bc942a26ac5a7b90e43'}
M4_EXACT_SOURCE = r'''
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
'''
M4_EXACT_SOURCE=M4_EXACT_SOURCE.lstrip("\n")
require(hashlib.sha256(M4_EXACT_SOURCE.encode()).hexdigest()=='9c008fe3edb05daba4f8839479b014662bde81b20842796cf11bcd6dc41169a9', 'M4 embedded source changed')
exec(compile(M4_EXACT_SOURCE,'<original_M4_G_head_graph>','exec'),globals())
"""Exact native M4 learned_local G/head reader and restore helpers.

Paste after the source-extracted M4 functions tao_generator, tao_head_v2,
build_graph_per_class and sha_array. The Kaggle parent supplies globals tf,
keras and NumPy. This file never imports TensorFlow or constructs optimizers.
"""
from pathlib import Path
import hashlib
import re
import numpy as np


_V2_CLASS_ORDER = [9, 7, 1, 16, 13, 4, 5, 8, 15, 12, 6, 3, 10, 14, 2, 11]
_V2_HEAD_COUNTS = [10, 16, 21, 49, 125, 138, 209, 228, 287, 311,
                   376, 426, 469, 720, 753]
_V2_WORKER_SHA = '6d33ca6a0d39600f5f44c0d7e63547b1237149d625471ac73315d982ffbd8fb5'
_V2_IMPL_B = '5e15f8214bf273bb889187b96c80407a8000e5e1a3ef30734bcfa86ced275a6f'
_V2_X_SHA = '26cb18329d396d46e9d639420f5c718ae98c9cd938de5fa6c35240bc6d6227b0'
_V2_Y_SHA = '28d08700b81863a431bf78230196387afe755ee37c4a69a47e871068e7d25921'
_V2_G_TRAINABLE = [
    (0, 'kernel', (116, 12544)),
    (1, 'gamma', (12544,)), (1, 'beta', (12544,)),
    (2, 'kernel', (5, 5, 128, 256)),
    (3, 'gamma', (128,)), (3, 'beta', (128,)),
    (4, 'kernel', (5, 5, 64, 128)),
    (5, 'gamma', (64,)), (5, 'beta', (64,)),
    (6, 'kernel', (5, 5, 1, 64)),
    (7, 'gamma', (1,)), (7, 'beta', (1,)),
]
_V2_G_NONTRAINABLE = [
    (layer, name, (width,))
    for layer, width in ((1, 12544), (3, 128), (5, 64), (7, 1))
    for name in ('moving_mean', 'moving_variance')
]


def _v2_restore_require(condition, message):
    if not condition:
        raise RuntimeError('M4 exact restore: ' + message)


def _v2_file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _v2_semantic_mapping():
    """Independent, original-index-verified native-name/semantic map (50 arrays)."""
    result = {}
    for part, variables in (('trainable', _V2_G_TRAINABLE),
                            ('nontrainable', _V2_G_NONTRAINABLE)):
        for index, (layer, name, shape) in enumerate(variables):
            semantic = f'{part}/G/{index}'
            result[semantic] = {
                'tensor_key': f'G/layer_with_weights-{layer}/{name}/.ATTRIBUTES/VARIABLE_VALUE',
                'shape': shape,
            }
    for head, count in enumerate(_V2_HEAD_COUNTS):
        for index, (name, shape) in enumerate((('kernel', (784, count)),
                                                ('bias', (count,)))):
            semantic = f'trainable/head_{head}/{index}'
            result[semantic] = {
                'tensor_key': f'head_{head:02d}/layer_with_weights-0/{name}/.ATTRIBUTES/VARIABLE_VALUE',
                'shape': shape,
            }
    return result


def _v2_check_metadata(meta):
    contract = meta.get('contract', {})
    wanted = {
        'protocol_B_id': '50e988fae260',
        'implementation_B': _V2_IMPL_B,
        'stage': 'final', 'arm': 'learned_local',
        'budget_C_updates': 8000, 'sha_worker': _V2_WORKER_SHA,
        'train_X_sha256': _V2_X_SHA, 'train_y_sha256': _V2_Y_SHA,
        'rho': 0.2, 'k': 3, 'class_order': _V2_CLASS_ORDER,
    }
    for key, value in wanted.items():
        _v2_restore_require(contract.get(key) == value,
                            f'checkpoint contract mismatch for {key}: {contract.get(key)!r}')
    _v2_restore_require(contract.get('seed') in (42, 43, 44), 'unexpected model seed')
    updates = meta.get('c_updates')
    _v2_restore_require(updates in (7500, 8000), 'unexpected audit checkpoint update count')
    loop = updates // 2
    for key, value in (('c_updates', updates), ('next_loop', loop),
                        ('optimizer_clock', 34 * loop), ('audit_calls', 34 * loop)):
        _v2_restore_require(meta.get(key) == value, f'checkpoint counter mismatch: {key}')
    expected = _v2_semantic_mapping()
    observed = {name for name in meta.get('state_hashes', {})
                if re.fullmatch(r'(?:trainable|nontrainable)/(?:G|head_\d+)/\d+', name)}
    _v2_restore_require(observed == set(expected),
                        'metadata G/head semantic state inventory is not the original 50 arrays')
    for semantic in expected:
        _v2_restore_require(bool(re.fullmatch(r'[0-9a-f]{64}', meta['state_hashes'][semantic])),
                            f'invalid semantic SHA-256: {semantic}')
    return contract


def preflight_checkpoint_tensors(folder: Path, meta: dict):
    """Return ``(semantic_to_numpy_arrays, audit)`` after strict native validation.

    Entire index/data-file integrity is checked, but only the 50 G/head model
    arrays are loaded. C, D, optimizer and training-control variables are
    deliberately not reconstructed or consumed.
    """
    folder = Path(folder)
    contract = _v2_check_metadata(meta)
    wanted_files = {'weights.index', 'weights.data-00000-of-00001'}
    _v2_restore_require(set(meta.get('weights_hashes', {})) == wanted_files,
                        'unexpected checkpoint file inventory/shard count')
    actual_file_hashes = {}
    for name in sorted(wanted_files):
        path = folder / name
        _v2_restore_require(path.is_file(), f'missing native checkpoint file: {path}')
        actual_file_hashes[name] = _v2_file_sha(path)
        _v2_restore_require(actual_file_hashes[name] == meta['weights_hashes'][name],
                            f'checkpoint file SHA-256 mismatch: {path}')
    reader = tf.train.load_checkpoint(str(folder / 'weights'))
    shapes = reader.get_variable_to_shape_map()
    dtypes = reader.get_variable_to_dtype_map()
    mapping = _v2_semantic_mapping()
    expected_keys = {item['tensor_key'] for item in mapping.values()}
    actual_keys = {key for key in shapes
                   if (key.startswith('G/') or key.startswith('head_'))
                   and '/.OPTIMIZER_SLOT/' not in key}
    _v2_restore_require(actual_keys == expected_keys,
                        'native G/head tensor inventory mismatch; missing=' +
                        repr(sorted(expected_keys - actual_keys)) + '; extra=' +
                        repr(sorted(actual_keys - expected_keys)))
    tensors, records = {}, []
    for semantic, item in mapping.items():
        key, expected_shape = item['tensor_key'], tuple(item['shape'])
        _v2_restore_require(tuple(shapes[key]) == expected_shape,
                            f'wrong native tensor shape: {key}')
        _v2_restore_require(getattr(dtypes[key], 'name', str(dtypes[key])) == 'float32',
                            f'wrong native tensor dtype: {key}')
        array = np.asarray(reader.get_tensor(key))
        _v2_restore_require(array.dtype == np.dtype('float32') and array.shape == expected_shape,
                            f'reader array signature mismatch: {key}')
        _v2_restore_require(np.isfinite(array).all(), f'nonfinite native array: {key}')
        observed_hash = sha_array(array)
        _v2_restore_require(observed_hash == meta['state_hashes'][semantic],
                            f'native array semantic SHA-256 mismatch: {semantic}')
        # Own the buffer after dropping TensorFlow checkpoint reader state.
        tensors[semantic] = np.array(array, dtype=np.float32, order='C', copy=True)
        records.append({'semantic_key': semantic, 'tensor_key': key,
                        'shape': list(expected_shape), 'dtype': 'float32',
                        'sha256': observed_hash, 'matches_metadata': True})
    excluded_count = len(shapes) - len(expected_keys)
    audit = {
        'status': 'PASS', 'seed': contract['seed'], 'folder': str(folder),
        'protocol_B_id': contract['protocol_B_id'], 'arm': contract['arm'],
        'c_updates': meta['c_updates'], 'checkpoint_file_sha256': actual_file_hashes,
        'loaded_model_array_count': len(records), 'G_trainable_array_count': 12,
        'G_BN_buffer_count': 8, 'head_dense_array_count': 30,
        'arrays': records,
        'excluded_checkpoint_entry_count': excluded_count,
        'excluded_scope': 'C, D, Adam/optimizer slots, gradient audit, dropout controls and object graph; not needed for G/head-only inference',
        'full_training_checkpoint_consumption_claimed': False,
        'architecture_fallback_used': False,
    }
    del reader
    return tensors, audit


def _v2_destination_variables(G, heads):
    """Must be called before setting model.trainable=False (BN partition matters)."""
    _v2_restore_require(len(G.trainable_variables) == 12, 'rebuilt G trainable count differs')
    _v2_restore_require(len(G.non_trainable_variables) == 8, 'rebuilt G BN buffer count differs')
    _v2_restore_require(len(heads) == 15, 'rebuilt head count differs')
    result = {f'trainable/G/{i}': variable
              for i, variable in enumerate(G.trainable_variables)}
    result.update({f'nontrainable/G/{i}': variable
                   for i, variable in enumerate(G.non_trainable_variables)})
    for head_index, head in enumerate(heads):
        _v2_restore_require(len(head.trainable_variables) == 2,
                            f'rebuilt head_{head_index} has unexpected trainables')
        _v2_restore_require(not head.non_trainable_variables,
                            f'rebuilt head_{head_index} unexpectedly tracks support/graph variables')
        for i, variable in enumerate(head.trainable_variables):
            result[f'trainable/head_{head_index}/{i}'] = variable
    _v2_restore_require(set(result) == set(_v2_semantic_mapping()),
                        'destination variable inventory differs')
    return result


def restore_generator_and_heads(prepared, meta, tensors):
    """Build exact source G + 15 local heads; return ``(G, heads, audit)``.

    No compile, optimizer, C/D model, training API, gradient tape, or freeze-state
    mutation is involved. Every subsequent caller must explicitly pass
    ``training=False`` and verify pre/post parameter hashes.
    """
    contract = _v2_check_metadata(meta)
    mapping = _v2_semantic_mapping()
    _v2_restore_require(set(tensors) == set(mapping), 'preflight tensor registry incomplete')
    _v2_restore_require((np.asarray(prepared['order']) + 1).tolist() == _V2_CLASS_ORDER,
                        'train-derived class order differs from checkpoint')
    _v2_restore_require(len(prepared['supports']) == len(prepared['neighbor_graphs']) == 15,
                        'wrong train-derived support/head inventory')
    actual_graph_hashes = []
    for h, (support, graph) in enumerate(zip(prepared['supports'], prepared['neighbor_graphs'])):
        _v2_restore_require(np.asarray(support).dtype == np.float32 and
                            tuple(support.shape) == (_V2_HEAD_COUNTS[h], 11, 11, 40, 1),
                            f'head_{h} support signature differs')
        _v2_restore_require(np.asarray(graph).dtype == np.int32 and
                            tuple(graph.shape) == (_V2_HEAD_COUNTS[h], 3),
                            f'head_{h} graph signature differs')
        actual_graph_hashes.append(sha_array(graph))
    _v2_restore_require(actual_graph_hashes == contract['graph_sha256'],
                        'recomputed train-only graph hashes differ from checkpoint')
    G = tao_generator(100, 16)
    heads = [tao_head_v2(support, graph, rho=0.2)
             for support, graph in zip(prepared['supports'], prepared['neighbor_graphs'])]
    destinations = _v2_destination_variables(G, heads)
    records = []
    for semantic, variable in destinations.items():
        array = np.asarray(tensors[semantic])
        expected_shape = tuple(mapping[semantic]['shape'])
        _v2_restore_require(tuple(variable.shape) == expected_shape and array.shape == expected_shape,
                            f'destination shape mismatch: {semantic}')
        _v2_restore_require(getattr(variable.dtype, 'name', str(variable.dtype)) == 'float32',
                            f'destination dtype mismatch: {semantic}')
        _v2_restore_require(sha_array(array) == meta['state_hashes'][semantic],
                            f'verified tensor was changed before assignment: {semantic}')
        variable.assign(array)
        assigned = np.asarray(variable.numpy())
        _v2_restore_require(sha_array(assigned) == meta['state_hashes'][semantic],
                            f'assigned destination semantic SHA-256 mismatch: {semantic}')
        records.append({'semantic_key': semantic, 'shape': list(expected_shape),
                        'dtype': 'float32', 'assigned_sha256': sha_array(assigned),
                        'matches_metadata': True})
    _v2_restore_require(tuple(G.output_shape) == (None, 784), 'rebuilt G output differs')
    _v2_restore_require(len(G.weights) == 20 and sum(len(h.weights) for h in heads) == 30,
                        'rebuilt G/head total array count differs')
    for head_index, head in enumerate(heads):
        _v2_restore_require(tuple(head.output_shape) == (None, 11, 11, 40, 1),
                            f'rebuilt head_{head_index} patch output differs')
        _v2_restore_require(head.get_layer('support_logits').activation == keras.activations.linear,
                            f'head_{head_index} logits activation differs')
    audit = {
        'status': 'PASS', 'seed': contract['seed'], 'array_count': len(records),
        'arrays': records, 'train_only_graph_sha256': actual_graph_hashes,
        'architecture_source_sha256': contract['sha_worker'],
        'G_constructor': 'source-extracted tao_generator(100,16)',
        'head_constructor': 'source-extracted tao_head_v2(support,graph,rho=0.2)',
        'source_architecture_used': True, 'architecture_fallback_used': False,
        'models_compiled': False, 'optimizer_constructed': False,
        'C_or_D_constructed': False, 'training_performed': False,
        'required_forward_mode': 'training=False, exact final BN moving buffers',
        'trainable_flags_changed': False,
        'majority_original_class_id_without_head': 11,
    }
    return G, heads, audit


"""Pure NumPy endpoint diagnostics for actual restored M4 V2 softmax weights.

No model construction, TensorFlow, fitting or inference is performed here.
``weights[a,m,j]`` must already be the actual 3-neighbor TensorFlow softmax
output, with fixed anchor ``a``, sampled noise ``m`` and graph distance rank
``j``. Distances are in the original train PCA-patch coordinates.
"""

import numpy as np


METRIC_DESCRIPTIONS = {
    "entropy_mean_nats": "Mean over noise of H(w)=-sum(w log w), natural logs, three neighbors only.",
    "entropy_std_nats": "Population standard deviation over the common sampled noises of H(w).",
    "effective_neighbors_mean": "Mean over noise of exp(H(w)); not exp(mean entropy); range 1..3.",
    "effective_neighbors_std": "Population standard deviation over noise of exp(H(w)).",
    "maximum_weight_mean": "Mean over noise of the maximum conditional neighbor weight; range 1/3..1.",
    "maximum_weight_std": "Population standard deviation over noise of maximum conditional weight.",
    "weight_mean_by_rank": "Mean actual weight for each fixed neighbor distance rank 1,2,3.",
    "weight_std_by_rank": "Population standard deviation over noise for each neighbor distance rank.",
    "weight_noise_rms_l2": "sqrt(mean_m sum_j (w_mj-mean_m(w_mj))**2).",
    "noise_jsd_nats": "H(mean_m w_m)-mean_m H(w_m), generalized JSD over sampled noises; no quality interpretation.",
    "tv_to_uniform_mean": "Mean over noise of 0.5*sum_j(abs(w_j-1/3)).",
    "kl_to_uniform_mean_nats": "ln(3)-mean entropy; algebraically redundant with entropy.",
    "mean_weight_entropy_nats": "Entropy of mean weights; differs from mean entropy across noise.",
    "mixture_entropy_mean_nats": "Entropy of [.8,.2*w] for rho=.2, or [1-rho,rho*w] generally.",
    "mixture_effective_supports_mean": "Mean exp(entropy) for the full anchor-plus-neighbor mixture.",
    "dominance_fraction_by_rank": "Frequency each rank attains maximum; exact ties are allocated fractionally.",
    "exact_argmax_tie_fraction": "Fraction of noises with more than one exactly maximal float weight.",
    "distinct_dominant_neighbors": "Number of the three ranks receiving nonzero fractional maximum frequency.",
    "patch_noise_rms_per_coordinate": "sqrt(mean_m ||x_m-mean_m x_m||**2 / P), fixed anchor/neighbors.",
    "patch_noise_rms_l2": "sqrt(mean_m ||x_m-mean_m x_m||**2), fixed anchor/neighbors.",
    "patch_pairwise_rms_per_coordinate": "RMS difference per coordinate over all distinct noise pairs; sqrt(2M/(M-1)) times noise RMS.",
    "anchor_displacement_rms_per_coordinate": "sqrt(mean_m ||x_m-x_anchor||**2 / P); second-moment RMS, not mean distance.",
    "uniform_displacement_rms_per_coordinate": "sqrt(mean_m ||x_m-x_uniform_neighbors||**2 / P).",
    "mean_patch_shift_from_uniform_per_coordinate": "||mean_m(x_m)-x_uniform_neighbors||/sqrt(P).",
    "neighbor_radius_rms_per_coordinate": "sqrt(mean_j ||x_neighbor_j-x_anchor||**2 / P).",
}


def entropy(weights):
    """Natural-log entropy on the final axis; exactly zero weights are valid."""
    value = np.asarray(weights, dtype=np.float64)
    logs = np.zeros_like(value)
    np.log(value, out=logs, where=value > 0)
    return -np.sum(value * logs, axis=-1)


def _nonnegative(value, scale=1.0, label="quadratic", absolute_tolerance=0.0):
    """Reject material negative results and clip only float64 roundoff."""
    value = np.asarray(value, dtype=np.float64)
    tolerance = max(absolute_tolerance, 256 * np.finfo(np.float64).eps * max(1.0, float(scale)))
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Nonfinite {label}")
    if np.min(value, initial=0) < -tolerance:
        raise ValueError(f"Material negative {label}: {float(np.min(value))}")
    return np.maximum(value, 0.0)


def analyze_weight_tensor(weights, supports, graph, rho=0.2):
    """Return per-anchor diagnostics without ever allocating A x M x P.

    Parameters
    ----------
    weights : (A, M, 3) array
        Actual softmax outputs for every support used once as an anchor, M
        common noises, neighbors in the exact training graph's distance order.
    supports : (A, P) array
        Original class supports flattened in original head-local row order.
    graph : (A, 3) integer array
        Three distinct non-self head-local neighbors of each anchor.
    rho : float
        Fixed mixing coefficient used in training; M4 V2 is exactly 0.2.

    Notes
    -----
    No re-softmax or renormalization of supplied weights occurs. Float32
    softmax sum roundoff is accepted to 2e-6. For noise/geometry differences,
    centered neighbor coordinates remove the common support component; the
    resulting O(eps_float32) translation residual is consistent with ideal
    convex-mixture semantics and must be checked against actual head outputs
    separately in the notebook. Statistics use population moments over this
    finite noise probe. No independence, confidence or causality claim follows.
    """
    w = np.asarray(weights, dtype=np.float64)
    s = np.asarray(supports)
    g = np.asarray(graph)
    if w.ndim != 3 or w.shape[-1] != 3:
        raise ValueError("weights must have shape (anchors,noises,3)")
    a_count, n_noise, _ = w.shape
    if a_count < 1 or n_noise < 2:
        raise ValueError("Require at least one anchor and two noise samples")
    if s.ndim != 2 or s.shape[0] != a_count or s.shape[1] < 1:
        raise ValueError("supports must have shape (anchors,patch_coordinates)")
    if g.shape != (a_count, 3) or not np.issubdtype(g.dtype, np.integer):
        raise ValueError("graph must be an integer (anchors,3) array")
    if np.any(g < 0) or np.any(g >= a_count):
        raise ValueError("graph contains out-of-range supports")
    if np.any(g == np.arange(a_count)[:, None]):
        raise ValueError("graph contains anchor itself")
    if np.any(np.diff(np.sort(g, axis=1), axis=1) == 0):
        raise ValueError("graph contains duplicate neighbors")
    if not np.isfinite(rho) or not (0 < float(rho) < 1):
        raise ValueError("rho must lie strictly between zero and one")
    if not np.isfinite(w).all() or np.any(w < 0) or np.any(w > 1):
        raise ValueError("weights must be finite probabilities")
    if not np.allclose(w.sum(-1), 1.0, rtol=0, atol=2e-6):
        raise ValueError("weights do not sum to one")
    if not np.isfinite(s).all():
        raise ValueError("supports contain nonfinite values")

    h = entropy(w)
    effective = np.exp(h)
    maximum = w.max(axis=-1)
    mean_w = w.mean(axis=1)
    dw = w - mean_w[:, None, :]
    # Enforce the analytically zero sum of a convex weight difference, only
    # inside centered-coordinate geometry, never by changing stored weights.
    dw_centered = dw - dw.mean(axis=-1, keepdims=True)
    mean_h = h.mean(axis=1)
    jsd = _nonnegative(entropy(mean_w) - mean_h, label="noise JSD")
    max_ties = w == maximum[..., None]
    fractional_dominance = max_ties / max_ties.sum(axis=-1, keepdims=True)
    mixture_h = (-(1-rho)*np.log(1-rho) - rho*np.log(rho)) + rho*h

    result = {
        "entropy_mean_nats": mean_h,
        "entropy_std_nats": h.std(axis=1),
        "effective_neighbors_mean": effective.mean(axis=1),
        "effective_neighbors_std": effective.std(axis=1),
        "maximum_weight_mean": maximum.mean(axis=1),
        "maximum_weight_std": maximum.std(axis=1),
        "weight_mean_by_rank": mean_w,
        "weight_std_by_rank": w.std(axis=1),
        "weight_noise_rms_l2": np.sqrt(np.mean(np.sum(dw**2, axis=-1), axis=1)),
        "noise_jsd_nats": jsd,
        "tv_to_uniform_mean": (0.5*np.abs(w-1/3).sum(axis=-1)).mean(axis=1),
        # Actual float32 probabilities can sum to 1 +/- 2e-6. For example
        # three float32 copies of 1/3 have sum 1.0000000298, so unnormalized
        # entropy can exceed ln(3) by ~3e-9. Never renormalize saved weights.
        "kl_to_uniform_mean_nats": _nonnegative(np.log(3)-mean_h, label="KL to uniform", absolute_tolerance=2e-6),
        "mean_weight_entropy_nats": entropy(mean_w),
        "mixture_entropy_mean_nats": mixture_h.mean(axis=1),
        "mixture_effective_supports_mean": np.exp(mixture_h).mean(axis=1),
        "dominance_fraction_by_rank": fractional_dominance.mean(axis=1),
        "exact_argmax_tie_fraction": (max_ties.sum(axis=-1)>1).mean(axis=1),
        "distinct_dominant_neighbors": (fractional_dominance.sum(axis=1)>0).sum(axis=-1),
    }

    p = s.shape[1]
    geometry = {key: np.empty(a_count, dtype=np.float64) for key in (
        "patch_noise_rms_per_coordinate", "patch_noise_rms_l2",
        "patch_pairwise_rms_per_coordinate", "anchor_displacement_rms_per_coordinate",
        "uniform_displacement_rms_per_coordinate", "mean_patch_shift_from_uniform_per_coordinate",
        "neighbor_radius_rms_per_coordinate",
    )}
    for a in range(a_count):
        neighbors = np.asarray(s[g[a]], dtype=np.float64)
        center = neighbors - neighbors[0]
        gram_center = center @ center.T / p
        gram_scale = np.max(np.abs(gram_center), initial=1.0)
        noise_m2 = _nonnegative(
            rho**2 * np.einsum("mj,jk,mk->m", dw_centered[a], gram_center, dw_centered[a], optimize=True),
            scale=gram_scale, label="patch noise variance",
        ).mean()
        geometry["patch_noise_rms_per_coordinate"][a] = np.sqrt(noise_m2)
        geometry["patch_noise_rms_l2"][a] = np.sqrt(p*noise_m2)
        geometry["patch_pairwise_rms_per_coordinate"][a] = np.sqrt(2*n_noise/(n_noise-1)*noise_m2)

        neighbor_deltas = neighbors - np.asarray(s[a], dtype=np.float64)
        gram_anchor = neighbor_deltas @ neighbor_deltas.T / p
        anchor_m2 = _nonnegative(
            rho**2*np.einsum("mj,jk,mk->m", w[a], gram_anchor, w[a], optimize=True),
            scale=np.max(np.abs(gram_anchor), initial=1.0), label="anchor displacement",
        ).mean()
        du = w[a]-1/3
        du -= du.mean(axis=-1, keepdims=True)
        uniform_m2 = _nonnegative(
            rho**2*np.einsum("mj,jk,mk->m", du, gram_center, du, optimize=True),
            scale=gram_scale, label="uniform displacement",
        ).mean()
        mean_du = du.mean(axis=0)
        mean_shift = _nonnegative(rho**2 * mean_du @ gram_center @ mean_du,
                                  scale=gram_scale, label="mean patch shift")
        geometry["anchor_displacement_rms_per_coordinate"][a] = np.sqrt(anchor_m2)
        geometry["uniform_displacement_rms_per_coordinate"][a] = np.sqrt(uniform_m2)
        geometry["mean_patch_shift_from_uniform_per_coordinate"][a] = np.sqrt(mean_shift)
        geometry["neighbor_radius_rms_per_coordinate"][a] = np.sqrt(np.trace(gram_anchor)/3)
    result.update(geometry)
    return result


def summarize_anchor_metrics(metrics):
    """Equal-anchor mean/population SD/q05/median/q95 for every metric.

    Scalar metrics produce scalar values; rank metrics retain their final
    length-three axis. This is a finite-probe descriptive summary only.
    """
    result = {}
    anchor_count = None
    for name, values in metrics.items():
        a = np.asarray(values, dtype=np.float64)
        if a.ndim not in (1, 2) or a.shape[0] < 1 or not np.isfinite(a).all():
            raise ValueError(f"Invalid per-anchor metric {name}")
        if anchor_count is None:
            anchor_count = a.shape[0]
        if a.shape[0] != anchor_count:
            raise ValueError("Per-anchor arrays have different lengths")
        result[name] = {
            "mean": np.mean(a, axis=0), "std": np.std(a, axis=0),
            "q05": np.quantile(a, .05, axis=0),
            "median": np.quantile(a, .5, axis=0),
            "q95": np.quantile(a, .95, axis=0),
        }
    return {"anchors": anchor_count, "metrics": result}


def run_synthetic_checks():
    """Verify metric boundaries and the Gram identity using synthetic arrays."""
    rng = np.random.default_rng(20261016)
    supports = rng.normal(size=(5, 19))
    graph = np.asarray([[j for j in range(5) if j != a][:3] for a in range(5)], dtype=np.int32)
    uniform = np.full((5, 63, 3), 1/3, dtype=np.float64)
    peaked = np.tile(np.asarray([1., 0., 0.]), (5, 63, 1))
    switching = np.tile(np.eye(3), (5, 21, 1))
    u = analyze_weight_tensor(uniform, supports, graph)
    f = analyze_weight_tensor(peaked, supports, graph)
    c = analyze_weight_tensor(switching, supports, graph)
    np.testing.assert_allclose(u["entropy_mean_nats"], np.log(3), atol=1e-14)
    np.testing.assert_allclose(u["effective_neighbors_mean"], 3, atol=1e-14)
    np.testing.assert_allclose(u["maximum_weight_mean"], 1/3, atol=1e-14)
    np.testing.assert_allclose(u["noise_jsd_nats"], 0, atol=1e-14)
    np.testing.assert_allclose(u["patch_noise_rms_per_coordinate"], 0, atol=1e-14)
    np.testing.assert_allclose(u["dominance_fraction_by_rank"], 1/3, atol=1e-14)
    u32 = analyze_weight_tensor(uniform.astype(np.float32), supports, graph)
    np.testing.assert_allclose(u32["kl_to_uniform_mean_nats"], 0, atol=2e-6)
    np.testing.assert_allclose(f["entropy_mean_nats"], 0, atol=1e-14)
    np.testing.assert_allclose(f["effective_neighbors_mean"], 1, atol=1e-14)
    np.testing.assert_allclose(f["maximum_weight_mean"], 1, atol=1e-14)
    np.testing.assert_allclose(f["noise_jsd_nats"], 0, atol=1e-14)
    np.testing.assert_allclose(f["patch_noise_rms_per_coordinate"], 0, atol=1e-14)
    np.testing.assert_allclose(c["entropy_mean_nats"], 0, atol=1e-14)
    np.testing.assert_allclose(c["noise_jsd_nats"], np.log(3), atol=1e-14)
    np.testing.assert_allclose(c["weight_noise_rms_l2"], np.sqrt(2/3), atol=1e-14)

    weights = rng.dirichlet(np.ones(3), size=(5, 64))
    actual = analyze_weight_tensor(weights, supports, graph)
    direct_errors = []
    for a in range(5):
        neigh = supports[graph[a]]
        patches = .8*supports[a] + .2*weights[a] @ neigh
        direct_noise = np.sqrt(np.mean((patches-patches.mean(axis=0))**2))
        direct_anchor = np.sqrt(np.mean((patches-supports[a])**2))
        uniform_patch = .8*supports[a] + .2*neigh.mean(axis=0)
        direct_uniform = np.sqrt(np.mean((patches-uniform_patch)**2))
        direct_shift = np.sqrt(np.mean((patches.mean(axis=0)-uniform_patch)**2))
        for key, ref in (
            ("patch_noise_rms_per_coordinate", direct_noise),
            ("anchor_displacement_rms_per_coordinate", direct_anchor),
            ("uniform_displacement_rms_per_coordinate", direct_uniform),
            ("mean_patch_shift_from_uniform_per_coordinate", direct_shift),
        ):
            np.testing.assert_allclose(actual[key][a], ref, rtol=1e-13, atol=1e-13)
            direct_errors.append(abs(actual[key][a]-ref))
    full_w = np.concatenate((np.full((5, 64, 1), .8), .2*weights), axis=-1)
    np.testing.assert_allclose(actual["mixture_entropy_mean_nats"], entropy(full_w).mean(axis=1), atol=1e-14)
    identical_supports = np.full((5, 19), 2.0)
    identical = analyze_weight_tensor(weights, identical_supports, graph)
    np.testing.assert_allclose(identical["patch_noise_rms_per_coordinate"], 0, atol=1e-14)
    summary = summarize_anchor_metrics(actual)
    assert summary["anchors"] == 5
    assert np.asarray(summary["metrics"]["weight_mean_by_rank"]["mean"]).shape == (3,)
    return {"status": "PASS", "synthetic_only": True,
            "checks": ["uniform", "constant_onehot", "noise_switching_onehot",
                       "centered_gram_vs_direct_patch", "mixture_entropy_identity",
                       "identical_neighbors", "equal_anchor_summary"],
            "max_direct_metric_absolute_error": float(max(direct_errors))}



"""Post-hoc forward diagnostics; NumPy only, no checkpoint/model mutation.

Inputs are the actual Generator features, full trained Dense-head logits and
three-neighbor weights captured by the caller. RMS values use population
moments over the fixed, common Monte Carlo noise bank, not an estimate of a
training gradient. Neighbor ranks are zero-based kNN ranks in ``graph``.
"""

import numpy as np


def compute_stage_diagnostics(features, logits, weights, graph):
    """Return ``(class_scalars, anchor_extra, allowed_logits)``.

    features: (M, D), logits: (M, Nc), weights: (A, M, 3),
    graph: (A, 3) local support-row IDs. Every returned per-anchor value has
    shape (A,). ``allowed_logits`` has shape (A, M, 3), float64.

    Logit contrast removes each three-neighbor mean *at each noise draw*;
    thus it ignores the common-logit shifts to which softmax is invariant.
    All moments are computed in float64. Stored input weights are unchanged.
    """
    f = np.asarray(features, dtype=np.float64)
    ell = np.asarray(logits, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    raw_graph = np.asarray(graph)
    if f.ndim != 2 or ell.ndim != 2 or w.ndim != 3:
        raise ValueError('Expected features (M,D), logits (M,Nc), weights (A,M,3).')
    if raw_graph.ndim != 2 or raw_graph.shape[1] != 3:
        raise ValueError('Expected graph (A,3).')
    if raw_graph.dtype.kind not in 'iu':
        raise ValueError('Graph must contain integer support-row IDs.')
    g = raw_graph.astype(np.int64, copy=False)
    a, m, k = w.shape
    if a < 1 or m < 2 or k != 3 or g.shape != (a, 3):
        raise ValueError('Need at least one anchor, two noise draws and exactly three neighbors.')
    if f.shape[0] != m or ell.shape[0] != m or f.shape[1] < 1 or ell.shape[1] < 3:
        raise ValueError('Noise dimensions or feature/logit dimensions do not match.')
    if np.any(g < 0) or np.any(g >= ell.shape[1]):
        raise ValueError('Graph support-row ID is outside the Dense-head output.')
    if np.any(np.diff(np.sort(g, axis=1), axis=1) == 0):
        raise ValueError('Each anchor must have three distinct neighbors.')
    if not all(np.isfinite(x).all() for x in (f, ell, w)):
        raise ValueError('Actual features/logits/weights must all be finite.')
    if np.any(w < -1e-7) or not np.allclose(w.sum(axis=2), 1.0, rtol=0, atol=2e-6):
        raise ValueError('Weights must be the normalized actual three-neighbor softmax output.')

    allowed = ell[:, g].transpose(1, 0, 2)
    feature_noise = f - f.mean(axis=0, keepdims=True)
    dense_noise = ell - ell.mean(axis=0, keepdims=True)
    allowed_noise = allowed - allowed.mean(axis=1, keepdims=True)
    contrast = allowed - allowed.mean(axis=2, keepdims=True)
    contrast_noise = contrast - contrast.mean(axis=1, keepdims=True)
    pairs = np.stack((allowed[:, :, 0] - allowed[:, :, 1],
                      allowed[:, :, 0] - allowed[:, :, 2],
                      allowed[:, :, 1] - allowed[:, :, 2]), axis=2)
    pair_noise = pairs - pairs.mean(axis=1, keepdims=True)
    ordered = np.sort(allowed, axis=2)
    gaps = ordered[:, :, -1] - ordered[:, :, -2]

    # The primary winner is from actual weights, including any finite-precision
    # ties. Equal maxima use NumPy's first-index rule, like TensorFlow argmax.
    winners = np.argmax(w, axis=2)
    counts = np.stack([(winners == rank).sum(axis=1) for rank in range(3)], axis=1)
    distinct = (counts > 0).sum(axis=1)
    mode_rank = np.argmax(counts, axis=1)
    mode_count = counts.max(axis=1)
    logit_winners = np.argmax(allowed, axis=2)
    logit_distinct = np.stack([(logit_winners == rank).any(axis=1)
                              for rank in range(3)], axis=1).sum(axis=1)

    def rms(x, axis=None):
        return np.sqrt(np.mean(np.square(x), axis=axis))

    anchor = {
        'allowed_logit_noise_rms': rms(allowed_noise, axis=(1, 2)),
        'allowed_logit_contrast_noise_rms': rms(contrast_noise, axis=(1, 2)),
        'pairwise_logit_difference_noise_rms': rms(pair_noise, axis=(1, 2)),
        'top1_top2_logit_gap_noise_mean': gaps.mean(axis=1),
        'top1_top2_logit_gap_noise_median': np.median(gaps, axis=1),
        'top1_top2_logit_gap_noise_min': gaps.min(axis=1),
        'top1_top2_logit_gap_noise_max': gaps.max(axis=1),
        'weight_argmax_stable_across_noise': distinct == 1,
        'weight_argmax_distinct_neighbor_count': distinct,
        'weight_argmax_mode_neighbor_rank_zero_based': mode_rank,
        'weight_argmax_mode_fraction': mode_count / float(m),
        'logit_argmax_stable_across_noise': logit_distinct == 1,
        'logit_argmax_distinct_neighbor_count': logit_distinct,
        'weight_vs_logit_argmax_disagreement_fraction':
            (winners != logit_winners).mean(axis=1),
    }
    feature_rms = float(rms(f))
    feature_noise_rms = float(rms(feature_noise))
    scalars = {
        'noise_draws': int(m),
        'anchors': int(a),
        'feature_dimensions': int(f.shape[1]),
        'dense_head_supports': int(ell.shape[1]),
        'feature_noise_rms_per_coordinate': feature_noise_rms,
        'feature_rms_per_coordinate': feature_rms,
        'feature_noise_rms_fraction_of_feature_rms':
            feature_noise_rms / feature_rms if feature_rms > 0 else float('nan'),
        'dense_logit_noise_rms_per_coordinate': float(rms(dense_noise)),
        'dense_logit_rms_per_coordinate': float(rms(ell)),
        'allowed_logit_noise_rms_over_anchors_noise_neighbors': float(rms(allowed_noise)),
        'allowed_logit_contrast_noise_rms_over_anchors_noise_neighbors': float(rms(contrast_noise)),
        'pairwise_logit_difference_noise_rms_over_anchors_noise_pairs': float(rms(pair_noise)),
        'top1_top2_logit_gap_mean_over_anchors_and_noise': float(gaps.mean()),
        'top1_top2_logit_gap_median_over_anchors_and_noise': float(np.median(gaps)),
        'top1_top2_logit_gap_min_over_anchors_and_noise': float(gaps.min()),
        'top1_top2_logit_gap_max_over_anchors_and_noise': float(gaps.max()),
        'weight_argmax_stable_anchor_fraction': float(np.mean(distinct == 1)),
        'weight_argmax_distinct_neighbor_count_anchor_mean': float(distinct.mean()),
        'weight_argmax_mode_fraction_anchor_mean': float(np.mean(mode_count / float(m))),
        'logit_argmax_stable_anchor_fraction': float(np.mean(logit_distinct == 1)),
        'weight_vs_logit_argmax_disagreement_fraction': float(np.mean(winners != logit_winners)),
    }
    for key in ('allowed_logit_noise_rms', 'allowed_logit_contrast_noise_rms',
                'pairwise_logit_difference_noise_rms'):
        scalars[key + '_anchor_mean'] = float(anchor[key].mean())
        scalars[key + '_anchor_median'] = float(np.median(anchor[key]))
    return scalars, anchor, allowed

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def run_stage_unit_checks():
    checks = []
    graph = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    f = np.tile(np.array([2.0, -3.0]), (4, 1))
    ell = np.tile(np.array([4.0, 3.0, 1.0, 0.0]), (4, 1))
    w = softmax(ell[:, graph].transpose(1, 0, 2)).astype(np.float32)
    original_w = w.copy()
    s, a, selected = compute_stage_diagnostics(f, ell, w, graph)
    assert s['feature_noise_rms_per_coordinate'] == 0
    assert np.isclose(s['feature_rms_per_coordinate'], np.sqrt(6.5))
    assert s['dense_logit_noise_rms_per_coordinate'] == 0
    assert s['allowed_logit_contrast_noise_rms_over_anchors_noise_neighbors'] == 0
    assert np.array_equal(selected, ell[:, graph].transpose(1, 0, 2))
    assert np.array_equal(w, original_w)
    assert np.array_equal(a['weight_argmax_distinct_neighbor_count'], [1, 1])
    checks.append('Constant nonzero features/logits/weights have zero noise RMS; weights unchanged.')
    shifts = np.array([-4.0, -1.0, 1.0, 4.0])
    shifted = ell + shifts[:, None]
    s2, a2, _ = compute_stage_diagnostics(f, shifted, w, graph)
    assert np.isclose(s2['dense_logit_noise_rms_per_coordinate'], np.sqrt(8.5))
    assert np.isclose(s2['allowed_logit_noise_rms_over_anchors_noise_neighbors'], np.sqrt(8.5))
    assert np.isclose(s2['allowed_logit_contrast_noise_rms_over_anchors_noise_neighbors'], 0, atol=1e-14)
    assert np.isclose(s2['pairwise_logit_difference_noise_rms_over_anchors_noise_pairs'], 0, atol=1e-14)
    for key in a:
        if key == 'allowed_logit_noise_rms':
            continue
        assert np.allclose(a2[key], a[key]), key
    checks.append('Noise-varying common-logit shifts are removed by contrast and pairwise metrics.')
    f3 = np.array([[-1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [1.0, 2.0]])
    ell3 = np.zeros((4, 4))
    ell3[:, 0] = [-1, 1, -1, 1]
    w3 = softmax(ell3[:, graph].transpose(1, 0, 2))
    s3, a3, _ = compute_stage_diagnostics(f3, ell3, w3, graph)
    assert np.isclose(s3['feature_noise_rms_per_coordinate'], np.sqrt(2.5))
    assert np.isclose(s3['feature_noise_rms_fraction_of_feature_rms'], 1.0)
    assert np.isclose(a3['allowed_logit_noise_rms'][0], np.sqrt(1 / 3))
    assert np.isclose(a3['allowed_logit_contrast_noise_rms'][0], np.sqrt(2) / 3)
    assert np.isclose(a3['pairwise_logit_difference_noise_rms'][0], np.sqrt(2 / 3))
    assert a3['allowed_logit_contrast_noise_rms'][1] == 0
    assert np.array_equal(a3['weight_argmax_distinct_neighbor_count'], [2, 1])
    assert np.array_equal(a3['weight_argmax_mode_neighbor_rank_zero_based'], [0, 0])
    assert np.allclose(a3['top1_top2_logit_gap_noise_mean'], [0.5, 0.0])
    assert all((value.shape == (2,) for value in a3.values()))
    checks.append('Constructed support contrast, winner changes, feature units and direct gap statistics match analytic values.')
    perm = np.array([2, 0, 1])
    s4, a4, allowed4 = compute_stage_diagnostics(f3, ell3, w3[:, :, perm], graph[:, perm])
    assert np.allclose(allowed4, ell3[:, graph[:, perm]].transpose(1, 0, 2))
    for key in ('allowed_logit_noise_rms', 'allowed_logit_contrast_noise_rms', 'pairwise_logit_difference_noise_rms', 'top1_top2_logit_gap_noise_mean'):
        assert np.allclose(a4[key], a3[key]), key
    checks.append('Scalar contrast/gap summaries are invariant to a consistent neighbor-rank permutation.')
    invalid = [np.array([[0, 0, 1], [1, 2, 3]]), np.array([[0, 1, 4], [1, 2, 3]]), graph.astype(float)]
    for bad in invalid:
        try:
            compute_stage_diagnostics(f, ell, w, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid graph was accepted.')
    checks.append('Duplicate/out-of-range/noninteger graph entries are rejected.')
    result = {'passed': True, 'check_count': len(checks), 'checks': checks, 'scope': 'Pure NumPy analytical fixtures only; no TensorFlow or checkpoint forward.'}
    return result


import datetime as dt
import gc, importlib.metadata as package_metadata, os, platform, time, traceback, uuid
CLASS_NAMES = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers']

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')

def load_probe_runtime():
    """Configure before tensor creation; never install or substitute Keras 3."""
    global tf, keras
    require('tensorflow' not in sys.modules and 'tf_keras' not in sys.modules,
            'TensorFlow is already imported. Restart the Kaggle session and Run All this notebook first.')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tfv = package_metadata.version('tensorflow')
    kv = package_metadata.version('tf-keras')
    require(tfv.split('.')[:2] == kv.split('.')[:2] == ['2', '20'],
            f'Expected TensorFlow 2.20.x and tf-keras 2.20.x; found {tfv}/{kv}. No architecture fallback or automatic install.')
    import tensorflow as tensorflow
    import tf_keras as oldkeras
    tf, keras = tensorflow, oldkeras
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[:1], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.enable_op_determinism()
    keras.mixed_precision.set_global_policy('float32')
    keras.backend.set_floatx('float32')
    keras.utils.set_random_seed(20261016)  # Initializers are overwritten by verified checkpoint arrays.
    return {
        'python': platform.python_version(), 'numpy': np.__version__,
        'tensorflow': tf.__version__, 'tf_keras': keras.__version__,
        'device_mode': 'one GPU' if gpus else 'CPU (same probe budget)',
        'physical_visible_gpu': [p.name for p in tf.config.get_visible_devices('GPU')],
        'float_policy': keras.mixed_precision.global_policy().name,
        'deterministic_ops_requested': True,
        'historical_training_forward_bitwise_replay_claimed': False,
    }

def model_state_hashes(G, heads):
    return {key: sha_array(variable.numpy())
            for key, variable in _v2_destination_variables(G, heads).items()}

def actual_neighbor_weights(G, head, graph, noises, head_index, batch_size):
    """Actual restored G and Dense; use exact source gather + TF softmax."""
    labels = np.zeros((len(noises), 16), np.float32)
    labels[:, head_index] = 1  # SORTED rank h, not original class_id-1.
    features = []
    for start in range(0, len(noises), batch_size):
        stop = start + batch_size
        features.append(G([noises[start:stop], labels[start:stop]], training=False).numpy())
    features = np.concatenate(features)
    require(features.shape == (len(noises), 784) and np.isfinite(features).all(),
            'Invalid actual Generator features')
    logits = head.get_layer('support_logits')(tf.convert_to_tensor(features), training=False)
    require(tuple(logits.shape) == (len(noises), len(graph)), 'Invalid actual Dense output')
    require(np.isfinite(logits.numpy()).all(), 'Nonfinite restored Dense logits')
    # tf.gather gives M,A,3; source local_mix uses the same 3 logits per row.
    allowed_logits = tf.gather(logits, graph, axis=1)
    weights = tf.transpose(tf.nn.softmax(allowed_logits, axis=-1), [1, 0, 2]).numpy()
    require(weights.shape == (len(graph), len(noises), 3), 'Weight dimensions changed')
    return weights, features, logits.numpy()

def verify_head_route(head, features, weights, support, graph, metrics, batch_size):
    """Fixed source-head checks; no adaptive selection of anchors/noises."""
    anchor_ids = sorted(set([0, len(support)//2, len(support)-1]))
    flat = support.reshape(len(support), -1)
    records = []
    for anchor in anchor_ids:
        ids = np.full((len(features), 1), anchor, np.int32)
        outputs = []
        for start in range(0, len(features), batch_size):
            stop = start + batch_size
            outputs.append(head([features[start:stop], ids[start:stop]], training=False).numpy())
        actual = np.concatenate(outputs).reshape(len(features), -1)
        expected = (np.float32(.8)*flat[anchor] + np.float32(.2)*np.einsum(
            'mj,jp->mp', weights[anchor], flat[graph[anchor]], optimize=False))
        require(np.isfinite(actual).all(), 'Nonfinite actual source-head patch')
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6,
                                   err_msg='Actual source-head patch differs from recovered weighting formula')
        # The Gram result describes ideal convex arithmetic. Check directly
        # rendered actual float32 patches too, allowing fixed rounding only.
        direct_noise_rms = float(np.sqrt(np.mean((actual.astype(np.float64)-actual.mean(axis=0, dtype=np.float64))**2)))
        gram_rms = float(metrics['patch_noise_rms_per_coordinate'][anchor])
        np.testing.assert_allclose(direct_noise_rms, gram_rms, rtol=5e-4, atol=3e-6,
                                   err_msg='Actual output noise RMS differs from Gram weighting metric')
        records.append({
            'anchor_local_id': anchor, 'noises_checked': len(features),
            'head_patch_max_abs_error': float(np.max(np.abs(actual-expected))),
            'direct_head_patch_noise_rms': direct_noise_rms,
            'gram_patch_noise_rms': gram_rms,
            'noise_rms_absolute_error': abs(direct_noise_rms-gram_rms),
            'patch_allclose_rtol': 2e-6, 'patch_allclose_atol': 2e-6,
            'noise_rms_rtol': 5e-4, 'noise_rms_atol': 3e-6,
            'status': 'PASS',
        })
    return records

def package_report(out):
    # No checkpoint/train tensors or scratch copies go into the report archive.
    payloads = sorted(p for p in out.rglob('*') if p.is_file())
    write_json(out/'output_file_hashes.json', {
        str(p.relative_to(out)): {'bytes':p.stat().st_size,'sha256':sha_file(p)}
        for p in payloads if p.name != 'output_file_hashes.json'})
    target=out.with_suffix('.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file(): archive.write(path,str(path.relative_to(out)))
    with zipfile.ZipFile(target) as archive:
        require(archive.testzip() is None, 'Output report ZIP CRC check failed')
    return target

"""Six-checkpoint endgame audit. Inference only; all inputs are pinned."""

def discover_endgame_checkpoints(refs, destination, out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    candidates={key:[] for key in ENDGAME_PINS}
    rejected=[]
    for ref in refs:
        try:
            meta=ref.read_json()
            c=meta.get('contract',{})
            if c.get('arm')!='learned_local': continue
            key=(c.get('seed'),meta.get('c_updates'))
            if key not in candidates: continue
            _v2_check_metadata(meta)
            pins=ENDGAME_PINS[key]
            require(ref.sha()==pins['meta.json'],'Metadata SHA differs from original M4: '+ref.label())
            require(meta['weights_hashes']=={n:pins[n] for n in ('weights.index','weights.data-00000-of-00001')},
                    'Checkpoint declared file hashes differ from original M4')
            for name in ('weights.index','weights.data-00000-of-00001'):
                payload=ref.sibling(name)
                require(payload.exists(),'Missing sibling: '+payload.label())
                require(payload.sha()==pins[name],'Payload SHA mismatch: '+payload.label())
            candidates[key].append((ref,meta))
        except Exception as exc:
            rejected.append({'source':ref.label(),'error':str(exc)})
    selected={}; registry=[]
    for (seed,updates), copies in sorted(candidates.items()):
        row={'seed':seed,'c_updates':updates,'valid_copies':len(copies),'status':'MISSING_OR_REJECTED'}
        if copies:
            copies.sort(key=lambda v:(v[0].member is not None,v[0].label()))
            ref,meta=copies[0]
            folder=Path(destination)/f'seed_{seed}_u_{updates}'
            paths={name:ref.sibling(name).materialize(folder,name)
                   for name in ('weights.index','weights.data-00000-of-00001','meta.json')}
            selected[seed,updates]={'meta':meta,'folder':str(paths['weights.index'].parent),
                'source':ref.label(),'pins':ENDGAME_PINS[seed,updates]}
            row.update(status='VERIFIED',metadata_source=ref.label(),folder=selected[seed,updates]['folder'])
        registry.append(row)
    pd.DataFrame(registry).to_csv(out/'checkpoint_registry.csv',index=False)
    write_json(out/'checkpoint_discovery.json',{'registry':registry,'rejected':rejected})
    missing=[key for key in ENDGAME_PINS if key not in selected]
    require(not missing,'Missing verified checkpoints '+str(missing)+
        '. Add M4_LEARNED_LOCAL_3SEEDS_CKPT07500.zip as another Kaggle Input; retain all existing 8000/B cache inputs. '+
        'Both unpacked folders and ZIP are supported. Inspect checkpoint_registry.csv and checkpoint_discovery.json.')
    return selected


def endgame_figures(out, table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields=[('feature_noise_rms','G feature noise RMS'),
        ('contrast_logit_noise_rms_mean','Local contrast-logit noise RMS'),
        ('top1_top2_gap_mean','Top-1 / top-2 logit gap'),
        ('entropy_mean_nats','Conditional neighbor entropy'),
        ('weight_noise_rms_l2','Weight noise RMS L2'),
        ('patch_noise_rms_per_coordinate','Patch noise RMS / PCA coordinate')]
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for ax,(field,label) in zip(axes.flat,fields):
        for seed in (42,43,44):
            d=table[table.model_seed==seed].sort_values('c_updates')
            ax.plot(d.c_updates,d[field],marker='o',label=f'Seed {seed}')
        ax.set_xticks([7500,8000]); ax.set_xlabel('C updates'); ax.set_title(label,fontsize=10)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3)); ax.legend(fontsize=8)
    fig.suptitle('Train-only endpoint comparison; equal-class means; stage units differ')
    folder=out/'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder/'six_stage_7500_8000.png',dpi=160); plt.close(fig)


def summarize_endgame(out, manifest):
    class_frames=[]; anchor_frames=[]
    for seed in (42,43,44):
        for updates in (7500,8000):
            folder=out/f'seed_{seed}_u_{updates}'
            require((folder/'complete.json').is_file(),f'Checkpoint {seed}/{updates} incomplete')
            class_frames.append(pd.read_csv(folder/'per_class.csv'))
            anchor_frames.append(pd.read_csv(folder/'per_anchor.csv'))
    classes=pd.concat(class_frames,ignore_index=True)
    anchors=pd.concat(anchor_frames,ignore_index=True)
    require(len(classes)==90 and len(anchors)==24828,'Incomplete class/anchor registry')
    classes.to_csv(out/'per_class_7500_8000.csv',index=False)
    anchors.to_csv(out/'per_anchor_7500_8000.csv',index=False)
    metrics=['feature_noise_rms','full_logit_noise_rms','contrast_logit_noise_rms_mean',
        'top1_top2_gap_mean','entropy_mean_nats','effective_neighbors_mean','maximum_weight_mean',
        'weight_noise_rms_l2','noise_jsd_nats','patch_noise_rms_per_coordinate','stable_argmax_fraction']
    macro=classes.groupby(['model_seed','c_updates'])[metrics].mean().reset_index()
    macro.to_csv(out/'per_seed_equal_class_summary.csv',index=False)
    comparisons=[]; paired_anchor_frames=[]
    for seed in (42,43,44):
        for cid in sorted(classes.class_id.unique()):
            paths=[out/f'seed_{seed}_u_{u}'/'raw'/f'class_{cid:02d}.npz' for u in (7500,8000)]
            with np.load(paths[0],allow_pickle=False) as d0,np.load(paths[1],allow_pickle=False) as d1:
                require(np.array_equal(d0['anchor_train_rows'],d1['anchor_train_rows']) and
                        np.array_equal(d0['graph'],d1['graph']) and
                        np.array_equal(d0['noise'],d1['noise']), 'Unpaired checkpoint observations')
                w0=d0['weights'].astype(np.float64); w1=d1['weights'].astype(np.float64)
                winners0=w0.argmax(-1); winners1=w1.argmax(-1)
                rows=d0['anchor_train_rows'].copy()
                pairdf=pd.DataFrame({'model_seed':seed,'class_id':cid,'anchor_local_id':np.arange(len(w0)),
                    'train_row_id':rows,'paired_noise_argmax_agreement':np.mean(winners0==winners1,axis=1),
                    'weight_L1_difference_mean':np.abs(w1-w0).sum(-1).mean(axis=1)})
                paired_anchor_frames.append(pairdf)
                summary={'model_seed':seed,'class_id':int(cid),'anchors':len(rows),
                         'delta_definition':'8000 minus 7500',
                         'paired_noise_argmax_agreement':float(np.mean(winners0==winners1)),
                         'weight_L1_difference_mean':float(np.abs(w1-w0).sum(-1).mean())}
            left=classes[(classes.model_seed==seed)&(classes.c_updates==7500)&(classes.class_id==cid)].iloc[0]
            right=classes[(classes.model_seed==seed)&(classes.c_updates==8000)&(classes.class_id==cid)].iloc[0]
            for metric in metrics:
                summary[metric+'__7500']=float(left[metric])
                summary[metric+'__8000']=float(right[metric])
                summary[metric+'__delta']=float(right[metric]-left[metric])
            comparisons.append(summary)
    pd.DataFrame(comparisons).to_csv(out/'delta_7500_to_8000_by_class.csv',index=False)
    pd.concat(paired_anchor_frames,ignore_index=True).to_csv(out/'paired_anchor_comparison.csv',index=False)
    endgame_figures(out,macro)
    lines=['# V2 endgame audit: 7.500 → 8.000', '',
        'Hoàn tất đủ 6 checkpoint: Learned-local V2, Protocol B `50e988fae260`.', '',
        'Các dòng dưới là trung bình đều 15 lớp có head, đều anchor trong từng lớp. Mỗi checkpoint dùng cùng 4.138 anchor và 64 noise/lớp.', '',
        '| Seed | Updates | Feature RMS | Contrast logit RMS | Gap mean | Entropy | Weight RMS | Patch RMS |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _,r in macro.iterrows():
        fields=['feature_noise_rms','contrast_logit_noise_rms_mean','top1_top2_gap_mean',
                'entropy_mean_nats','weight_noise_rms_l2','patch_noise_rms_per_coordinate']
        lines.append('| '+str(int(r.model_seed))+' | '+str(int(r.c_updates))+' | '+
                     ' | '.join(f'{r[f]:.7g}' for f in fields)+' |')
    lines += ['', '## Implementation và kiểm toàn vẹn', '',
        '- Dùng nguyên hàm G, head, kNN graph của M4. Dữ liệu final-train/PCA provenance và graph hashes khớp cả sáu contract.',
        '- Pin metadata/index/data của sáu checkpoint gốc. Kiểm 50 native G/head arrays, shapes/dtypes/semantic hashes, rồi hash lại sau gán và sau forward.',
        '- Dùng Dense layer thật và tf.nn.softmax trên đúng ba neighbor. Đối chiếu patch head thật ở anchor đầu/giữa/cuối, toàn bộ noise.',
        '- Chỉ training=False, không C/D/optimizer/gradient tape, không train/test/validation. Đây là tuyến inference sinh fake cho C, không tái hiện bước tối ưu G training-mode BN.', '',
        '## Định nghĩa và giới hạn', '',
        '- Feature/full-logit/local-logit RMS là biến thiên trên bộ noise cố định. Các tầng khác đơn vị và số chiều: không lấy tỷ số RMS khác tầng để gọi là gain hoặc mức gradient.',
        '- Contrast logits trừ mean của ba neighbor tại từng noise trước khi đo biến thiên. Common offset làm logits thay đổi nhưng không đổi softmax; metric này loại thành phần đó.',
        '- Weight statistics tính mean/variance trong float64 từ xác suất float32 thật, tránh noise RMS giả do mean float32.',
        '- Patch RMS dùng Gram ổn định của neighbor; kiểm với head float32 có dung sai ghi trong actual_head_checks.json. RMS rất nhỏ chưa chứng minh thay đổi đáng kể của patch thực.',
        '- So argmax giữa checkpoint tại đúng cặp anchor/noise, không chỉ so modal rank. Lưu raw noise, G features, Dense logits, allowed logits, weights, graph, global row IDs để kiểm lại.',
        '- Nếu cả 7.500 và 8.000 đều tập trung, chỉ kết luận hiện tượng đã có tại hai mốc đó trong probe; không biết bắt đầu từ bước nào, không chứng minh ổn định suốt khoảng giữa hoặc đơn điệu.',
        '- Nếu hai mốc khác, đó là khác biệt cuối training; không xác định nguyên nhân, usefulness, collapse, gradient lịch sử hoặc nguyên nhân seed43 kém.',
        '- Không chọn k/rho/temperature/metric/kiến trúc từ báo cáo này. Lớp đa số11 không có head, xem class_mapping.csv.', '',
        '## File chính', '',
        '`per_class_7500_8000.csv`, `per_seed_equal_class_summary.csv`, `delta_7500_to_8000_by_class.csv`, `paired_anchor_comparison.csv`, `figures/` và `seed_*_u_*/raw/class_*.npz`.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return macro


def run_endgame(config):
    started=time.monotonic()
    tag=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=Path('/kaggle/working')/('V2_ENDGAME_AUDIT_7500_8000_'+tag)
    scratch=Path('/kaggle/working')/('_v2_endgame_inputs_'+tag)
    out.mkdir(parents=True,exist_ok=False)
    manifest={'status':'RUNNING','complete':False,'protocol_B_id':PROTOCOL_ID,
        'arm':'learned_local','seeds':[42,43,44],'updates':[7500,8000],'config':config,
        'completed_checkpoints':[],'required_checkpoint_count':6,'source_worker_sha256':M4_WORKER_SHA,
        'exact_function_sha256':EXACT_FUNCTION_SHA,'repaired_user_code_sha256':USER_CODE_SHA,
        'training_performed':False,'classifier_constructed':False,'test_loaded':False,
        'validation_loaded':False,'architecture_fallback_used':False,'forward_mode':'training=False'}
    error=None; macro=None
    try:
        require(config['model_seeds']==[42,43,44] and config['updates']==[7500,8000],'Required six-checkpoint plan changed')
        require(config['noises_per_class']==64,'Locked probe uses 64 noises/class across all checkpoints')
        write_json(out/'metric_checks.json',{'base':run_synthetic_checks(),'stage':run_stage_unit_checks()})
        refs,trains,issues=inventory_input_refs(config['input_roots'])
        write_json(out/'input_inventory.json',{'metadata_sources':[r.label() for r in refs],
            'final_train_sources':[r.label() for r in trains],'issues':issues})
        selected=discover_endgame_checkpoints(refs,scratch/'checkpoints',out)
        print('Đủ 6 checkpoint đúng hash/metadata; chưa chạy forward.',flush=True)
        X,y,coords,provenance=load_final_train_verified(trains,scratch/'train')
        write_json(out/'train_provenance.json',provenance)
        prepared=build_graph_per_class(X,y)
        graph_hashes=[sha_array(g) for g in prepared['neighbor_graphs']]
        require(all(graph_hashes==v['meta']['contract']['graph_sha256'] for v in selected.values()),
                'Rebuilt train-only graph differs from one or more checkpoint contracts')
        write_json(out/'graph_verification.json',{'status':'PASS','graph_sha256':graph_hashes,
                  'verified_checkpoint_count':6})
        pd.DataFrame([{'class_id':int(c),'class_name':CLASS_NAMES[c-1],'sorted_rank':h,
            'train_supports':int(prepared['counts'][c-1]),
            'status':'LEARNED_HEAD' if h<15 else 'NOT_APPLICABLE_NO_HEAD'}
            for h,c in enumerate(prepared['order']+1)]).to_csv(out/'class_mapping.csv',index=False)
        noises=np.stack([np.random.default_rng(np.random.SeedSequence([config['analysis_seed'],int(c)]))
            .normal(size=(64,100)).astype(np.float32) for c in prepared['order'][:15]+1])
        np.savez_compressed(out/'common_probe_inputs.npz',noise_by_head=noises,
                            class_order=prepared['order']+1,train_labels=y,train_coords=coords)
        manifest['common_noise_sha256']=sha_array(noises)
        manifest['runtime']=load_probe_runtime()
        tensors_all={}
        for key,item in selected.items():
            tensors,audit=preflight_checkpoint_tensors(Path(item['folder']),item['meta'])
            tensors_all[key]=tensors
            write_json(out/f'seed_{key[0]}_u_{key[1]}'/'native_tensor_verification.json',audit)
        for (seed,updates),item in sorted(selected.items()):
            folder=out/f'seed_{seed}_u_{updates}'; folder.mkdir(exist_ok=True)
            print(f'Seed {seed}, C={updates}: restore và forward train-only.',flush=True)
            keras.backend.clear_session(); keras.utils.set_random_seed(20261016)
            G,heads,audit=restore_generator_and_heads(prepared,item['meta'],tensors_all.pop((seed,updates)))
            write_json(folder/'restore_verification.json',audit)
            before=model_state_hashes(G,heads)
            class_rows=[]; anchor_frames=[]; head_checks=[]
            try:
                raw_folder=folder/'raw'; raw_folder.mkdir(exist_ok=True)
                for h,(support,graph) in enumerate(zip(prepared['supports'],prepared['neighbor_graphs'])):
                    c=int(prepared['order'][h]+1); rows=np.flatnonzero(y==c-1)
                    weights,features,logits=actual_neighbor_weights(G,heads[h],graph,noises[h],h,config['batch_size'])
                    base=analyze_weight_tensor(weights,support.reshape(len(support),-1),graph,rho=.2)
                    stages,extra,allowed=compute_stage_diagnostics(features,logits,weights,graph)
                    stages.update(
                        feature_noise_rms=stages['feature_noise_rms_per_coordinate'],
                        full_logit_noise_rms=stages['dense_logit_noise_rms_per_coordinate'],
                        contrast_logit_noise_rms_mean=stages['allowed_logit_contrast_noise_rms_anchor_mean'],
                        top1_top2_gap_mean=stages['top1_top2_logit_gap_mean_over_anchors_and_noise'],
                        stable_argmax_fraction=stages['weight_argmax_stable_anchor_fraction'])
                    checks=verify_head_route(heads[h],features,weights,support,graph,base,config['batch_size'])
                    head_checks.append({'class_id':c,'head_index':h,'checks':checks})
                    common={'model_seed':seed,'c_updates':updates,'class_id':c,'class_name':CLASS_NAMES[c-1],
                            'head_index_sorted':h,'anchors':len(support),'noises_per_anchor':64}
                    frame=pd.DataFrame({k:np.repeat(v,len(support)) for k,v in common.items()})
                    frame['anchor_local_id']=np.arange(len(support)); frame['train_row_id']=rows
                    record={**common,**stages}
                    for name,values in {**base,**extra}.items():
                        values=np.asarray(values)
                        if values.ndim==1:
                            frame[name]=values
                            record[name]=float(values.mean())
                            record[name+'__median_across_anchors']=float(np.median(values))
                    anchor_frames.append(frame); class_rows.append(record)
                    np.savez_compressed(raw_folder/f'class_{c:02d}.npz',weights=weights,
                        G_features=features,actual_dense_logits=logits,allowed_logits=allowed.astype(np.float32),
                        graph=graph,anchor_train_rows=rows,noise=noises[h],class_id=np.int64(c),
                        model_seed=np.int64(seed),c_updates=np.int64(updates))
                    print(f'  class {c:2d}: H={record["entropy_mean_nats"]:.3e}; head PASS',flush=True)
                pd.DataFrame(class_rows).to_csv(folder/'per_class.csv',index=False)
                pd.concat(anchor_frames,ignore_index=True).to_csv(folder/'per_anchor.csv',index=False)
                write_json(folder/'actual_head_checks.json',head_checks)
            finally:
                after=model_state_hashes(G,heads)
                write_json(folder/'nonmutation_verification.json',{'status':'PASS' if before==after else 'FAIL',
                    'array_count':50,'before':before,'after':after,'unchanged':before==after})
                require(before==after,'G/head trainable weights or BN buffers changed during forward')
            actual={name:sha_file(Path(item['folder'])/name) for name in item['pins']}
            require(actual==item['pins'],'Checkpoint bytes changed after forward')
            write_json(folder/'checkpoint_bytes_after.json',actual)
            write_json(folder/'complete.json',{'status':'PASS','seed':seed,'c_updates':updates,
                'heads':15,'anchors':4138,'noise_sha256':manifest['common_noise_sha256']})
            manifest['completed_checkpoints'].append([seed,updates])
            write_json(out/'manifest.json',manifest)
            del G,heads; keras.backend.clear_session(); gc.collect()
        require(len(manifest['completed_checkpoints'])==6,'Six-checkpoint registry incomplete')
        macro=summarize_endgame(out,manifest)
        manifest['status']='COMPLETE_6_CHECKPOINTS'; manifest['complete']=True
    except Exception as exc:
        error=exc; manifest.update(status='INCOMPLETE_FAILED_CHECK',complete=False,error_type=type(exc).__name__,error=str(exc))
        (out/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        (out/'REPORT.md').write_text('# CHƯA HOÀN TẤT\n\n'+str(exc)+
            '\n\nKhông dùng output dở dang làm kết quả sáu checkpoint. Xem registry/inventory/ERROR.\n',encoding='utf-8')
    finally:
        manifest['elapsed_seconds']=round(time.monotonic()-started,3)
        write_json(out/'manifest.json',manifest)
        target=package_report(out)
        print('\nSTATUS:',manifest['status'],'\nZIP:',target,flush=True)
        try:
            from IPython.display import FileLink,display
            display(FileLink(str(target)))
            if macro is not None: display(macro)
        except ImportError: pass
    if error is not None: raise RuntimeError('Audit incomplete; diagnostic ZIP saved. '+str(error)) from error
    return {'output_directory':str(out),'report_zip':str(target),'manifest':manifest}

