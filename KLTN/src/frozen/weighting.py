from pathlib import Path, PurePosixPath
import hashlib, json, sys, zipfile
import numpy as np

CONFIG = {
    "input_roots": ["/kaggle/input"],
    "model_seeds": [42, 43, 44],
    "analysis_seed": 20261016,
    "noises_per_class": 64,
    "batch_size": 32,
}
# Không import TensorFlow ở đây: cần cấu hình visibility/memory growth trước.
tf = keras = None


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
CHECKPOINT_SHA = {
    42: {"weights.index":"3767710c97ee62f2f2fb3b2470f85f8bb2587c71044b7a0a3d8babb243317a7b", "weights.data-00000-of-00001":"5af9bd48710d98a9a883309506a8cba4365929713927b9aa78ed34842e471415"},
    43: {"weights.index":"b96d463087f8ea2dbcf6b1631a293b00d752b8313bf9bb681790ca9fb3ccb9c3", "weights.data-00000-of-00001":"97cc8b798f58601872de02cbc3b396a4514821d40f0561b898eec5213fb32cdb"},
    44: {"weights.index":"1847a904e5286a56a888f7300f06f17a73e1958fb88103a9cfe0b138aa63c976", "weights.data-00000-of-00001":"8fa310ec1f4c33e397735afe6b13ef4602c0a08c4e696895335fee783a1b0942"},
}
METADATA_SHA = {
    42: "cc0f0b8817ea074b30ed3d044cb96f3069c6d591fc4eb345d5d0012880431e42",
    43: "1f9478c8339266746a8218f70b99da4a1948c866fffe2197a011234d77cfc9eb",
    44: "678b69d03feb3c9fabe03e2567c41abc08c961a2a4f82e1bd3c9e96b38105674",
}


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


def validate_checkpoint_metadata(meta):
    c = meta.get("contract", {})
    expected = {"protocol_B_id": PROTOCOL_ID, "implementation_B": IMPLEMENTATION_B,
                "stage": "final", "arm": "learned_local", "budget_C_updates": 8000,
                "rho": .2, "k": 3, "sha_worker": M4_WORKER_SHA,
                "train_X_sha256": TRAIN_X_SHA, "train_y_sha256": TRAIN_Y_SHA,
                "class_order": CLASS_ORDER}
    for key, value in expected.items():
        require(c.get(key) == value, f"Checkpoint contract {key}: expected {value!r}, got {c.get(key)!r}")
    seed = c.get("seed")
    require(seed in (42,43,44), f"Unexpected seed: {seed}")
    require(meta.get("schema") == 1, "Unexpected checkpoint schema")
    require(meta.get("c_updates") == 8000 and meta.get("next_loop") == 4000,
            "Checkpoint is not final 8000 C updates / 4000 loops")
    require(meta.get("optimizer_clock") == 136000, "Optimizer clock inconsistent with M4")
    require(meta.get("weights_hashes") == CHECKPOINT_SHA[seed], "Checkpoint payload declaration differs from independently verified original")
    require(len(c.get("graph_sha256", [])) == 15, "Missing 15 support graph digests")
    require(sum(key.startswith("trainable/G/") for key in meta.get("state_hashes", {})) == 12,
            "Missing 12 Generator trainable state hashes")
    return seed


def discover_checkpoint_inputs(meta_refs, destination):
    valid = {42: [],43: [],44: []}
    rejected = []
    for ref in meta_refs:
        try:
            meta = ref.read_json()
            # Do not mistake a baseline/other job for a candidate just because
            # its path contains seed42 or its TensorFlow files share names.
            c = meta.get("contract", {})
            if c.get("arm") != "learned_local":
                continue
            seed = validate_checkpoint_metadata(meta)
            require(ref.sha() == METADATA_SHA[seed], "Metadata byte hash differs from independently verified original")
            for name, expected in CHECKPOINT_SHA[seed].items():
                payload = ref.sibling(name)
                require(payload.exists(), f"Missing sibling {payload.label()}")
                require(payload.sha() == expected, f"Checkpoint payload SHA256 mismatch: {payload.label()}")
            valid[seed].append((ref,meta))
        except Exception as error:
            rejected.append({"source": ref.label(), "reason": str(error)})
    selected = {}
    for seed, candidates in valid.items():
        if not candidates:
            continue
        # Prefer direct input paths to avoid needless copies, then stable path.
        candidates.sort(key=lambda x: (x[0].member is not None, x[0].label()))
        ref,meta = candidates[0]
        folder = Path(destination) / f"seed_{seed}"
        paths = {name: ref.sibling(name).materialize(folder,name)
                 for name in ("weights.index","weights.data-00000-of-00001","meta.json")}
        selected[seed] = {"prefix": str(paths["weights.index"].with_suffix("")),
                          "meta": meta, "metadata_source": ref.label(),
                          "metadata_sha256": ref.sha(), "identical_copy_count":len(candidates),
                          "payload_sha256": CHECKPOINT_SHA[seed]}
    return selected, rejected


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


SHAPE = (11, 11, 40, 1)
FLAT = 4840
N_CLASSES = 16
K = 3
RHO = 0.2
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
# The initial newline is excluded from the pinned source string.
M4_EXACT_SOURCE = M4_EXACT_SOURCE.lstrip("\n")
require(hashlib.sha256(M4_EXACT_SOURCE.encode()).hexdigest() == '9c008fe3edb05daba4f8839479b014662bde81b20842796cf11bcd6dc41169a9',
        "Embedded M4 function source changed")
exec(compile(M4_EXACT_SOURCE, "<original_M4_G_head_graph>", "exec"), globals())


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
    for key, value in (('c_updates', 8000), ('next_loop', 4000),
                        ('optimizer_clock', 136000), ('audit_calls', 136000)):
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


"""Kaggle train-only endpoint probe. Embedded into the delivered notebook."""
import datetime as dt
import gc
import importlib.metadata as package_metadata
import os
import platform
import time
import traceback
import uuid

CLASS_NAMES = [
    'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
    'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats',
    'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods',
    'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers',
]


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


def save_class_tables(folder, seed, class_id, h, weights, metrics, graph, rows, coords):
    import pandas as pd
    folder.mkdir(parents=True, exist_ok=True)
    common = {'model_seed': seed, 'class_id': class_id, 'class_name': CLASS_NAMES[class_id-1],
              'head_index_sorted': h, 'anchors': len(graph), 'noises_per_anchor': weights.shape[1]}
    scalar_metrics = {key: value for key, value in metrics.items() if np.asarray(value).ndim == 1}
    anchor_table = pd.DataFrame({key: np.repeat(value, len(graph)) for key, value in common.items()})
    anchor_table['anchor_local_id'] = np.arange(len(graph))
    anchor_table['train_row_id'] = rows
    anchor_table['row_coordinate'] = coords[rows, 0]
    anchor_table['col_coordinate'] = coords[rows, 1]
    for key, value in scalar_metrics.items():
        anchor_table[key] = value
    for rank in range(3):
        anchor_table[f'neighbor_{rank+1}_local_id'] = graph[:, rank]
        anchor_table[f'neighbor_{rank+1}_train_row_id'] = rows[graph[:, rank]]
        anchor_table[f'weight_rank_{rank+1}_mean'] = metrics['weight_mean_by_rank'][:, rank]
        anchor_table[f'weight_rank_{rank+1}_noise_std'] = metrics['weight_std_by_rank'][:, rank]
    class_row = {**common, 'status': 'ANALYZED'}
    for key, value in scalar_metrics.items():
        for stat, number in [('mean', np.mean(value)), ('std_across_anchors', np.std(value)),
                             ('q05', np.quantile(value,.05)), ('median', np.median(value)),
                             ('q95', np.quantile(value,.95))]:
            class_row[f'{key}__{stat}'] = float(number)
    ranks, hist, usage = [], [], []
    bins = np.linspace(0, 1, 41)
    for rank in range(3):
        values = weights[:, :, rank].astype(np.float64)
        ranks.append({**common, 'neighbor_distance_rank': rank+1,
                      'uniform_reference': 1/3, 'weight_mean': float(values.mean()),
                      'mean_anchor_noise_std': float(metrics['weight_std_by_rank'][:, rank].mean()),
                      'weight_p05': float(np.quantile(values,.05)), 'weight_median': float(np.median(values)),
                      'weight_p95': float(np.quantile(values,.95)),
                      'mean_fractional_dominance': float(metrics['dominance_fraction_by_rank'][:,rank].mean())})
        counts, _ = np.histogram(values, bins=bins)
        for i, count in enumerate(counts):
            hist.append({**common, 'neighbor_distance_rank': rank+1, 'bin_left': bins[i],
                         'bin_right': bins[i+1], 'count': int(count),
                         'relative_frequency': float(count/values.size)})
    observed = np.bincount(graph.ravel(), weights=metrics['weight_mean_by_rank'].ravel(), minlength=len(graph))/len(graph)
    uniform = np.bincount(graph.ravel(), minlength=len(graph))/(3*len(graph))
    for source in range(len(graph)):
        usage.append({**common, 'support_local_id': source, 'train_row_id': int(rows[source]),
                      'observed_neighbor_share': observed[source], 'uniform_on_same_graph_share': uniform[source],
                      'excess_share_vs_graph_uniform': observed[source]-uniform[source]})
    # Preserve actual float32 probabilities, full trained Dense logits, fixed
    # source IDs and feature hashes. These are analysis outputs, not checkpoints.
    return anchor_table, class_row, ranks, hist, usage


def plot_tables(folder, class_table, rank_table, hist_table, anchor_table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=True)
    active = class_table[class_table.status == 'ANALYZED']
    fields = [
        ('entropy_mean_nats__mean', 'Mean conditional entropy (nats)', 0, np.log(3)),
        ('effective_neighbors_mean__mean', 'Mean exp(entropy)', 1, 3),
        ('maximum_weight_mean__mean', 'Mean maximum neighbor weight', 1/3, 1),
        ('noise_jsd_nats__mean', 'Noise JSD (nats)', 0, None),
        ('weight_noise_rms_l2__mean', 'Weight noise RMS L2', 0, None),
        ('patch_noise_rms_per_coordinate__mean', 'Patch noise RMS / coordinate', 0, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), constrained_layout=True)
    for ax, (field, title, low, high) in zip(axes.flat, fields):
        table = active.pivot(index='class_id', columns='model_seed', values=field).sort_index()
        im = ax.imshow(table.values, aspect='auto', vmin=low, vmax=high, cmap='viridis')
        ax.set_xticks(range(3), ['42','43','44'])
        ax.set_yticks(range(len(table)), [f'{i}: {CLASS_NAMES[i-1]}' for i in table.index], fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Trained model seed')
        fig.colorbar(im, ax=ax, fraction=.04)
    fig.suptitle('Learned-local V2: final checkpoint, train-only endpoint probe')
    fig.savefig(folder/'class_seed_diagnostics.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, constrained_layout=True)
    for ax, seed in zip(axes, (42,43,44)):
        seed_hist = hist_table[hist_table.model_seed == seed]
        for rank in (1,2,3):
            data = seed_hist[seed_hist.neighbor_distance_rank == rank]
            # Average per-class distributions equally; no majority dominance.
            data = data.groupby(['bin_left','bin_right']).relative_frequency.mean().reset_index()
            ax.step((data.bin_left+data.bin_right)/2, data.relative_frequency, where='mid', label=f'Neighbor rank {rank}')
        ax.axvline(1/3, color='black', linestyle='--', linewidth=1, label='Uniform 1/3')
        ax.set_title(f'Seed {seed}'); ax.set_xlabel('Conditional neighbor weight')
        ax.legend(fontsize=7)
    axes[0].set_ylabel('Equal-class mean relative frequency')
    fig.savefig(folder/'neighbor_weight_distributions.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, seed in zip(axes, (42,43,44)):
        data = anchor_table[anchor_table.model_seed == seed]
        p = ax.scatter(data.weight_noise_rms_l2, data.patch_noise_rms_per_coordinate,
                       c=data.class_id, cmap='tab20', s=3, alpha=.5, vmin=1, vmax=16)
        ax.set_title(f'Seed {seed}'); ax.set_xlabel('Weight noise RMS L2')
        ax.set_ylabel('Patch noise RMS / coordinate')
    fig.colorbar(p, ax=axes, label='Original class ID', fraction=.02)
    fig.savefig(folder/'weight_vs_patch_noise.png', dpi=160); plt.close(fig)


def final_tables_and_report(out, manifest):
    import pandas as pd
    class_frames, anchor_frames, rank_frames, hist_frames, usage_frames = [], [], [], [], []
    for seed in (42,43,44):
        sf = out/f'seed_{seed}'
        require((sf/'seed_complete.json').is_file(), f'Seed {seed} has no completion record')
        class_frames.append(pd.read_csv(sf/'per_class_metrics.csv'))
        anchor_frames.append(pd.read_csv(sf/'per_anchor_metrics.csv'))
        rank_frames.append(pd.read_csv(sf/'neighbor_rank_summary.csv'))
        hist_frames.append(pd.read_csv(sf/'neighbor_weight_histograms.csv'))
        usage_frames.append(pd.read_csv(sf/'global_support_usage.csv'))
    class_df, anchor_df, rank_df, hist_df, usage_df = [pd.concat(v, ignore_index=True)
        for v in (class_frames,anchor_frames,rank_frames,hist_frames,usage_frames)]
    require(len(class_df)==48 and (class_df.status=='ANALYZED').sum()==45,
            'Incomplete class registry; expected 45 learned heads + 3 majority N/A rows')
    require(len(anchor_df)==3*4138, 'Incomplete matched anchor registry')
    for name, table in [('per_class_metrics',class_df),('per_anchor_metrics',anchor_df),
                        ('neighbor_rank_summary',rank_df),('neighbor_weight_histograms',hist_df),
                        ('global_support_usage',usage_df)]:
        table.to_csv(out/f'{name}.csv', index=False)
    metrics = ['entropy_mean_nats','effective_neighbors_mean','maximum_weight_mean',
               'noise_jsd_nats','weight_noise_rms_l2','patch_noise_rms_per_coordinate',
               'uniform_displacement_rms_per_coordinate']
    active = class_df[class_df.status=='ANALYZED']
    macro = active.groupby('model_seed')[[m+'__mean' for m in metrics]].mean().reset_index()
    macro.to_csv(out/'per_seed_equal_class_summary.csv',index=False)
    deltas = []
    for left,right in ((42,43),(42,44),(43,44)):
        a=anchor_df[anchor_df.model_seed==left].set_index(['class_id','anchor_local_id'])
        b=anchor_df[anchor_df.model_seed==right].set_index(['class_id','anchor_local_id'])
        require(a.index.equals(b.index), 'Anchor IDs differ between seeds')
        for c in sorted(a.index.get_level_values('class_id').unique()):
            for metric in metrics:
                values=b.loc[c,metric].values-a.loc[c,metric].values
                deltas.append({'left_seed':left,'right_seed':right,'delta_definition':'right minus left',
                               'class_id':c,'metric':metric,'anchors':len(values),
                               'mean_paired_delta':values.mean(),'median_paired_delta':np.median(values),
                               'q05_paired_delta':np.quantile(values,.05),'q95_paired_delta':np.quantile(values,.95)})
    pd.DataFrame(deltas).to_csv(out/'matched_seed_pair_deltas.csv',index=False)
    plot_tables(out/'figures',class_df,rank_df,hist_df,anchor_df)
    lines = [
        '# Learned-local V2 — 3 seeds: hậu nghiệm tại checkpoint cuối', '',
        f"Trạng thái: **{manifest['status']}**. Protocol B `50e988fae260`, 8.000 C updates.", '',
        '## Số liệu mô tả', '',
        'Mỗi số dưới đây là trung bình đều qua 15 lớp có learned head; trong mỗi lớp lấy đều các anchor và cùng ngân sách noise.', '',
        '| Seed | H (nats) | mean exp(H) | max weight | noise JSD | weight noise RMS | patch noise RMS/coordinate |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for _,row in macro.iterrows():
        values=[row[m+'__mean'] for m in metrics[:6]]
        lines.append('| '+str(int(row.model_seed))+' | '+' | '.join(f'{v:.7g}' for v in values)+' |')
    lines += ['', '## Tái hiện implementation M4', '',
        '- Dựng G và head từ nguyên văn các hàm source M4 đã kiểm SHA; đúng 50 tensor G/head gồm 8 buffer BatchNorm.',
        '- Đúng train final 5.528 mẫu và PCA cache của B; đúng class rank, support row order, kNN Euclidean float64, k=3 và rho=0.2.',
        '- Dùng logits của Dense được khôi phục; gather ba neighbor rồi tf.nn.softmax đúng local_mix. Forward training=False tương ứng tuyến sinh fake cho C.',
        '- Không dựng C/D hoặc Adam. Chỉ khôi phục phần G/head; không tuyên bố restore toàn bộ training state.', '',
        '## Lựa chọn phân tích hậu nghiệm', '',
        f"- Toàn bộ 4.138 anchor thiểu số; {manifest['config']['noises_per_class']} noise N(0,1) mỗi lớp, analysis seed {manifest['config']['analysis_seed']}. Cùng anchor và noise giữa 42/43/44; không phải replay noise đã dùng trong training.",
        '- G không phụ thuộc anchor: tính feature và Dense logits một lần cho mỗi noise/lớp rồi dùng đúng mask của từng anchor. Kiểm đầu ra head gốc trên anchor đầu/giữa/cuối mỗi lớp với toàn bộ noise.',
        '- Entropy và exp(H) xét riêng ba neighbor. Mixture entropy được ghi riêng cho [0.8,0.2*w1,0.2*w2,0.2*w3].',
        '- Patch noise RMS dùng Gram matrix của ba neighbor trong PCA space; đối chiếu với output head float32 ở tập anchor cố định. Chỉ bỏ sai số tổng xác suất float32 khi làm phép tính hình học, không sửa trọng số được lưu.',
        '- So sánh usage của từng support với uniform trên cùng graph; không dùng 1/Nc vì indegree khác nhau.', '',
        '## Chưa thể suy luận', '',
        '- Trọng số khác uniform hoặc phụ thuộc noise chưa chứng minh chúng có ích cho classification. Trọng số gần cố định cũng chưa chứng minh mô hình hỏng.',
        '- Không có trajectory trọng số trong training: không suy ngược thời điểm học, collapse hay hội tụ từ endpoint này.',
        '- Khác biệt của seed 43 trong báo cáo này chỉ là mô tả; không xác định nguyên nhân Macro-F1 kém Random-local. Không chạy classifier hoặc dùng test/validation để chọn tham số.',
        '- Noise được chia sẻ giữa anchor tạo phụ thuộc; không coi từng anchor/noise là một lần huấn luyện độc lập và không tính p-value kiểu đó.',
        '- Lớp 11 Soybean-mintill không có learned head: N/A. Kết quả chỉ áp dụng cho inference-mode BN và tập noise hữu hạn này.', '',
        '## File chính', '',
        '`per_class_metrics.csv`, `per_anchor_metrics.csv`, `neighbor_rank_summary.csv`, `global_support_usage.csv`, `matched_seed_pair_deltas.csv`, `figures/`, `seed_*/weights/class_*.npz` và các JSON xác minh.',
    ]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return macro


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


def run_three_seed_analysis(config):
    import pandas as pd
    started=time.monotonic()
    tag=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=Path('/kaggle/working')/('V2_LEARNED_WEIGHTING_3SEEDS_'+tag)
    scratch=Path('/kaggle/working')/('_v2_weighting_inputs_'+tag)
    out.mkdir(parents=True,exist_ok=False)
    manifest={'status':'RUNNING','complete':False,'config':config,
              'protocol_B_id':PROTOCOL_ID,'arm':'learned_local','c_updates':8000,
              'required_seeds':[42,43,44],'completed_seeds':[],
              'training_performed':False,'validation_arrays_loaded':False,
              'test_arrays_loaded':False,'architecture_fallback_used':False,
              'source_worker_sha256':M4_WORKER_SHA,'exact_function_sha256':EXACT_FUNCTION_SHA,
              'sampling_plan':{'minority_anchors_per_seed':4138,
                  'noise_vectors_per_class':config['noises_per_class'],
                  'conditional_weight_triplets_per_seed':4138*config['noises_per_class'],
                  'generator_vectors_per_seed':15*config['noises_per_class'],
                  'head_route_check_anchor_rule':'sorted unique [0, N//2, N-1] in each class; all common noises',
                  'noise_shared_across_trained_seeds':True,
                  'noise_shared_across_anchors_within_class':True},
              'seed_registry':{str(s):{'status':'NOT_STARTED'} for s in (42,43,44)}}
    error=None; macro=None
    try:
        require(config['noises_per_class']>=2 and config['batch_size']>=1,'Invalid fixed sampling budget')
        require(config['model_seeds']==[42,43,44], 'All three requested model seeds are mandatory')
        write_json(out/'metric_unit_checks.json',run_synthetic_checks())
        refs,trains,issues=inventory_input_refs(config['input_roots'])
        write_json(out/'input_inventory.json',{'metadata_files':[r.label() for r in refs],
                   'final_train_files':[r.label() for r in trains],'inventory_issues':issues})
        selected,rejected=discover_checkpoint_inputs(refs,scratch/'checkpoints')
        for seed in (42,43,44):
            manifest['seed_registry'][str(seed)]={'status':'IDENTITY_VERIFIED' if seed in selected else 'MISSING_OR_REJECTED',
                'metadata_source':selected[seed]['metadata_source'] if seed in selected else None}
        write_json(out/'checkpoint_discovery.json',{'selected':selected,'rejected':rejected})
        write_json(out/'analysis_manifest.json',manifest)
        require(set(selected)=={42,43,44}, 'Missing verified Learned-local final checkpoint(s): '+
                str(sorted({42,43,44}-set(selected)))+'. Inspect checkpoint_discovery.json; flat seed42 root is supported. No two-seed fallback.')
        print('Đủ checkpoint đúng metadata/hash: seeds 42, 43, 44. Chưa chạy forward.',flush=True)
        X,y,coords,train_audit=load_final_train_verified(trains,scratch/'train')
        write_json(out/'train_provenance.json',train_audit)
        prepared=build_graph_per_class(X,y)
        graph_hashes=[sha_array(g) for g in prepared['neighbor_graphs']]
        require(all(graph_hashes==s['meta']['contract']['graph_sha256'] for s in selected.values()),
                'Train-only neighbor graphs differ from checkpoint contracts')
        require(int(sum(len(s) for s in prepared['supports']))==4138,'Wrong minority anchor count')
        write_json(out/'support_graph_verification.json',{'status':'PASS','graph_sha256':graph_hashes,
                   'class_order_original_ids':(prepared['order']+1).tolist(),'anchors':4138})
        mapping=[]
        for h,c in enumerate(prepared['order']+1):
            mapping.append({'class_id':int(c),'class_name':CLASS_NAMES[c-1],
                            'train_supports':int(prepared['counts'][c-1]),'sorted_class_rank':h,
                            'status':'LEARNED_HEAD' if h<15 else 'NOT_APPLICABLE_NO_HEAD'})
        pd.DataFrame(mapping).to_csv(out/'class_mapping.csv',index=False)
        noises=np.stack([np.random.default_rng(np.random.SeedSequence([config['analysis_seed'],int(c)]))
                         .normal(size=(config['noises_per_class'],100)).astype(np.float32)
                         for c in prepared['order'][:15]+1])
        np.savez_compressed(out/'common_probe_inputs.npz',noise_by_head=noises,
                            class_order=prepared['order']+1,train_row_ids=np.arange(len(y)),
                            train_labels=y,train_coords=coords)
        manifest['common_noise_sha256']=sha_array(noises)
        manifest['runtime']=load_probe_runtime()
        print('Runtime:',manifest['runtime'],flush=True)
        tensor_sets={}
        # Native tensor audit for ALL seeds before any model forward.
        for seed in (42,43,44):
            item=selected[seed]
            try:
                tensors,audit=preflight_checkpoint_tensors(Path(item['prefix']).parent,item['meta'])
            except Exception as exc:
                manifest['seed_registry'][str(seed)]['status']='FAILED_NATIVE_TENSOR_GATE'
                manifest['seed_registry'][str(seed)]['error']=str(exc)
                raise
            tensor_sets[seed]=tensors
            write_json(out/f'seed_{seed}'/'native_tensor_verification.json',audit)
            manifest['seed_registry'][str(seed)]['status']='NATIVE_TENSORS_VERIFIED'
        for seed in (42,43,44):
            print(f'Seed {seed}: restore chính xác G + 15 Dense heads; chỉ forward train.',flush=True)
            sf=out/f'seed_{seed}'; item=selected[seed]
            try:
                keras.backend.clear_session(); keras.utils.set_random_seed(20261016)
                G,heads,restore_audit=restore_generator_and_heads(prepared,item['meta'],tensor_sets.pop(seed))
                write_json(sf/'restore_verification.json',restore_audit)
                before=model_state_hashes(G,heads)
                require(all(before[k]==item['meta']['state_hashes'][k] for k in before),'Pre-forward hash mismatch')
                all_anchors=[]; class_rows=[]; rank_rows=[]; hist_rows=[]; usage_rows=[]; head_checks=[]
                try:
                    for h,(support,graph) in enumerate(zip(prepared['supports'],prepared['neighbor_graphs'])):
                        c=int(prepared['order'][h]+1)
                        weights,features,logits=actual_neighbor_weights(G,heads[h],graph,noises[h],h,config['batch_size'])
                        metrics=analyze_weight_tensor(weights,support.reshape(len(support),-1),graph,rho=.2)
                        checks=verify_head_route(heads[h],features,weights,support,graph,metrics,config['batch_size'])
                        head_checks.append({'class_id':c,'head_index':h,'checks':checks})
                        rows=np.flatnonzero(y==c-1)
                        a,cl,rr,hh,uu=save_class_tables(sf,seed,c,h,weights,metrics,graph,rows,coords)
                        all_anchors.append(a); class_rows.append(cl); rank_rows.extend(rr); hist_rows.extend(hh); usage_rows.extend(uu)
                        wf=sf/'weights'; wf.mkdir(exist_ok=True)
                        np.savez_compressed(wf/f'class_{c:02d}.npz',weights=weights,actual_dense_logits=logits,
                            anchor_local_ids=np.arange(len(support),dtype=np.int32),anchor_train_row_ids=rows,
                            neighbor_local_ids=graph,neighbor_train_row_ids=rows[graph],noise_indices=np.arange(len(noises[h])),
                            model_seed=np.int64(seed),class_id=np.int64(c),head_index=np.int64(h),rho=np.float64(.2))
                        print(f'  lớp {c:2d}: {len(support)} anchors × {len(noises[h])} noise; head route PASS',flush=True)
                    class_rows.append({'model_seed':seed,'class_id':11,'class_name':CLASS_NAMES[10],
                        'head_index_sorted':15,'anchors':1390,'noises_per_anchor':0,'status':'NOT_APPLICABLE_NO_HEAD'})
                    pd.concat(all_anchors,ignore_index=True).to_csv(sf/'per_anchor_metrics.csv',index=False)
                    pd.DataFrame(class_rows).to_csv(sf/'per_class_metrics.csv',index=False)
                    pd.DataFrame(rank_rows).to_csv(sf/'neighbor_rank_summary.csv',index=False)
                    pd.DataFrame(hist_rows).to_csv(sf/'neighbor_weight_histograms.csv',index=False)
                    pd.DataFrame(usage_rows).to_csv(sf/'global_support_usage.csv',index=False)
                    write_json(sf/'actual_head_forward_verification.json',head_checks)
                finally:
                    after=model_state_hashes(G,heads)
                    unchanged=before==after
                    write_json(sf/'nonmutation_verification.json',{'status':'PASS' if unchanged else 'FAIL',
                        'array_count':len(before),'before':before,'after':after,
                        'all_trainable_and_BN_buffers_unchanged':unchanged})
                    require(unchanged,'Forward changed Generator/head weights or BN buffers')
                # Recheck original read-only checkpoint payload after forward.
                payload_after={name:sha_file(Path(item['prefix']).parent/name) for name in item['payload_sha256']}
                require(payload_after==item['payload_sha256'],'Input checkpoint bytes changed')
                write_json(sf/'checkpoint_bytes_after.json',payload_after)
                write_json(sf/'seed_complete.json',{'seed':seed,'status':'PASS','head_count':15,'anchors':4138,
                    'common_noise_sha256':manifest['common_noise_sha256'],'all_50_arrays_unchanged':True})
                manifest['completed_seeds'].append(seed)
                manifest['seed_registry'][str(seed)]['status']='COMPLETE'
                del G,heads; keras.backend.clear_session(); gc.collect()
            except Exception as exc:
                manifest['seed_registry'][str(seed)]['status']='FAILED'
                manifest['seed_registry'][str(seed)]['error']=str(exc)
                raise
            finally:
                write_json(out/'analysis_manifest.json',manifest)
        require(manifest['completed_seeds']==[42,43,44],'Three-seed registry incomplete')
        manifest['status']='COMPLETE_3_SEEDS'
        macro=final_tables_and_report(out,manifest)
        manifest['complete']=True
    except Exception as exc:
        error=exc
        manifest['complete']=False; manifest['status']='INCOMPLETE_FAILED_CHECK'
        manifest['error_type']=type(exc).__name__; manifest['error']=str(exc)
        (out/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        (out/'REPORT.md').write_text('# Phân tích CHƯA HOÀN TẤT\n\n'+str(exc)+
            '\n\nKhông diễn giải phần còn lại như so sánh đủ ba seed. Không có fallback, train hoặc test. Xem analysis_manifest.json và ERROR.txt.\n',encoding='utf-8')
    finally:
        manifest['elapsed_seconds']=round(time.monotonic()-started,3)
        write_json(out/'analysis_manifest.json',manifest)
        write_json(out/'metric_definitions.json',METRIC_DESCRIPTIONS)
        target=package_report(out)
        print('\nTrạng thái:',manifest['status'],'| seeds hoàn tất:',manifest['completed_seeds'],flush=True)
        print('Báo cáo:',out,'\nZIP DUY NHẤT:',target,flush=True)
        try:
            from IPython.display import FileLink,display
            display(FileLink(str(target)))
            if macro is not None: display(macro)
        except ImportError:
            pass
    if error is not None:
        raise RuntimeError('Analysis incomplete; report ZIP saved. '+str(error)) from error
    return {'report_directory':str(out),'report_zip':str(target),'manifest':manifest}

