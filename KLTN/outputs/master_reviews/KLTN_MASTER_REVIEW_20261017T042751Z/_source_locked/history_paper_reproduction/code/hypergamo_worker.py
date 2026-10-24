"""3D-HyperGAMO training adapter: original networks, MSE and adversarial update sequence.

Only data protocol, compatibility, logging, validation selection and persistence differ.
Never import the test cache in the training stage.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['smoke', 'train', 'samples', 'evaluate', 'infer'])
parser.add_argument('--run', required=True)
parser.add_argument('--input', help='Prepared inference patches; no fitting or labels.')
parser.add_argument('--output', help='Destination for inference probabilities.')
parser.add_argument('--cpu', action='store_true', help='CPU only for engineering smoke tests.')
args = parser.parse_args()
RUN = Path(args.run).resolve()
CFG = json.loads((RUN / 'config.json').read_text(encoding='utf-8'))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if args.cpu else str(CFG['gpu_index'])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, precision_recall_fscore_support

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
devices = tf.config.list_physical_devices('GPU')
for dev in devices:
    tf.config.experimental.set_memory_growth(dev, True)
if args.stage != 'smoke' and (not devices or args.cpu):
    raise RuntimeError('Huấn luyện/đánh giá cần GPU. Chưa khởi chạy CPU thay thế.')
print('TensorFlow:', tf.__version__, '| tf-keras:', keras.__version__, '| GPU:', devices, flush=True)
keras.utils.set_random_seed(int(CFG['train_seed']))
try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass

# The pinned author source is embedded next to this adapter. Import-only compatibility changes.
author_source = (RUN / 'code' / 'dense_net3D_SA_original.py').read_text(encoding='utf-8')
source = author_source.replace('from keras.layers.advanced_activations import', 'from tf_keras.layers import')
source = source.replace('from keras', 'from tf_keras')
NT = {}
exec(compile(source, 'dense_net3D_SA_compat.py', 'exec'), NT)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def bundle():
    target = RUN.parent / (RUN.name + '.zip')
    tmp = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(RUN.rglob('*')):
            if path.is_file() and not path.name.endswith('.tmp'):
                archive.write(path, str(Path(RUN.name) / path.relative_to(RUN)))
    tmp.replace(target)
    print('ZIP:', target, flush=True)


def make_models(x, y, seed):
    keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    ids, counts = np.unique(y, return_counts=True)
    assert np.array_equal(ids, np.arange(len(ids)))
    order = np.argsort(counts, kind='mergesort')
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    y = inverse[y].astype(np.int64)
    counts = counts[order]
    permutation = np.random.choice(np.arange(len(x)), size=len(x), replace=False)
    x, y = x[permutation], y[permutation]
    c = len(order)
    minority = np.flatnonzero(counts < counts[-1])
    if len(minority) == 0:
        raise RuntimeError('Train đã cân bằng; không có lớp thiểu số cho HyperGAMO.')
    assert np.array_equal(minority, np.arange(len(minority)))
    supports = [x[y == i] for i in minority]
    shape = tuple(x.shape[1:])
    # Original code uses ONE shared Adam across C, D and composite G models.
    opt = keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)
    C = NT['denseMlpCreate'](sh=shape, num_class=c)
    C.compile(loss='mean_squared_error', optimizer=opt)
    C.trainable = False
    D = NT['denseDisCreate'](shape, num_class=c)
    D.compile(loss='mean_squared_error', optimizer=opt)
    D.trainable = False
    G = NT['denseGamoGenCreate'](100, c)
    heads, vs_c, vs_d = [], [], []
    for support in supports:
        head = NT['denseGenProcessCreate'](len(support), support, sh=shape, mul=int(np.prod(shape)))
        heads.append(head)
        z = keras.layers.Input(shape=(100,))
        label = keras.layers.Input(shape=(c,))
        mixed = head(G([z, label]))
        g_c = keras.Model([z, label], C(mixed))
        g_c.compile(loss='mean_squared_error', optimizer=opt)
        vs_c.append(g_c)
        z2 = keras.layers.Input(shape=(100,))
        label2 = keras.layers.Input(shape=(c,))
        label3 = keras.layers.Input(shape=(c,))
        mixed2 = head(G([z2, label2]))
        g_d = keras.Model([z2, label2, label3], D([mixed2, label3]))
        g_d.compile(loss='mean_squared_error', optimizer=opt)
        vs_d.append(g_d)
    tracked = tf.Module()
    tracked.classifier, tracked.discriminator, tracked.generator = C, D, G
    tracked.optimizer = opt
    for i, head in enumerate(heads):
        setattr(tracked, f'head_{i:03d}', head)
    checkpoint = tf.train.Checkpoint(models=tracked)
    return dict(C=C, D=D, G=G, heads=heads, vs_c=vs_c, vs_d=vs_d, checkpoint=checkpoint,
                optimizer=opt, x=x, y=y, order=order, inverse=inverse, counts=counts,
                supports=supports, shape=shape, c=c)


def scalar(value):
    value = float(np.asarray(value).reshape(-1)[0])
    if not np.isfinite(value):
        raise FloatingPointError('Loss NaN/Inf. Dừng, không ghi kết quả thành công.')
    return value


def predict(C, x, batch=32):
    return np.concatenate([C.predict_on_batch(x[i:i+batch]) for i in range(0, len(x), batch)])


def metrics(y, pred, c):
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(c), zero_division=0)
    result = dict(OA=float(accuracy_score(y, pred)), AA=float(recall.mean()),
                  Macro_F1=float(f1.mean()), GM=float(np.exp(np.log(recall).mean())) if (recall > 0).all() else 0.,
                  Kappa=float(cohen_kappa_score(y, pred)))
    return result, (precision, recall, f1, support)


def snap(model):
    return [w.numpy().copy() for w in model.weights]


def changed(model, before):
    return any(not np.array_equal(w.numpy(), old) for w, old in zip(model.weights, before))


def smoke():
    # Same spatial/spectral dimensions as the intended run; three artificial classes only.
    shape = (CFG['patch_size'], CFG['patch_size'], CFG['pca_components'], 1)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(9, *shape)).astype('float32')
    y = np.repeat(np.arange(3), [2, 3, 4])
    m = make_models(x, y, 7)
    real = m['supports'][0][:2]
    labels = keras.utils.to_categorical([0, 0], 3).astype('float32')
    noise = rng.normal(size=(2, 100)).astype('float32')
    for name, model, inputs, targets in [
        ('C real', m['C'], real, labels),
        ('D real', m['D'], [real, labels], np.ones((2, 1), 'float32')),
    ]:
        before = snap(model)
        loss = scalar(model.train_on_batch(inputs, targets))
        assert changed(model, before), name + ': trọng số không cập nhật.'
        print('PASS', name, 'loss', loss, flush=True)
    fake = m['heads'][0].predict_on_batch(m['G'].predict_on_batch([noise, labels]))
    assert fake.shape == real.shape and np.isfinite(fake).all()
    assert (fake >= real.min(0) - 1e-5).all() and (fake <= real.max(0) + 1e-5).all()
    for name, model, inputs, targets in [
        ('C fake', m['C'], fake, labels),
        ('D fake', m['D'], [fake, labels], np.zeros((2, 1), 'float32')),
    ]:
        before = snap(model)
        loss = scalar(model.train_on_batch(inputs, targets))
        assert changed(model, before), name + ': trọng số không cập nhật.'
        print('PASS', name, 'loss', loss, flush=True)
    for name, model, inputs, targets in [
        ('G against C', m['vs_c'][0], [noise, labels], 1-labels),
        ('G against D', m['vs_d'][0], [noise, labels, labels], np.ones((2, 1), 'float32')),
    ]:
        cb, db, gb, hb = map(snap, [m['C'], m['D'], m['G'], m['heads'][0]])
        loss = scalar(model.train_on_batch(inputs, targets))
        assert not changed(m['C'], cb) and not changed(m['D'], db), name + ': C/D không đóng băng.'
        assert changed(m['G'], gb) and changed(m['heads'][0], hb), name + ': G/head không cập nhật.'
        print('PASS', name, 'loss', loss, flush=True)
    # Checkpoint reload must preserve model output before any long run.
    smoke_dir = RUN / 'smoke_checkpoint'
    smoke_dir.mkdir(exist_ok=True)
    path = m['checkpoint'].write(str(smoke_dir / 'check'))
    expected = m['C'].predict_on_batch(real)
    first = m['C'].weights[0]
    first.assign(tf.zeros_like(first))
    m['checkpoint'].restore(path).expect_partial()
    np.testing.assert_allclose(m['C'].predict_on_batch(real), expected, rtol=1e-5, atol=1e-6)
    atomic_json(RUN / 'smoke_pass.json', dict(passed=True, tensorflow=tf.__version__, tf_keras=keras.__version__,
                shape=list(shape), gpu=bool(devices), data_is_artificial=True))
    print('SMOKE PASSED — đây không phải kết quả nghiên cứu.', flush=True)


def write_history(history):
    path = RUN / 'training_history.csv'
    tmp = path.with_suffix('.csv.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    tmp.replace(path)


def train():
    started = time.monotonic()
    if not (RUN / 'smoke_pass.json').exists():
        raise RuntimeError('Chưa có smoke test thành công.')
    state_path = RUN / 'training_state.json'
    old_state = json.loads(state_path.read_text()) if state_path.exists() else None
    if old_state and old_state['complete']:
        print('Lần chạy này đã hoàn thành. Không train lại.', flush=True)
        return
    data = np.load(RUN / 'train_val.npz', allow_pickle=False)
    x, y = data['X_train'], data['y_train']
    val_x, val_y_orig = data['X_val'], data['y_val']
    m = make_models(x, y, int(CFG['train_seed']))
    val_y = m['inverse'][val_y_orig]
    class_ids = np.asarray(CFG['accepted_class_ids'], dtype=int)[m['order']]
    atomic_json(RUN / 'model_class_order.json', dict(original_class_ids=class_ids.tolist()))
    latest = tf.train.CheckpointManager(m['checkpoint'], str(RUN / 'checkpoints' / 'latest'), max_to_keep=2)
    best = tf.train.CheckpointManager(m['checkpoint'], str(RUN / 'checkpoints' / 'best'), max_to_keep=2)
    use_val = CFG['protocol'] == 'dip_train_only_validation'
    if use_val != bool(len(val_x)):
        raise RuntimeError('Cache validation không khớp protocol.')
    step, history, best_f1 = 0, [], -1.
    generated_counts = np.zeros(m['c'], dtype=np.int64)
    # Roll back metadata too if a prior process died between saving best and latest.
    if old_state and old_state.get('selection'):
        atomic_json(RUN / 'selection.json', old_state['selection'])
    if (RUN / 'selection.json').exists():
        best_f1 = json.loads((RUN / 'selection.json').read_text()).get('Macro_F1', -1.)
    if old_state:
        if not CFG['resume_training']:
            raise RuntimeError('Có checkpoint dở dang. Bật resume_training để tiếp tục, không ghi đè.')
        path = RUN / old_state['checkpoint']
        m['checkpoint'].restore(str(path)).expect_partial()
        step = int(old_state['step'])
        generated_counts = np.array(old_state['generated_counts'], dtype=np.int64)
        state = old_state['numpy_rng']
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        if (RUN / 'training_history.csv').exists():
            with open(RUN / 'training_history.csv', newline='') as f:
                history = [dict(row) for row in csv.DictReader(f) if int(row['step']) <= step]
        print('Tiếp tục từ step', step, '(không cam kết trùng bit với chạy liên tục: RNG Dropout).', flush=True)
    labels = keras.utils.to_categorical(m['y'], m['c']).astype('float32')
    n, bs = len(m['x']), int(CFG['batch_size'])
    n_batches = int(math.ceil(n / bs))
    gen_points = int(math.ceil(bs / m['c']))
    deficits = m['counts'][-1] - m['counts'][:len(m['heads'])]
    cumulative = np.cumsum(deficits / deficits.sum())
    max_step = int(CFG['max_step'])
    if max_step <= 0 or max_step % 2:
        raise ValueError('max_step phải là số chẵn dương (mỗi vòng có hai cập nhật C).')
    recent = np.zeros(4, dtype=float)
    recent_n = 0
    while step < max_step:
        j = (step // 2) % n_batches
        a, b = j * bs, min((j + 1) * bs, n)
        size = b - a
        valid_r = 1. - np.random.uniform(0., .1, (size, 1))
        loss_c_real = scalar(m['C'].train_on_batch(m['x'][a:b], labels[a:b]))
        loss_d_real = scalar(m['D'].train_on_batch([m['x'][a:b], labels[a:b]], valid_r))
        invalid = np.random.uniform(0., .1, (size, 1))
        z = np.random.normal(size=(size, 100))
        fake_ids = np.searchsorted(cumulative, np.random.rand(size), side='right')
        generated_counts += np.bincount(fake_ids, minlength=m['c'])
        fake_labels = keras.utils.to_categorical(fake_ids, m['c']).astype('float32')
        features = m['G'].predict_on_batch([z, fake_labels])
        fake = np.zeros((size, *m['shape']), dtype='float32')
        for i, head in enumerate(m['heads']):
            idx = np.flatnonzero(fake_ids == i)
            if len(idx):
                fake[idx] = head.predict_on_batch(features[idx])
        loss_c_fake = scalar(m['C'].train_on_batch(fake, fake_labels))
        loss_d_fake = scalar(m['D'].train_on_batch([fake, fake_labels], invalid))
        g_c, g_d = [], []
        for i in range(len(m['heads'])):
            condition = np.zeros((gen_points, m['c']), dtype='float32')
            condition[:, i] = 1
            noise = np.random.normal(size=(gen_points, 100))
            # Important: retain the original adversarial target, NOT normal class CE.
            g_c.append(scalar(m['vs_c'][i].train_on_batch([noise, condition], 1-condition)))
            g_d.append(scalar(m['vs_d'][i].train_on_batch(
                [noise, condition, condition], np.ones((gen_points, 1)))))
        recent += [(loss_c_real + loss_c_fake) / 2, (loss_d_real + loss_d_fake) / 2,
                   np.mean(g_c), np.mean(g_d)]
        recent_n += 1
        step += 2
        timed_out = (time.monotonic() - started) / 60 >= CFG['max_minutes']
        if step % int(CFG['eval_every']) == 0 or step == max_step or timed_out:
            values = {}
            if use_val:
                val_pred = predict(m['C'], val_x, CFG['eval_batch_size']).argmax(1)
                values, _ = metrics(val_y, val_pred, m['c'])
            row = dict(step=step, **values, C_loss=recent[0]/recent_n, D_loss=recent[1]/recent_n,
                       G_vs_C_loss=recent[2]/recent_n, G_vs_D_loss=recent[3]/recent_n,
                       elapsed_minutes=(time.monotonic()-started)/60)
            history.append(row)
            write_history(history)
            if (use_val and values['Macro_F1'] > best_f1) or (not use_val and step == max_step):
                best_f1 = values.get('Macro_F1', -1.)
                best_path = best.save(checkpoint_number=step)
                # The metadata must not point to an unfinished/mismatched classifier file.
                classifier_path = RUN / 'checkpoints' / 'best' / f'classifier_{step}.weights.h5'
                temporary = classifier_path.with_name(f'classifier_{step}.tmp.weights.h5')
                m['C'].save_weights(str(temporary))
                temporary.replace(classifier_path)
                atomic_json(RUN / 'selection.json', dict(
                    step=step, rule='validation_macro_f1' if use_val else 'fixed_final_budget',
                    **values, checkpoint=str(Path(best_path).relative_to(RUN)),
                    classifier_weights=str(classifier_path.relative_to(RUN))))
            last_path = latest.save(checkpoint_number=step)
            rng = np.random.get_state()
            state = dict(step=step, complete=step >= max_step, checkpoint=str(Path(last_path).relative_to(RUN)),
                         numpy_rng=[rng[0], rng[1].tolist(), int(rng[2]), int(rng[3]), float(rng[4])],
                         model_class_ids=class_ids.tolist(), generated_counts=generated_counts.tolist(),
                         selection=json.loads((RUN / 'selection.json').read_text()) if (RUN / 'selection.json').exists() else None,
                         reason='time_limit' if timed_out and step < max_step else 'checkpoint')
            atomic_json(state_path, state)
            atomic_json(RUN / 'online_generated_counts.json', dict(class_ids=class_ids.tolist(),
                real_train_counts=m['counts'].tolist(), generated_presentations=generated_counts.tolist(),
                step=step, note='Cumulative fake examples presented to C, not unique saved samples.'))
            print('VALIDATION' if use_val else 'TRAIN', json.dumps({k: v for k, v in row.items() if k not in ['elapsed_minutes']}, allow_nan=False), flush=True)
            bundle()
            recent[:] = 0
            recent_n = 0
        if timed_out and step < max_step:
            print('Hết thời gian đã đặt. Đã lưu checkpoint và ZIP; CHƯA đánh giá test.', flush=True)
            return
    print('TRAIN COMPLETE. Chọn checkpoint:', 'Macro-F1 validation' if use_val else 'bước cuối cố định', flush=True)


def evaluate():
    state = json.loads((RUN / 'training_state.json').read_text())
    if not state['complete']:
        raise RuntimeError('Train chưa đủ ngân sách đã đặt; không tự đánh giá test.')
    if (RUN / 'test_complete.json').exists():
        for name in ['test_metrics.json', 'test_per_class.csv', 'test_predictions.npz']:
            if not (RUN / name).is_file():
                raise RuntimeError('Kết quả test thiếu file: ' + name)
        print('Đã có test cuối của run này. Không đánh giá lại.', flush=True)
        return
    # Build only the classifier and load protocol-selected weights.
    order = json.loads((RUN / 'model_class_order.json').read_text())['original_class_ids']
    shape = (CFG['patch_size'], CFG['patch_size'], CFG['pca_components'], 1)
    C = NT['denseMlpCreate'](shape, len(order))
    selected = json.loads((RUN / 'selection.json').read_text())
    C.load_weights(str(RUN / selected['classifier_weights']))
    test = np.load(RUN / 'test_locked.npz', allow_pickle=False)
    x, original_y = test['X_test'], test['original_class_ids']
    lookup = {int(cid): i for i, cid in enumerate(order)}
    y = np.array([lookup[int(cid)] for cid in original_y], dtype=np.int64)
    prob = predict(C, x, CFG['eval_batch_size'])
    pred = prob.argmax(1)
    values, (precision, recall, f1, support) = metrics(y, pred, len(order))
    atomic_json(RUN / 'test_metrics.json', dict(**values, selected_step=selected['step'],
                evaluated_classes=order, protocol=CFG['protocol'], selection_rule=selected['rule'], patch_overlap_expected=True, seed=CFG['train_seed']))
    with open(RUN / 'test_per_class.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'precision', 'recall', 'f1', 'test_support'])
        for i, cid in sorted(enumerate(order), key=lambda item: item[1]):
            writer.writerow([cid, precision[i], recall[i], f1[i], int(support[i])])
    np.savez_compressed(RUN / 'test_predictions.npz', coords=test['coords_test'],
                        true_class_ids=original_y, predicted_class_ids=np.asarray(order)[pred],
                        probabilities=prob, probability_class_ids=np.asarray(order),
                        confusion_matrix=confusion_matrix(y, pred, labels=np.arange(len(order))))
    atomic_json(RUN / 'test_complete.json', dict(complete=True, selected_step=selected['step']))
    print('FINAL TEST — DIP, đầy đủ các lớp đã cấu hình; xem audit chồng lấn:', values, flush=True)


def samples():
    """Diagnostics use training supports only; no filtering or quality claim is implied."""
    state = json.loads((RUN / 'training_state.json').read_text())
    if not state['complete']:
        raise RuntimeError('Chưa hoàn thành huấn luyện.')
    if (RUN / 'samples_complete.json').exists():
        for name in ['generated_examples.npz', 'generation_diagnostics.csv']:
            if not (RUN / name).is_file():
                raise RuntimeError('Minh họa mẫu sinh thiếu file: ' + name)
        print('Đã có mẫu minh họa của checkpoint được chọn.', flush=True)
        return
    cache = np.load(RUN / 'train_val.npz', allow_pickle=False)
    m = make_models(cache['X_train'], cache['y_train'], int(CFG['train_seed']))
    selection = json.loads((RUN / 'selection.json').read_text())
    m['checkpoint'].restore(str(RUN / selection['checkpoint'])).expect_partial()
    ids = np.asarray(CFG['accepted_class_ids'])[m['order']]
    rng = np.random.default_rng(1701)
    examples, real_examples, sample_ids, rows, generated_probs = [], [], [], [], []
    from scipy.spatial.distance import cdist
    for i, head in enumerate(m['heads']):
        condition = np.zeros((16, m['c']), dtype='float32')
        condition[:, i] = 1
        feature = m['G'].predict_on_batch([rng.normal(size=(16, 100)), condition])
        fake = head.predict_on_batch(feature)
        generated_probs.append(predict(m['C'], fake, CFG['eval_batch_size']))
        weights_model = keras.Model(head.inputs, head.layers[1].output)
        weights = weights_model.predict_on_batch(feature)
        h = -(weights * np.log(np.maximum(weights, 1e-12))).sum(axis=1)
        support = m['supports'][i]
        distance = cdist(fake.reshape(16, -1), support.reshape(len(support), -1), metric='euclidean')
        nearest = distance.argmin(axis=1)
        real_examples.append(support[nearest])
        examples.append(fake)
        sample_ids.extend([int(ids[i])] * len(fake))
        rows.append(dict(class_id=int(ids[i]), real_train=len(support),
                         normalized_entropy=float((h / np.log(len(support))).mean()),
                         effective_supports=float(np.exp(h).mean()),
                         mean_max_weight=float(weights.max(axis=1).mean()),
                         nearest_real_RMSE=float((distance.min(axis=1) / np.sqrt(np.prod(m['shape']))).mean()),
                         variance_ratio=float(fake.var(axis=0).mean() / (support.var(axis=0).mean() + 1e-12))))
    np.savez_compressed(RUN / 'generated_examples.npz', fake=np.concatenate(examples),
                        nearest_real=np.concatenate(real_examples), original_class_ids=np.asarray(sample_ids),
                        probabilities=np.concatenate(generated_probs), probability_class_ids=ids)
    with open(RUN / 'generation_diagnostics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(RUN / 'samples_complete.json', dict(complete=True, selected_step=selection['step']))
    print('Đã lưu mẫu và chẩn đoán theo lớp; chưa phải bằng chứng GAN cải thiện phân loại.', flush=True)



def infer():
    """Predict user-supplied preprocessed patches with the selected classifier."""
    if not (RUN / 'test_complete.json').exists():
        raise RuntimeError('Hoàn thành test cố định trước khi dùng ứng dụng minh họa.')
    if not args.input or not args.output:
        raise ValueError('Cần --input và --output cho infer.')
    destination = Path(args.output).resolve()
    if not destination.is_relative_to(RUN):
        raise ValueError('Đầu ra ứng dụng phải nằm trong run.')
    x = np.load(args.input, allow_pickle=False)['patches']
    shape = (CFG['patch_size'], CFG['patch_size'], CFG['pca_components'], 1)
    if tuple(x.shape[1:]) != shape or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('Shape hoặc dữ liệu ứng dụng không hợp lệ.')
    order = json.loads((RUN / 'model_class_order.json').read_text())['original_class_ids']
    C = NT['denseMlpCreate'](shape, len(order))
    selected = json.loads((RUN / 'selection.json').read_text())
    C.load_weights(str(RUN / selected['classifier_weights']))
    prob = predict(C, x, CFG['eval_batch_size'])
    np.savez_compressed(destination, probabilities=prob, class_ids=np.asarray(order))
    print('Ứng dụng: đã phân lớp', len(x), 'vùng ảnh; không cộng vào độ chính xác test.', flush=True)


try:
    if args.stage == 'smoke':
        smoke()
    elif args.stage == 'train':
        train()
    elif args.stage == 'samples':
        samples()
    elif args.stage == 'evaluate':
        evaluate()
    else:
        infer()
except Exception:
    (RUN / f'error_{args.stage}.txt').write_text(traceback.format_exc(), encoding='utf-8')
    raise
finally:
    # Persist partial results even if a normal Python exception occurs.
    # A forcibly killed Kaggle VM cannot run this finally block: Save Version is still required.
    bundle()
