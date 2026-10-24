"""Fixed author DIP masks: retain all 16 classes and report, never hide, overlap."""
import hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter

CLASS_NAMES = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
    'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill',
    'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods',
    'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers']
DIP_MAPPING = np.array([0, 2, 3, 5, 6, 8, 10, 11, 12, 14, 1, 4, 7, 9, 13, 15, 16])
EXPECTED_TRAIN = np.array([29, 762, 435, 146, 232, 394, 16, 235, 10, 470, 1424, 328, 132, 728, 291, 57])
EXPECTED_TEST = np.array([25, 675, 404, 99, 274, 354, 2, 250, 10, 503, 1065, 282, 80, 545, 99, 44])


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dip_split(gt, encoded_mask, cfg):
    assert encoded_mask.shape == gt.shape == (145, 145)
    assert np.issubdtype(encoded_mask.dtype, np.integer)
    assert encoded_mask.min() >= 0 and encoded_mask.max() <= 16
    author_train = DIP_MAPPING[encoded_mask].astype('int16')
    test_labels = np.where(author_train == 0, gt, 0).astype('int16')
    assert np.array_equal([(author_train == c).sum() for c in range(1, 17)], EXPECTED_TRAIN)
    assert np.array_equal([(test_labels == c).sum() for c in range(1, 17)], EXPECTED_TEST)
    mismatch = (author_train > 0) & (author_train != gt)
    label_audit = dict(train_vs_standard_gt_disagreements=int(mismatch.sum()),
        assigned_on_gt_background=int(((author_train > 0) & (gt == 0)).sum()),
        assigned_different_nonzero_label=int((mismatch & (gt > 0)).sum()))
    # Reproduction keeps author train labels, rather than silently changing 161 samples.
    labels = np.where(author_train > 0, author_train, test_labels)
    split = np.full(gt.shape, -1, dtype='int8')
    split[author_train > 0] = 0
    split[test_labels > 0] = 2
    if cfg['protocol'] == 'dip_train_only_validation':
        rng = np.random.default_rng(cfg['split_seed'])
        for cid in range(1, 17):
            indices = np.flatnonzero(author_train.ravel() == cid)
            n_val = max(2, int(np.floor(len(indices) * cfg['validation_fraction'])))
            if n_val > len(indices) - 2:
                raise ValueError('Không đủ train/validation cho lớp ' + str(cid))
            split.ravel()[rng.permutation(indices)[:n_val]] = 1
    elif cfg['protocol'] != 'dip_reproduction':
        raise ValueError('Protocol không hợp lệ.')
    coords = [np.argwhere(split == i) for i in range(3)]
    return split, labels, coords, label_audit


def audit_overlap(split, patch_size):
    centers = [split == i for i in range(3)]
    footprints = [maximum_filter(m.astype('uint8'), size=patch_size, mode='constant') > 0 for m in centers]
    report = {}
    names = ['train', 'validation', 'test']
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        prefix = names[a] + '_' + names[b]
        report[prefix + '_shared_centers'] = int((centers[a] & centers[b]).sum())
        report[prefix + '_shared_raw_pixels'] = int((footprints[a] & footprints[b]).sum())
        # A patch intersects another patch when center distance <= 2 * radius.
        near_a = maximum_filter(centers[a].astype('uint8'), size=2 * patch_size - 1, mode='constant') > 0
        report[prefix + '_overlapping_centers_in_second_set'] = int((near_a & centers[b]).sum())
    assert all(v == 0 for k, v in report.items() if k.endswith('_shared_centers'))
    # Nonzero RAW footprint overlap is expected here; it is recorded, not asserted away.
    return report


def extract_patches(cube, coords, patch_size):
    if patch_size <= 0 or patch_size % 2 != 1:
        raise ValueError('patch_size phải lẻ, dương.')
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.issubdtype(coords.dtype, np.integer):
        raise ValueError('Tọa độ cần có dạng (N, 2), số nguyên.')
    if len(coords) and ((coords < 0).any() or (coords >= np.array(cube.shape[:2])).any()):
        raise ValueError('Tọa độ ngoài ảnh.')
    r = patch_size // 2
    padded = np.pad(cube, ((r, r), (r, r), (0, 0)), mode='constant')
    out = np.empty((len(coords), patch_size, patch_size, cube.shape[-1], 1), dtype='float32')
    for i, (row, col) in enumerate(coords):
        out[i, ..., 0] = padded[row:row+patch_size, col:col+patch_size]
    if not np.isfinite(out).all():
        raise ValueError('Dữ liệu có NaN/Inf.')
    return out
