"""Saved-output summaries. Never opens a classifier, optimizer, GPU, or test tensor."""
from __future__ import annotations
import csv
from pathlib import Path
from statistics import mean, stdev
from .audit import ARMS, SEEDS, csv_from_zip


def read_csv_file_or_zip(source, suffix):
    path=Path(source)
    if path.is_file() and path.suffix.lower()=='.zip':
        return csv_from_zip(path,suffix)
    if path.is_file() and path.name==suffix:
        with path.open('r',encoding='utf-8-sig',newline='') as f:
            return list(csv.DictReader(f))
    if path.is_dir():
        matches=list(path.rglob(suffix))
        if len(matches)!=1: raise ValueError(f'Expected one {suffix}, found {len(matches)}')
        with matches[0].open('r',encoding='utf-8-sig',newline='') as f:
            return list(csv.DictReader(f))
    raise FileNotFoundError(source)


def metrics_by_seed(source):
    rows=read_csv_file_or_zip(source,'test_by_seed.csv')
    keys={(x['arm'],int(x['seed'])) for x in rows}
    if keys!={(arm,seed) for arm in ARMS for seed in SEEDS} or len(rows)!=12:
        raise ValueError('Incomplete M4 test table')
    for row in rows:
        row['seed']=int(row['seed'])
        for k in ('OA','AA','Macro_F1','GM','Kappa'):
            row[k]=float(row[k])
    return rows


def test_paired_deltas(rows):
    keys={(r['arm'],int(r['seed'])):r for r in rows}
    expected={(arm,seed) for arm in ARMS for seed in SEEDS}
    if set(keys)!=expected:raise ValueError('Incomplete M4 seed registry')
    pairs=[]
    for seed in SEEDS:
        learned=keys['learned_local',seed]
        for baseline in ('original_gan','matched_ros','random_local'):
            row=keys[baseline,seed]
            pairs.append({'seed':seed,'against':baseline,
                          'macro_f1_pp':100*(learned['Macro_F1']-row['Macro_F1']),
                          'oa_pp':100*(learned['OA']-row['OA'])})
    return pairs


def recompute_summary(rows):
    """Matches test worker: sample standard deviation (ddof=1)."""
    result=[]
    for arm in ARMS:
        samples=[r for r in rows if r['arm']==arm]
        if len(samples)!=3 or {r['seed'] for r in samples}!={42,43,44}:
            raise ValueError('Wrong seed count')
        row={'arm':arm,'seeds':3}
        for metric in ('OA','AA','Macro_F1','GM','Kappa'):
            vals=[r[metric] for r in samples]
            row[metric+'_mean']=mean(vals)
            row[metric+'_sd']=stdev(vals)
        result.append(row)
    return result


def check_against_stored(computed, stored, tolerance=1e-9):
    stored_by_arm={r['arm']:r for r in stored}
    if set(stored_by_arm)!=set(ARMS):raise ValueError('Stored summary has wrong arms')
    for c in computed:
        original=stored_by_arm[c['arm']]
        for k,v in c.items():
            if k in ('arm','seeds'):continue
            if abs(float(original[k])-v)>tolerance:
                raise ValueError(f'Mismatch in {c["arm"]} {k}: {v} vs {original[k]}')
    return True


def weighting_tables(source):
    """Read completed train-only endpoint tables without model forward."""
    filenames=('per_seed_equal_class_summary.csv','matched_seed_pair_deltas.csv')
    return {filename:read_csv_file_or_zip(source,filename) for filename in filenames}
