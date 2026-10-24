"""Read-only artifact checks for the reorganized Indian Pines research project."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

ARMS = ("original_gan", "matched_ros", "random_local", "learned_local")
SEEDS = (42, 43, 44)
PROTOCOL_ID = "50e988fae260"
M4_SHA = "6d33ca6a0d39600f5f44c0d7e63547b1237149d625471ac73315d982ffbd8fb5"
MAX_ZIP_ENTRY = 10_000_000


def root_of(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents, start / 'KLTN_clean']:
        if (candidate / 'manifest.json').is_file() and (candidate / 'src').is_dir():
            return candidate
    raise FileNotFoundError('Project root not found. Open notebook from extracted KLTN_clean folder.')


def digest(path, block=1024*1024):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(block),b''):
            h.update(b)
    return h.hexdigest()


def manifest(root):
    return json.loads((Path(root)/'manifest.json').read_text(encoding='utf-8'))


def verify_sources(root, prefixes=('src/frozen/',)):
    root=Path(root)
    records=[x for x in manifest(root)['files'] if any(x['path'].startswith(t) for t in prefixes)]
    checks=[]
    for item in records:
        file=root/item['path']
        actual=digest(file) if file.is_file() else None
        checks.append({'file':item['path'],'status':'PASS' if actual==item['sha256'] else 'FAIL',
                       'sha256':actual,'expected':item['sha256']})
    if not records or any(r['status']!='PASS' for r in checks):
        raise RuntimeError('Frozen source validation failed. See checks before proceeding.')
    return checks


def source_functions(file):
    import ast
    path=Path(file)
    data=path.read_text(encoding='utf-8')
    tree=ast.parse(data,filename=path.name)
    return [(n.name,n.lineno) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]


def function_source(file, names):
    import ast
    data=Path(file).read_text(encoding='utf-8')
    tree=ast.parse(data)
    wanted=set(names)
    found={node.name:ast.get_source_segment(data,node) for node in tree.body
           if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in wanted}
    if wanted-set(found):raise KeyError(sorted(wanted-set(found)))
    return found


def scan(roots, extensions=('.zip','.json','.csv','.npz','.mat'), max_files=12000):
    items=[]
    for base in roots:
        if not base:continue
        base=Path(base).expanduser()
        if base.is_file():
            items.append(base)
            continue
        if not base.is_dir():continue
        count=0
        for p in base.rglob('*'):
            count+=1
            if count>max_files:
                raise RuntimeError(f'Too many entries under {base}; specify a smaller input directory')
            if p.is_file() and p.suffix.lower() in extensions:
                items.append(p)
    return sorted(set(items),key=lambda p:str(p))


def zip_listing(path):
    with zipfile.ZipFile(path,'r') as z:
        bad=[i.filename for i in z.infolist() if i.filename.startswith('/') or '..' in Path(i.filename).parts]
        if bad:raise ValueError('Unsafe ZIP entries: '+str(bad[:3]))
        return [x.filename for x in z.infolist() if not x.is_dir()]


def read_zip_member(z, name):
    info=z.getinfo(name)
    if info.file_size>MAX_ZIP_ENTRY:raise ValueError('Large metadata entry: '+name)
    return z.read(name)


def zip_matches(path, suffix):
    with zipfile.ZipFile(path) as z:
        matches=[n for n in z.namelist() if n.endswith(suffix)]
    return matches


def info(path):
    p=Path(path)
    return {'name':p.name,'size':p.stat().st_size,'sha256':digest(p)}


def npz_headers(path):
    """Read array shapes only; do not allocate full NumPy arrays."""
    import numpy as np
    p=Path(path)
    result={}
    with zipfile.ZipFile(p) as z:
        for entry in z.namelist():
            if not entry.endswith('.npy'):continue
            with z.open(entry) as f:
                version=np.lib.format.read_magic(f)
                if version==(1,0):shape, order, dtype=np.lib.format.read_array_header_1_0(f)
                else:shape, order, dtype=np.lib.format.read_array_header_2_0(f)
            result[Path(entry).stem]={'shape':list(shape),'dtype':str(dtype),'fortran':order}
    return result


def csv_from_zip(path, suffix):
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith(suffix)]
        if len(names)!=1:raise ValueError(f'Expected 1 × {suffix}; found {len(names)}')
        return list(csv.DictReader(io.StringIO(read_zip_member(z,names[0]).decode('utf-8-sig'))))


def read_test_summary(path):
    p=Path(path)
    if p.is_dir():
        paths=list(p.rglob('test_summary_mean_sd.csv'))
        if len(paths)!=1:raise ValueError(f'Expected 1 summary: {len(paths)}')
        with paths[0].open(newline='',encoding='utf-8-sig') as f:
            rows=list(csv.DictReader(f))
    else:
        rows=csv_from_zip(p,'test_summary_mean_sd.csv')
    if len(rows)!=4 or {r.get('arm') for r in rows}!=set(ARMS):
        raise ValueError('Summary must contain four arms, no extras')
    return rows


def read_test_by_seed(path):
    p=Path(path)
    if p.is_dir():
        paths=list(p.rglob('test_by_seed.csv'))
        if len(paths)!=1:raise ValueError('Expected one test_by_seed.csv')
        with paths[0].open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    else:
        rows=csv_from_zip(p,'test_by_seed.csv')
    keys={(r.get('arm'),int(r['seed'])) for r in rows}
    expected={(a,s) for a in ARMS for s in SEEDS}
    if len(rows)!=12 or keys!=expected:raise ValueError('Test rows are not 4 arms × 3 seeds')
    return rows


def zip_job_contracts(path):
    """Read small contract metadata from a training ZIP without extracting weights."""
    rows=[]
    with zipfile.ZipFile(path) as z:
        for item in z.infolist():
            if not item.filename.endswith('/contract.json') and item.filename!='contract.json':continue
            if item.file_size > MAX_ZIP_ENTRY:continue
            try:
                data=json.loads(read_zip_member(z,item.filename))
            except (json.JSONDecodeError,UnicodeDecodeError):continue
            rows.append({'member':item.filename,'arm':data.get('arm'),
                         'seed':data.get('seed'),'protocol':data.get('protocol_B_id'),
                         'stage':data.get('stage')})
    return rows


def statuses(roots, terms):
    files=scan(roots,extensions=('.zip',))
    return {term:[str(f) for f in files if term.lower() in f.name.lower()] for term in terms}


def export_json(root, section, name, value):
    folder=Path(root)/'outputs'/section
    folder.mkdir(parents=True,exist_ok=True)
    out=folder/name
    out.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    return out


def resolve_inputs(root):
    """Portable project-relative paths. No paths depend on notebook working dir."""
    root=Path(root).resolve()
    raw=json.loads((root/'inputs.json').read_text(encoding='utf-8'))
    def resolve(value):
        if value is None:return None
        if isinstance(value,list):return [resolve(x) for x in value]
        path=Path(value).expanduser()
        path=(root/path).resolve() if not path.is_absolute() else path.resolve()
        if not path.exists():raise FileNotFoundError('Missing configured input: '+str(path))
        return str(path)
    return {key:resolve(value) for key,value in raw.items()}

def verified_m4_registry(paths):
    """Verify actual completed metadata and checkpoint trio for all twelve jobs."""
    found={}
    for path in paths:
        with zipfile.ZipFile(path) as z:
            names=set(z.namelist())
            for name in sorted(names):
                if not name.endswith('/contract.json'):continue
                c=json.loads(z.read(name))
                if c.get('arm') not in ARMS or c.get('stage')!='final':continue
                key=(c['arm'],int(c['seed']));prefix=name[:-len('contract.json')]
                done=json.loads(z.read(prefix+'completed.json'))
                meta=json.loads(z.read(prefix+'state/ckpt_08000/meta.json'))
                if c.get('protocol_B_id')!=PROTOCOL_ID or done.get('complete') is not True or done.get('c_updates')!=8000:
                    raise RuntimeError('Incomplete/wrong final M4 job: '+str(key))
                if meta.get('c_updates')!=8000 or meta.get('contract')!=c:raise RuntimeError('Checkpoint/contract mismatch: '+str(key))
                for leaf in ('weights.index','weights.data-00000-of-00001','meta.json'):
                    if prefix+'state/ckpt_08000/'+leaf not in names:raise RuntimeError('Missing checkpoint '+str(key))
                if key in found:raise RuntimeError('Duplicate job in registry '+str(key))
                found[key]={'arm':key[0],'seed':key[1],'stage':'final','status':done['status'],
                    'c_updates':done['c_updates'],'adam_clock':done['adam_clock'],'protocol':PROTOCOL_ID,'archive':str(path)}
    if set(found)!={(a,s) for a in ARMS for s in SEEDS}:raise RuntimeError('M4 registry must have all four arms x three seeds')
    return [found[k] for k in sorted(found)]
