"""第39B轮公共数据唯一入口。

只允许经审计的公开数据缓存：
  * STEAD：/root/5.6+chanshui1/outputs/t2_cache_station27
  * INSTANCE：/root/5.6+chanshui1/outputs/t2_cache_instance31（仅缓存审计通过后）

绝不读取比赛包、其派生波形、标签、缓存或预测。INSTANCE 官方 test split 绝不进入本接口。
"""
from dataclasses import dataclass
import hashlib
import os
import sys
import numpy as np

ROOT = '/root/5.6+chanshui1'
STEAD = f'{ROOT}/outputs/t2_cache_station27'
INSTANCE = f'{ROOT}/outputs/t2_cache_instance31'

class SplitViolation(ValueError):
    pass

try:
    sys.path.insert(0, ROOT)
    import compliance_guard as _cg
    ComplianceViolation = _cg.ComplianceViolation
except Exception:
    class ComplianceViolation(RuntimeError):
        pass
    _cg = None

_guard_enabled = False

def enable_guard():
    """在当前解释器中安装文件访问守卫，只放行两个公开缓存路径。"""
    global _guard_enabled
    if _cg is None:
        raise RuntimeError('缺少 compliance_guard；拒绝在无守卫状态运行')
    _cg.allow(STEAD, INSTANCE, f'{ROOT}/outputs/t2_round39b_cross_dataset')
    _cg.install()
    _guard_enabled = True

def assert_split_allowed(split):
    a = np.asarray(split).astype(str)
    allowed = {'train', 'dev'}
    seen = set(a.tolist())
    bad = seen - allowed
    if bad:
        raise SplitViolation('INSTANCE 只允许官方 train/dev，拒绝: ' + ','.join(sorted(bad)))
    if not seen:
        raise SplitViolation('split 不能为空')

def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

@dataclass
class PublicDataset:
    name: str
    source: str
    protocol: str
    root: str
    X: object
    y: np.ndarray
    event_id: np.ndarray
    network: np.ndarray
    station: np.ndarray
    region: np.ndarray | None = None
    split: np.ndarray | None = None

    def provenance(self):
        # y 哈希固定且廉价，X 由 shape/protocol 同时约束；完整 X 哈希在构建缓存时记录。
        return dict(name=self.name, source=self.source, protocol=self.protocol, n=int(len(self.y)),
                    sha256=_sha256(os.path.join(self.root, 'y.npy')))

def _load_common(root, name, source, protocol, require_split=False):
    need = ['X.npy', 'y.npy', 'src.npy', 'net.npy', 'sta.npy']
    missing = [x for x in need if not os.path.isfile(os.path.join(root, x))]
    if missing:
        raise FileNotFoundError(f'{name} 缓存不完整，缺少: {missing}')
    X = np.load(os.path.join(root, 'X.npy'), mmap_mode='r')
    y = np.load(os.path.join(root, 'y.npy'))
    src = np.load(os.path.join(root, 'src.npy')).astype(str)
    net = np.load(os.path.join(root, 'net.npy')).astype(str)
    sta = np.load(os.path.join(root, 'sta.npy')).astype(str)
    n = len(y)
    if X.ndim != 3 or tuple(X.shape[1:]) != (3, 1000):
        raise ValueError(f'{name} X 必须为 (n,3,1000)，实际 {X.shape}')
    if not all(len(a) == n for a in [src, net, sta]):
        raise ValueError(f'{name} 元数据长度与标签不一致')
    if not np.isfinite(y).all():
        raise ValueError(f'{name} 标签含非有限数')
    spl = None
    sp = os.path.join(root, 'split.npy')
    if require_split:
        if not os.path.isfile(sp):
            raise FileNotFoundError('INSTANCE 缓存缺 split.npy，拒绝使用')
        spl = np.load(sp).astype(str)
        if len(spl) != n:
            raise ValueError('INSTANCE split 长度不一致')
        assert_split_allowed(spl)
    return PublicDataset(name, source, protocol, root, X, y, src, net, sta, split=spl)

def load_stead():
    ds = _load_common(STEAD, 'stead', '公开 STEAD', 'P-5s..P+5s, 100 Hz, 3C')
    groups = {'ALASKA': ['AK','AT','AV'], 'CALIF': ['CI','BK','AZ','NN'],
              'GREECE': ['HL','HP','HT','HA'], 'CHILE': ['C','C1','PB'], 'NZ': ['NZ'],
              'OTHER': ['TA','OK','GS','PR','SV','IU','MN','CN','KR','GO']}
    lookup = {n:r for r, ns in groups.items() for n in ns}
    ds.region = np.array([lookup.get(n, 'DROP') for n in ds.network])
    return ds

def load_instance():
    return _load_common(INSTANCE, 'instance', '公开 INSTANCE（官方 train/dev）',
                        'P-5s..P+5s, 100 Hz, 3C', require_split=True)

def event_split(rows, event_id, frac, seed):
    rows = np.asarray(rows, dtype=np.int64)
    eid = np.asarray(event_id).astype(str)
    if not (0.0 < frac < 1.0):
        raise ValueError('frac 必须介于 0 与 1')
    ev = np.unique(eid[rows])
    if len(ev) < 2:
        raise ValueError('至少需要两个事件')
    rng = np.random.RandomState(seed)
    rng.shuffle(ev)
    n = min(max(1, int(len(ev) * frac)), len(ev) - 1)
    left = set(ev[:n])
    mask = np.array([eid[i] in left for i in rows])
    a, b = rows[mask], rows[~mask]
    if set(eid[a]) & set(eid[b]):
        raise AssertionError('事件泄漏')
    return a, b