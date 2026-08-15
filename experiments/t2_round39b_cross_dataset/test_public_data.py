"""第39B轮数据接口测试（先红后绿）。

被测对象：public_data.py —— 跨数据集实验的唯一数据入口。

必须成立的行为：
  T1 加载公开 STEAD 缓存，返回定长 (3,1000) 窗口与震级标签，且按事件分组可用。
  T2 加载 INSTANCE 缓存时，只接受官方 split 中的 train/dev；出现 test 必须抛异常。
  T3 事件级划分不允许同一 source_id 跨划分出现。
  T4 任何比赛数据路径（08/R1/R2 衍生物）在解释器层被硬阻断。
  T5 数据集描述里必须记录来源、样本数与内容哈希，便于复现。
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/root/5.6+chanshui1')

import public_data as pd

FAIL = []

def check(name, fn):
    try:
        fn()
        print('PASS', name, flush=True)
    except Exception as e:
        FAIL.append((name, repr(e)))
        print('FAIL', name, repr(e), flush=True)

def t1_stead():
    ds = pd.load_stead()
    assert ds.X.shape[1:] == (3, 1000), ds.X.shape
    assert ds.y.ndim == 1 and len(ds.y) == len(ds.X)
    assert len(ds.event_id) == len(ds.y)
    assert ds.name == 'stead'
    assert float(ds.y.min()) >= 3.0
    assert ds.region is not None and len(np.unique(ds.region)) >= 3

def t2_instance_rejects_test_split():
    """INSTANCE 官方 test 划分绝不允许进入实验。"""
    bad = np.array(['train', 'dev', 'test'])
    try:
        pd.assert_split_allowed(bad)
    except pd.SplitViolation:
        return
    raise AssertionError('未拒绝官方 test 划分')

def t2b_instance_accepts_train_dev():
    pd.assert_split_allowed(np.array(['train', 'dev', 'train']))

def t3_event_isolation():
    ev = np.array(['e1', 'e1', 'e2', 'e2', 'e3', 'e3', 'e4', 'e4'])
    a, b = pd.event_split(np.arange(len(ev)), ev, frac=0.5, seed=7)
    assert len(a) > 0 and len(b) > 0
    assert not (set(ev[a]) & set(ev[b])), (ev[a], ev[b])
    assert len(a) + len(b) == len(ev)

def t4_guard_blocks_competition():
    """比赛衍生物必须在文件打开层被拒绝。"""
    pd.enable_guard()
    for p in ['/root/5.6+chanshui1/t2data/T2.A.Q0001.mseed',
              '/root/5.6+chanshui1/outputs/r1_t2_meta.csv']:
        try:
            open(p, 'rb').close()
        except pd.ComplianceViolation:
            continue
        except FileNotFoundError:
            raise AssertionError(f'{p} 不存在，无法证明拦截生效')
        raise AssertionError(f'守卫失效：读到了 {p}')

def t5_provenance():
    ds = pd.load_stead()
    prov = ds.provenance()
    for k in ['name', 'source', 'n', 'sha256', 'protocol']:
        assert k in prov and prov[k], (k, prov)
    assert prov['n'] == len(ds.y)
    assert len(prov['sha256']) == 64

check('T1 stead loader', t1_stead)
check('T2 reject official test split', t2_instance_rejects_test_split)
check('T2b accept train/dev', t2b_instance_accepts_train_dev)
check('T3 event isolation', t3_event_isolation)
check('T4 guard blocks competition data', t4_guard_blocks_competition)
check('T5 provenance record', t5_provenance)

print('SUMMARY failures=%d' % len(FAIL), flush=True)
if FAIL:
    for n, e in FAIL:
        print('  -', n, e)
    sys.exit(1)
print('ALL_GREEN')