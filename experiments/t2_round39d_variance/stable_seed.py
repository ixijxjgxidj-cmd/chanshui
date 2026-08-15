"""跨进程稳定的实验种子。

背景缺陷：轮38/39A/39C 用 `hash(region_name) % 9999` 作为事件切分种子。
CPython 默认对 str 的 hash 加盐（PYTHONHASHSEED=random），因此**每个进程**得到不同切分。
后果：同名 A->B->C 组合在不同轮次并不是同一实验单元，轮次之间的数字不可直接比较
（实测 GREECE->CHILE->CALIF 因此出现 2.7 分级差异，远超配置差异）。

本模块用 SHA-256 生成确定性种子，保证同名划分在任何进程、任何机器上一致。
"""
import hashlib

def stable_seed(*parts, mod=2**31 - 1):
    key = '|'.join(str(p) for p in parts).encode('utf-8')
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % mod

def region_seed(region, protocol='t2_cross_region_v1'):
    return stable_seed(protocol, region)