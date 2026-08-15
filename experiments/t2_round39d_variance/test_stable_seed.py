"""稳定区域种子测试：禁止用 Python hash() 产生跨进程不稳定的实验切分。"""
import os, sys, subprocess, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_seed import region_seed

vals = {x: region_seed(x) for x in ['ALASKA', 'CALIF', 'GREECE', 'CHILE', 'OTHER']}
print('LOCAL', vals)
assert len(set(vals.values())) == len(vals)
assert all(isinstance(v, int) and 0 <= v < 2**31 for v in vals.values())

# 不同 PYTHONHASHSEED 的子解释器必须给出相同结果。
code = "import sys;sys.path.insert(0,r'/root/5.6+chanshui1');from stable_seed import region_seed;print(region_seed('GREECE'))"
out = []
for h in ['1', '999', 'random']:
    e = dict(os.environ, PYTHONHASHSEED=h)
    out.append(subprocess.check_output([sys.executable, '-c', code], env=e, text=True).strip())
print('CHILD', out)
assert len(set(out)) == 1 and int(out[0]) == vals['GREECE']
print('STABLE_SEED_ALL_GREEN')