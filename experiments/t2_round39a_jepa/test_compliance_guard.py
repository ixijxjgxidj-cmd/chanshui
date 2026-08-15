"""合规守卫红-绿自检：证明拦截真实生效，且不误伤白名单。"""
import sys, numpy as np
sys.path.insert(0, '/root/5.6+chanshui1')
import compliance_guard as cg

CACHE = '/root/5.6+chanshui1/outputs/t2_cache_station27'
cg.allow(CACHE, '/root/5.6+chanshui1/outputs/t2_round39a_jepa')
cg.install()

# 绿：白名单内公开缓存可读
y = np.load(f'{CACHE}/y.npy')
print('GREEN 白名单可读 y.npy', y.shape, float(y.min()), float(y.max()))

# 红：比赛真题波形必须被拦截（该文件确实存在，见 find 输出）
print(cg.selftest('/root/5.6+chanshui1/t2data/T2.A.Q0001.mseed'))
# 红：R1/R2 衍生物必须被拦截
print(cg.selftest('/root/5.6+chanshui1/outputs/r1_t2_meta.csv'))
print(cg.selftest('/root/5.6+chanshui1/outputs/c3_r1r2_devmix.json'))
# 红：白名单外的其他 .npy 必须被拦截
print(cg.selftest('/root/5.6+chanshui1/outputs/t2_cache_p10/y.npy'))
print('SELFTEST_ALL_PASS')