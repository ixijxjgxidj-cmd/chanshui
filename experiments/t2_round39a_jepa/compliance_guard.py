"""运行时合规守卫：用审计钩子硬阻断任何比赛数据读取。

背景：远程实验目录里残留历史比赛衍生物（t2data/T2.*.mseed、r1_*、r2_probe.json、
c3_r1r2_*.json 等）。仅靠"脚本里没写路径"不足以证明没读；本模块在解释器层面拦截。

规则：
  1) 黑名单命中即抛异常（比赛包、真题波形、R1/R2 衍生物）。
  2) 数据类扩展名（.npy/.h5/.hdf5/.mseed/.sac/.zip/.bz2/.joblib/.pt/.csv）只允许
     出现在显式登记的白名单前缀下；其余一律拒绝。
  3) 非数据类文件（.py/.json/.log/库文件等）不受第 2 条限制，避免误伤解释器自身。
"""
import os, sys

DENY_TOKENS = ('t2data', 't1data', '08-exam', '08-an', '08exam',
               '第1轮', '第2轮', 'r1train', 'r2train', 'r1_train',
               'official_r1_to_r2', 'exam_aug6', 'crew_sp23')
DENY_BASENAME_PREFIX = ('r1_', 'r2_', 'c3_r1r2')
DATA_EXT = ('.npy', '.h5', '.hdf5', '.mseed', '.sac', '.zip', '.bz2',
            '.joblib', '.pt', '.pth', '.csv', '.parquet')

_allow = []
_installed = False
_violations = []

class ComplianceViolation(RuntimeError):
    pass

def allow(*prefixes):
    for p in prefixes:
        _allow.append(os.path.realpath(p))

def _check(path):
    try:
        rp = os.path.realpath(path)
    except Exception:
        return
    low = rp.lower()
    base = os.path.basename(low)
    for t in DENY_TOKENS:
        if t.lower() in low:
            raise ComplianceViolation(f'拒绝读取比赛数据（黑名单 {t}）: {rp}')
    for t in DENY_BASENAME_PREFIX:
        if base.startswith(t):
            raise ComplianceViolation(f'拒绝读取比赛衍生物（前缀 {t}）: {rp}')
    if low.endswith(DATA_EXT):
        if not any(rp == a or rp.startswith(a + os.sep) for a in _allow):
            raise ComplianceViolation(f'数据文件不在白名单内: {rp}\n白名单={_allow}')

def install():
    global _installed
    if _installed:
        return
    def hook(event, args):
        if event == 'open':
            p = args[0]
            if isinstance(p, (str, bytes, os.PathLike)):
                if isinstance(p, bytes):
                    p = p.decode('utf-8', 'replace')
                _check(os.fspath(p))
    sys.addaudithook(hook)
    _installed = True

def selftest(bad_path):
    """红-绿证据：白名单内可读、黑名单必须抛异常。"""
    try:
        open(bad_path, 'rb').close()
    except ComplianceViolation as e:
        return f'GUARD_BLOCKED {bad_path}: {e.__class__.__name__}'
    except FileNotFoundError:
        raise AssertionError(f'自检无效：{bad_path} 不存在，无法证明拦截生效')
    raise AssertionError(f'守卫失效：竟然读到了 {bad_path}')