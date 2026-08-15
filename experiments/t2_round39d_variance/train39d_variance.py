"""轮39D：种子方差基线——判断轮39C 的配置差异是否可区分于随机噪声（仅公开 STEAD）。

动机（由 39A 与 39C 的直接对照产生）：
  39A 的 JEPA-MR 与 39C 的 p8_m32_b10 在遮蔽率/块长/上下文比例上几乎相同，
  但同一 A->B->C 划分上的标定分相差最多 2.7 分（GREECE->CHILE->CALIF: 161.611 vs 158.913），
  而 39C 六个配置之间的平均分差仅约 0.2 分。
  因此"哪个遮蔽配置更好"可能完全被种子噪声吞没。本轮先估计噪声，再决定是否能选配置。

做法：固定两个代表配置，各用 3 个独立种子跑完 5 个划分；
     报告每划分的标准差、配置间差异，以及"配置差异 / 种子噪声"的比值。
     种子同时控制 JEPA 预训练、微调与预训练池抽样。

合规：只读公开 STEAD 缓存；白名单守卫；比赛包与 R1/R2 衍生物硬阻断；B 选 lam，C 一次评估。
"""
import os, sys, json, time
import numpy as np, torch

ROOT = '/root/5.6+chanshui1'
OUT = f'{ROOT}/outputs/t2_round39d_variance'
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)
from stable_seed import region_seed, stable_seed

# 复用 39C 的模型与评估实现，避免重写造成不一致；只替换实验循环。
src = open(f'{ROOT}/train39c_ablation.py').read()
head = src[:src.index('SETUPS = [')]
ns = {'__name__': 'round39d_base'}
exec(compile(head, 'round39d_base', 'exec'), ns)

pdata = ns['pdata']
_cg = sys.modules['compliance_guard']
_cg.allow(OUT)
print(_cg.selftest(f'{ROOT}/t2data/T2.A.Q0001.mseed'), flush=True)

Ym, SRCm, REG, idx = ns['Ym'], ns['SRCm'], ns['REG'], ns['idx']
jepa_pretrain, finetune, predict, evaluate = ns['jepa_pretrain'], ns['finetune'], ns['predict'], ns['evaluate']
N_TOK = {8: 125, 16: 63, 32: 32}

SETUPS = [('ALASKA', 'CALIF', 'GREECE'), ('CALIF', 'GREECE', 'ALASKA'), ('GREECE', 'CHILE', 'CALIF'),
          ('CHILE', 'ALASKA', 'NZ'), ('OTHER', 'NZ', 'CHILE')]
CFGS = [dict(tag='p8_m32_b10', patch=8, mask_ratio=.32, blk=10, ctx_keep=.60),
        dict(tag='p8_m48_b25', patch=8, mask_ratio=.48, blk=25, ctx_keep=.45)]
SEEDS = [11, 2029]
# 切分重复：同一区域用不同稳定切分种子，估计"事件切分"带来的方差。
# 这是本轮新增的关键量：轮39A 与 39C 的巨大差异来自 hash() 随机化导致的不同切分，
# 而非配置差异，必须把切分方差与模型种子方差分开报告。
SPLIT_REPS = [0, 1]
PRE_EPOCHS = int(os.environ.get('PRE_EPOCHS', '10'))
PRE_CAP = int(os.environ.get('PRE_CAP', '30000'))

rows = []
for (A, B, Cr) in SETUPS:
    key = f'{A}->{B}->{Cr}'
    mA, mB, mC = REG == A, REG == B, REG == Cr
    rows_b, rows_c = np.where(mB)[0], np.where(mC)[0]
    yb, yc = Ym[mB], Ym[mC]
    for rep in SPLIT_REPS:
        split_seed = region_seed(A) if rep == 0 else stable_seed('split_rep', A, rep)
        tr, ho = pdata.event_split(np.where(mA)[0], SRCm, 0.85, seed=split_seed)
        ho_set = set(ho.tolist())
        yi = Ym[ho]
        print('=== %s rep=%d split_seed=%d train=%d holdout=%d calib=%d eval=%d' %
              (key, rep, split_seed, len(tr), len(ho), mB.sum(), mC.sum()), flush=True)
        for cfg in CFGS:
            for seed in SEEDS:
                t0 = time.time()
                rs = np.random.RandomState(seed)
                pool = np.array([i for i in range(len(idx))
                                 if (not mB[i]) and (not mC[i]) and (i not in ho_set)])
                if len(pool) > PRE_CAP:
                    pool = np.sort(rs.choice(pool, PRE_CAP, replace=False))
                assert not mB[pool].any() and not mC[pool].any() and not (set(pool.tolist()) & ho_set)
                enc, _ = jepa_pretrain(idx[pool], seed, PRE_EPOCHS, cfg['patch'], N_TOK[cfg['patch']],
                                       cfg['mask_ratio'], cfg['blk'], cfg['ctx_keep'])
                mdl = finetune(enc, tr, seed)
                d = evaluate(predict(mdl, ho), yi, predict(mdl, rows_b), yb, predict(mdl, rows_c), yc)
                d.update(setup=key, split_rep=rep, split_seed=split_seed, seed=seed, tag=cfg['tag'], secs=time.time() - t0)
                rows.append(d)
                print('   %-11s seed=%-6d rho_eval=%.3f lam=%.2f score_calib=%.3f (%.0fs)' %
                  (cfg['tag'], seed, d['rho_eval'], d['lam_calib'], d['score_eval_lam_calib'], d['secs']),
                      flush=True)

def pick(**kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]

def sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

# 方差分解：固定(划分, 配置)只变模型种子 -> 种子方差；
#           固定(配置, 种子)只变事件切分 -> 切分方差。
seed_sds, split_sds = [], []
per_setup = {}
for (A, B, Cr) in SETUPS:
    key = f'{A}->{B}->{Cr}'
    e = dict(by_cfg={})
    for cfg in CFGS:
        t = cfg['tag']
        allv = [r['score_eval_lam_calib'] for r in pick(setup=key, tag=t)]
        s_seed = [sd([r['score_eval_lam_calib'] for r in pick(setup=key, tag=t, split_rep=rep)])
                  for rep in SPLIT_REPS]
        s_split = [sd([r['score_eval_lam_calib'] for r in pick(setup=key, tag=t, seed=s)])
                   for s in SEEDS]
        seed_sds += s_seed
        split_sds += s_split
        e['by_cfg'][t] = dict(scores=allv, mean=float(np.mean(allv)), sd=sd(allv),
                              spread=float(max(allv) - min(allv)),
                              seed_sd_per_split=s_seed, split_sd_per_seed=s_split,
                              mean_seed_sd=float(np.mean(s_seed)), mean_split_sd=float(np.mean(s_split)))
    e['config_gap'] = float(e['by_cfg'][CFGS[1]['tag']]['mean'] - e['by_cfg'][CFGS[0]['tag']]['mean'])
    e['noise_sd'] = float(np.sqrt(np.mean([e['by_cfg'][c['tag']]['sd'] ** 2 for c in CFGS])))
    e['gap_over_noise'] = (abs(e['config_gap']) / e['noise_sd']) if e['noise_sd'] > 0 else None
    per_setup[key] = e

means = {}
for cfg in CFGS:
    t = cfg['tag']
    setup_means = [float(np.mean([r['score_eval_lam_calib'] for r in pick(setup=f'{a}->{b}->{c}', tag=t)]))
                   for (a, b, c) in SETUPS]
    run_means = [float(np.mean([r['score_eval_lam_calib']
                                for r in pick(tag=t, seed=s, split_rep=rep)]))
                 for s in SEEDS for rep in SPLIT_REPS]
    means[t] = dict(setup_means=setup_means, mean_of_setups=float(np.mean(setup_means)),
                    worst_setup=float(np.min(setup_means)),
                    per_run_mean=run_means, sd_of_run_means=sd(run_means))

gap = means[CFGS[1]['tag']]['mean_of_setups'] - means[CFGS[0]['tag']]['mean_of_setups']
run_noise = float(np.sqrt(np.mean([means[c['tag']]['sd_of_run_means'] ** 2 for c in CFGS])))
verdict = dict(config_gap_mean=float(gap),
               mean_seed_sd=float(np.mean(seed_sds)), max_seed_sd=float(np.max(seed_sds)),
               mean_split_sd=float(np.mean(split_sds)), max_split_sd=float(np.max(split_sds)),
               run_level_noise_sd=run_noise,
               gap_over_run_noise=(abs(gap) / run_noise if run_noise > 0 else None),
               config_distinguishable=bool(run_noise > 0 and abs(gap) > 2 * run_noise),
               split_dominates_seed=bool(np.mean(split_sds) > np.mean(seed_sds)))
res = dict(protocol='public STEAD only; A->B->C exclusive networks; event isolation; '
                    'stable SHA-256 split seeds (fixes cross-process hash() instability); '
                    'variance decomposition over model seeds and event splits; '
                    'B selects lam, C evaluated once; competition packages never read',
           data_provenance=ns['prov'], torch=torch.__version__, hip=torch.version.hip,
           device=ns['dev'], pre_epochs=PRE_EPOCHS, pre_cap=PRE_CAP,
           configs=CFGS, seeds=SEEDS, split_reps=SPLIT_REPS, setups=[list(s) for s in SETUPS],
           runs=rows, per_setup=per_setup, config_means=means, verdict=verdict)
json.dump(res, open(f'{OUT}/variance.json', 'w'), indent=2)
print('PER_SETUP', json.dumps({k: dict(gap=v['config_gap'], noise_sd=v['noise_sd'],
                                       gap_over_noise=v['gap_over_noise']) for k, v in per_setup.items()}, indent=1), flush=True)
print('CONFIG_MEANS', json.dumps(means, indent=1), flush=True)
print('VERDICT', json.dumps(verdict, indent=1), flush=True)