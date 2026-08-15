"""第39C轮结束后文献检索：三源（OpenAlex + Crossref + arXiv）双源以上交叉核验。

本轮新问题（由 39A/39C 结果直接产生）：
  N1 遮蔽率/块结构在低信噪比、非平稳时序上如何选择？为什么高遮蔽在冗余信号上更好？
  N2 自监督预训练规模（池大小、epoch）与下游小样本回归增益的关系
  N3 跨数据源（不同台网/仪器）迁移时的归一化与幅度不变性处理
  N4 从批内相对量到绝对量的最优收缩/校准估计
  N5 集成与蒸馏在跨域回归上的收益与失败条件
  N6 地震震级与地面运动强度回归的深度模型评估方法
"""
import json, os, re, time, urllib.parse, urllib.request

OUT = '/root/5.6+chanshui1/outputs/round39c_lit'
os.makedirs(OUT, exist_ok=True)
UA = {'User-Agent': 'dizheng-research/1.0 (mailto:research@example.org)'}

TITLES = [
 ('N1', 'Masked Autoencoders Are Scalable Vision Learners'),
 ('N1', 'SimMIM: A Simple Framework for Masked Image Modeling'),
 ('N1', 'Ti-MAE: Self-Supervised Masked Time Series Autoencoders'),
 ('N1', 'SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling'),
 ('N1', 'Self-supervised Learning for Anomaly Detection in Time Series'),
 ('N2', 'Scaling Laws for Neural Language Models'),
 ('N2', 'Data Scaling Laws in Imitation Learning'),
 ('N2', 'How Well Do Self-Supervised Models Transfer?'),
 ('N2', 'Rethinking Pre-training and Self-training'),
 ('N3', 'Instance Normalization: The Missing Ingredient for Fast Stylization'),
 ('N3', 'Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift'),
 ('N3', 'Test-Time Training with Self-Supervision for Generalization under Distribution Shifts'),
 ('N3', 'Tent: Fully Test-Time Adaptation by Entropy Minimization'),
 ('N4', 'Calibrating Deep Neural Networks using Focal Loss'),
 ('N4', 'On Calibration of Modern Neural Networks'),
 ('N4', 'Distribution-Free Prediction Sets'),
 ('N4', 'Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification'),
 ('N5', 'Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles'),
 ('N5', 'Ensemble Distribution Distillation'),
 ('N5', 'Self-Distillation Amplifies Regularization in Hilbert Space'),
 ('N5', 'Does Knowledge Distillation Really Work?'),
 ('N6', 'Machine Learning Seismic Wave Discrimination: Application to Earthquake Early Warning'),
 ('N6', 'Deep learning for magnitude estimation of earthquakes using single station records'),
 ('N6', 'MagNet: A machine learning framework for magnitude estimation'),
 ('N6', 'Ground motion prediction equations and machine learning'),
 ('N6', 'Rapid estimation of earthquake magnitude from the first few seconds of P wave'),
]

def get(u, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception:
            if k == tries - 1:
                return ''
            time.sleep(3)
    return ''

def inv(d):
    if not d:
        return ''
    pos = {}
    for w, ps in d.items():
        for p in ps:
            pos[p] = w
    return ' '.join(pos[k] for k in sorted(pos))

def sim(a, b):
    na = set(re.sub(r'[^a-z0-9 ]', ' ', (a or '').lower()).split())
    nb = set(re.sub(r'[^a-z0-9 ]', ' ', (b or '').lower()).split())
    return len(na & nb) / max(1, len(na | nb))

def oa(t):
    d = get('https://api.openalex.org/works?search=' + urllib.parse.quote(t) + '&per-page=4')
    if not d:
        return None
    try:
        rs = json.loads(d).get('results', [])
    except Exception:
        return None
    best, bs = None, 0
    for w in rs:
        s = sim(t, w.get('title'))
        if s > bs:
            best, bs = w, s
    if not best or bs < 0.3:
        return None
    return dict(title=best.get('title'), year=best.get('publication_year'), doi=best.get('doi'),
                cited=best.get('cited_by_count'),
                venue=((best.get('primary_location') or {}).get('source') or {}).get('display_name'),
                abstract=inv(best.get('abstract_inverted_index')), match=round(bs, 3))

def ax(t):
    d = get('http://export.arxiv.org/api/query?search_query=' + urllib.parse.quote('ti:"' + t[:80] + '"') + '&max_results=2')
    m = re.search(r'<entry>(.*?)</entry>', d or '', re.S)
    if not m:
        return None
    e = m.group(1)
    ti = re.search(r'<title>(.*?)</title>', e, re.S)
    ab = re.search(r'<summary>(.*?)</summary>', e, re.S)
    idm = re.search(r'<id>(.*?)</id>', e, re.S)
    return dict(title=(ti.group(1).strip().replace('\n', ' ') if ti else ''),
                url=(idm.group(1).strip() if idm else ''),
                abstract=(re.sub(r'\s+', ' ', ab.group(1)).strip() if ab else ''))

def cr(t):
    d = get('https://api.crossref.org/works?query.title=' + urllib.parse.quote(t) + '&rows=1')
    if not d:
        return None
    try:
        it = json.loads(d)['message']['items']
    except Exception:
        return None
    if not it:
        return None
    w = it[0]
    if sim(t, (w.get('title') or [''])[0]) < 0.3:
        return None
    return dict(title=(w.get('title') or [''])[0], doi=w.get('DOI'),
                year=(w.get('issued', {}).get('date-parts', [[None]])[0][0]),
                venue=(w.get('container-title') or [None])[0], cited=w.get('is-referenced-by-count'),
                abstract=re.sub(r'<[^>]+>', ' ', w.get('abstract', '') or '').strip())

# 加严匹配：题名相似度阈值提高，并且摘要必须来自与查询题名一致的记录。
# 动机：宽松匹配会把不同论文的摘要混入（例如把 Scaling Laws 的摘要替换成无关文章），
# 使"精读"内容与题名不符。此处对每个来源单独校验题名一致性后再取摘要。
TITLE_SIM_MIN = 0.62

def consistent(rec, query):
    if not rec:
        return None
    if sim(query, rec.get('title') or '') < TITLE_SIM_MIN:
        return None
    return rec

recs = []
for grp, t in TITLES:
    a, b, c = consistent(oa(t), t), consistent(ax(t), t), consistent(cr(t), t)
    time.sleep(1.2)
    srcs = [n for n, v in (('openalex', a), ('arxiv', b), ('crossref', c)) if v]
    cands = [(v or {}).get('abstract', '') or '' for v in (a, b, c)]
    abst = max(cands, key=len) if cands else ''
    recs.append(dict(group=grp, query=t, sources=srcs, n_sources=len(srcs),
                     title_sim_min=TITLE_SIM_MIN,
                     resolved_title=((a or b or c) or {}).get('title'),
                     openalex=a, arxiv=b, crossref=c, abstract=abst, abstract_len=len(abst)))
    print('%s | %-28s | absLen=%5d | %s' % (grp, ','.join(srcs) or '-', len(abst), t[:70]), flush=True)

json.dump(recs, open(f'{OUT}/readlist.json', 'w'), indent=1)
ok = [r for r in recs if r['n_sources'] >= 2 and r['abstract_len'] > 200]
print('TOTAL', len(recs), 'MULTISOURCE_WITH_ABSTRACT', len(ok), flush=True)