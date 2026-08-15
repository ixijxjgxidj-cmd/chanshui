"""第39轮定向精读：按题名取回摘要，OpenAlex 与 arXiv/Crossref 双源核验。

精读清单直接对应第39A轮暴露的三个问题：
  A JEPA/掩码自监督在时序与波形上的有效条件（是否需要大预训练池、是否需要多域）
  B 单台站震级估计的跨区域偏移与站点项
  C 批内相对量到绝对量的标定（lam 收缩）与分布偏移下的稳健性
"""
import json, re, time, os, urllib.parse, urllib.request

OUT = '/root/5.6+chanshui1/outputs/round39_lit'
os.makedirs(OUT, exist_ok=True)
UA = {'User-Agent': 'dizheng-research/1.0 (mailto:research@example.org)'}

TITLES = [
 # A: 自监督/JEPA/掩码建模
 ('A', 'Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture'),
 ('A', 'Revisiting Feature Prediction for Learning Visual Representations from Video'),
 ('A', 'data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language'),
 ('A', 'TS-MAE: A masked autoencoder for time series representation learning'),
 ('A', 'A Time Series is Worth 64 Words: Long-term Forecasting with Transformers'),
 ('A', 'TS2Vec: Towards Universal Representation of Time Series'),
 ('A', 'Bootstrap your own latent: A new approach to self-supervised Learning'),
 ('A', 'HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units'),
 ('A', 'SSAST: Self-Supervised Audio Spectrogram Transformer'),
 ('A', 'Masked Autoencoders Are Scalable Vision Learners'),
 # B: 地震学模型与震级
 ('B', 'SeisLM: a Foundation Model for Seismic Waveforms'),
 ('B', 'A machine-learning approach for earthquake magnitude estimation'),
 ('B', 'Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking'),
 ('B', 'SeisBench - A Toolbox for Machine Learning in Seismology'),
 ('B', 'Which picker fits my data? A quantitative evaluation of deep learning based seismic pickers'),
 ('B', 'STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI'),
 ('B', 'INSTANCE - the Italian seismic dataset for machine learning'),
 ('B', 'Network-Based Earthquake Magnitude Determination via Deep Learning'),
 ('B', 'Earthquake Magnitude Estimation Based on Machine Learning: Application to Earthquake Early Warning System'),
 ('B', 'Curated Pacific Northwest AI-ready Seismic Dataset'),
 ('B', 'SeismicXM: A Cross-Task Foundation Model for Single-Station Seismic Waveform Processing'),
 # C: 标定/分布偏移/稳健性
 ('C', 'WILDS: A Benchmark of in-the-Wild Distribution Shifts'),
 ('C', 'In Search of Lost Domain Generalization'),
 ('C', 'Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution'),
 ('C', 'Surgical Fine-Tuning Improves Adaptation to Distribution Shifts'),
 ('C', 'Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization'),
 ('C', 'Distributionally Robust Neural Networks for Group Shifts'),
 ('C', 'Conformal Prediction Under Covariate Shift'),
 ('C', 'Deep Ensembles: A Loss Landscape Perspective'),
 ('C', 'Knowledge distillation: A good teacher is patient and consistent'),
]

def get(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception:
            if k == tries - 1:
                return ''
            time.sleep(3)
    return ''

def inv_to_abs(inv):
    if not inv:
        return ''
    pos = {}
    for w, ps in inv.items():
        for p in ps:
            pos[p] = w
    return ' '.join(pos[k] for k in sorted(pos))

def oa_lookup(title):
    u = 'https://api.openalex.org/works?search=' + urllib.parse.quote(title) + '&per-page=3'
    t = get(u)
    if not t:
        return None
    try:
        rs = json.loads(t).get('results', [])
    except Exception:
        return None
    def sim(a, b):
        na = set(re.sub(r'[^a-z0-9 ]', ' ', a.lower()).split())
        nb = set(re.sub(r'[^a-z0-9 ]', ' ', b.lower()).split())
        return len(na & nb) / max(1, len(na | nb))
    best, bs = None, 0
    for w in rs:
        s = sim(title, w.get('title') or '')
        if s > bs:
            best, bs = w, s
    if not best or bs < 0.35:
        return None
    return dict(title=best.get('title'), year=best.get('publication_year'),
                doi=best.get('doi'), cited=best.get('cited_by_count'),
                venue=((best.get('primary_location') or {}).get('source') or {}).get('display_name'),
                oa_url=(best.get('open_access') or {}).get('oa_url'),
                abstract=inv_to_abs(best.get('abstract_inverted_index')), match=round(bs, 3))

def arxiv_lookup(title):
    u = ('http://export.arxiv.org/api/query?search_query=' + urllib.parse.quote('ti:"' + title[:80] + '"') +
         '&max_results=2')
    t = get(u)
    if not t:
        return None
    m = re.search(r'<entry>(.*?)</entry>', t, re.S)
    if not m:
        return None
    e = m.group(1)
    ti = re.search(r'<title>(.*?)</title>', e, re.S)
    ab = re.search(r'<summary>(.*?)</summary>', e, re.S)
    idm = re.search(r'<id>(.*?)</id>', e, re.S)
    return dict(title=(ti.group(1).strip().replace('\n', ' ') if ti else ''),
                url=(idm.group(1).strip() if idm else ''),
                abstract=(re.sub(r'\s+', ' ', ab.group(1)).strip() if ab else ''))

def crossref_lookup(title):
    u = 'https://api.crossref.org/works?query.title=' + urllib.parse.quote(title) + '&rows=1'
    t = get(u)
    if not t:
        return None
    try:
        it = json.loads(t)['message']['items']
    except Exception:
        return None
    if not it:
        return None
    w = it[0]
    return dict(title=(w.get('title') or [''])[0], doi=w.get('DOI'),
                venue=(w.get('container-title') or [None])[0],
                year=(w.get('issued', {}).get('date-parts', [[None]])[0][0]),
                cited=w.get('is-referenced-by-count'),
                abstract=re.sub(r'<[^>]+>', ' ', w.get('abstract', '') or '').strip())

recs = []
for grp, ti in TITLES:
    oa = oa_lookup(ti); time.sleep(1.0)
    ax = arxiv_lookup(ti); time.sleep(1.0)
    cr = crossref_lookup(ti); time.sleep(1.0)
    srcs = [n for n, v in (('openalex', oa), ('arxiv', ax), ('crossref', cr)) if v]
    abst = ''
    for v in (oa, ax, cr):
        if v and v.get('abstract') and len(v['abstract']) > len(abst):
            abst = v['abstract']
    recs.append(dict(group=grp, query_title=ti, sources=srcs, n_sources=len(srcs),
                     openalex=oa, arxiv=ax, crossref=cr, abstract=abst,
                     abstract_len=len(abst)))
    print('%s | src=%s | absLen=%d | %s' % (grp, ','.join(srcs) or '-', len(abst), ti[:78]), flush=True)

json.dump(recs, open(f'{OUT}/readlist.json', 'w'), indent=1)
ok = [r for r in recs if r['n_sources'] >= 2 and r['abstract_len'] > 200]
print('TOTAL', len(recs), 'MULTISOURCE_WITH_ABSTRACT', len(ok), flush=True)