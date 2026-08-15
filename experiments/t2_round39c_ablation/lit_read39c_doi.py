"""第39C轮文献精读（DOI 级双源核验版）。

修正的方法缺陷：按猜测题名检索会把不同论文的摘要错配到题名上（上一版出现
Scaling Laws / Distribution-Free Prediction Sets 等条目摘要与题名不符）。
本版改为：
  1) 用主题查询在 OpenAlex 取候选（带摘要、带 DOI、按被引排序）；
  2) 用该 DOI 去 Crossref /works/{doi} 精确取权威记录；
  3) 只有当两源的题名相似度 >= 0.75 时，才认定该论文双源一致；
  4) arXiv 作为可选第三源，仅在题名一致时登记。
这样"题名—摘要—DOI"三者绑定，精读内容与引用可核对。

本轮问题：
  N1 遮蔽率与块结构在低信噪比非平稳时序上的选择
  N2 自监督预训练规模与下游小样本回归增益
  N3 跨台网/仪器迁移的归一化与幅度不变性
  N4 相对量到绝对量的收缩与校准
  N5 集成与蒸馏在跨域回归上的收益与失败条件
  N6 地震震级/强度回归的深度模型评估
"""
import json, os, re, time, urllib.parse, urllib.request

OUT = '/root/5.6+chanshui1/outputs/round39c_lit'
os.makedirs(OUT, exist_ok=True)
UA = {'User-Agent': 'dizheng-research/1.0 (mailto:research@example.org)'}

QUERIES = {
 'N1': ['masked image modeling mask ratio study',
        'masked time series modeling pre-training',
        'block masking strategy self-supervised representation'],
 'N2': ['scaling self-supervised pre-training data size downstream transfer',
        'how much pretraining data self-supervised transfer study'],
 'N3': ['instance normalization distribution shift time series forecasting',
        'test-time adaptation distribution shift entropy',
        'amplitude invariance normalization deep learning signals'],
 'N4': ['calibration of regression neural networks uncertainty',
        'conformal prediction regression covariate shift weighted',
        'shrinkage estimation prediction relative to absolute scale'],
 'N5': ['ensemble distillation uncertainty regression',
        'knowledge distillation limitations fidelity study',
        'deep ensembles out-of-distribution regression robustness'],
 'N6': ['deep learning earthquake magnitude estimation single station',
        'earthquake early warning magnitude machine learning evaluation',
        'ground motion intensity prediction deep learning'],
}

CACHE = f'{OUT}/http_cache.json'
_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
_last = [0.0]
MIN_INTERVAL = 2.0     # 全局限速，避免 OpenAlex/Crossref 429
BACKOFF = [5, 15, 45, 90]

def _save_cache():
    json.dump(_cache, open(CACHE, 'w'))

def get(u, tries=4):
    """带磁盘缓存与指数退避的取回；429 必须退避而不是当成空结果。"""
    if u in _cache:
        return _cache[u]
    for k in range(tries):
        wait = MIN_INTERVAL - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=90) as r:
                t = r.read().decode('utf-8', 'replace')
            _cache[u] = t
            _save_cache()
            return t
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and k < tries - 1:
                print('   [backoff %ds] %d %s' % (BACKOFF[k], e.code, u[:70]), flush=True)
                time.sleep(BACKOFF[k])
                continue
            if k == tries - 1:
                print('   [http-fail] %s %s' % (e, u[:70]), flush=True)
                return ''
        except Exception as e:
            if k == tries - 1:
                print('   [err] %r %s' % (e, u[:70]), flush=True)
                return ''
            time.sleep(BACKOFF[k])
    return ''

def inv(d):
    if not d:
        return ''
    pos = {}
    for w, ps in d.items():
        for p in ps:
            pos[p] = w
    return ' '.join(pos[k] for k in sorted(pos))

def norm_words(t):
    return set(re.sub(r'[^a-z0-9 ]', ' ', (t or '').lower()).split())

def sim(a, b):
    na, nb = norm_words(a), norm_words(b)
    return len(na & nb) / max(1, len(na | nb))

def openalex_candidates(q, n=25):
    u = ('https://api.openalex.org/works?search=' + urllib.parse.quote(q) +
         f'&per-page={n}&sort=cited_by_count:desc&filter=from_publication_date:2015-01-01,has_doi:true')
    t = get(u)
    if not t:
        return []
    try:
        rs = json.loads(t).get('results', [])
    except Exception:
        return []
    out = []
    for w in rs:
        ab = inv(w.get('abstract_inverted_index'))
        doi = (w.get('doi') or '').replace('https://doi.org/', '')
        if not doi or len(ab) < 400:
            continue
        out.append(dict(title=w.get('title'), year=w.get('publication_year'), doi=doi,
                        cited=w.get('cited_by_count'),
                        venue=((w.get('primary_location') or {}).get('source') or {}).get('display_name'),
                        abstract=ab))
    return out

def crossref_by_doi(doi):
    t = get('https://api.crossref.org/works/' + urllib.parse.quote(doi))
    if not t:
        return None
    try:
        m = json.loads(t)['message']
    except Exception:
        return None
    return dict(title=(m.get('title') or [''])[0], doi=m.get('DOI'),
                year=(m.get('issued', {}).get('date-parts', [[None]])[0][0]),
                venue=(m.get('container-title') or [None])[0],
                cited=m.get('is-referenced-by-count'), type=m.get('type'))

def arxiv_by_title(t):
    d = get('http://export.arxiv.org/api/query?search_query=' +
            urllib.parse.quote('ti:"' + (t or '')[:80] + '"') + '&max_results=1')
    m = re.search(r'<entry>(.*?)</entry>', d or '', re.S)
    if not m:
        return None
    e = m.group(1)
    ti = re.search(r'<title>(.*?)</title>', e, re.S)
    idm = re.search(r'<id>(.*?)</id>', e, re.S)
    name = ti.group(1).strip().replace('\n', ' ') if ti else ''
    if sim(t, name) < 0.75:
        return None
    return dict(title=name, url=(idm.group(1).strip() if idm else ''))

PER_GROUP = 4
recs = []
seen_doi = set()
for gid, qs in QUERIES.items():
    kept = 0
    for q in qs:
        if kept >= PER_GROUP:
            break
        for c in openalex_candidates(q):
            if kept >= PER_GROUP:
                break
            if c['doi'] in seen_doi:
                continue
            cr = crossref_by_doi(c['doi'])
            if not cr or not cr.get('title'):
                continue
            s = sim(c['title'], cr['title'])
            if s < 0.75:
                continue
            ax = arxiv_by_title(c['title'])
            srcs = ['openalex', 'crossref'] + (['arxiv'] if ax else [])
            seen_doi.add(c['doi'])
            kept += 1
            recs.append(dict(group=gid, query=q, doi=c['doi'], title=c['title'],
                             crossref_title=cr['title'], title_sim=round(s, 3),
                             year=c['year'], venue=c['venue'] or cr.get('venue'),
                             cited=c['cited'], sources=srcs, n_sources=len(srcs),
                             arxiv_url=(ax or {}).get('url'),
                             abstract=c['abstract'], abstract_len=len(c['abstract'])))
            print('%s | sim=%.2f | %-22s | c=%5s | %s' %
                  (gid, s, ','.join(srcs), c['cited'], (c['title'] or '')[:78]), flush=True)

json.dump(recs, open(f'{OUT}/readlist_doi.json', 'w'), indent=1)
ok = [r for r in recs if r['n_sources'] >= 2 and r['abstract_len'] > 400 and r['title_sim'] >= 0.75]
print('TOTAL', len(recs), 'VERIFIED_MULTISOURCE', len(ok), flush=True)