"""第39轮文献检索：三源（OpenAlex + Crossref + arXiv）交叉核验。

问题清单（由第38/39A轮实验直接产生，检索必须服务这些问题）：
  Q1 波形/时序 JEPA 与掩码自监督在小标签量下究竟何时优于监督基线？
  Q2 地震震级估计的跨区域/域偏移校正与站点项分离
  Q3 排名收缩尺度 lam 一类的"批内相对量 -> 绝对量"标定理论
  Q4 自监督预训练池的域组成（同域 vs 多域）对域外泛化的影响
  Q5 地震基础模型/预训练（SeisLM、SeisBench、Transformer 拾取器）迁移证据
  Q6 蒸馏与集成在分布偏移下是否仍然有效
"""
import json, time, urllib.parse, urllib.request, os, re

OUT = '/root/5.6+chanshui1/outputs/round39_lit'
os.makedirs(OUT, exist_ok=True)
UA = {'User-Agent': 'dizheng-research/1.0 (mailto:research@example.org)'}

QUERIES = {
 'Q1': ['joint embedding predictive architecture time series',
        'masked autoencoder time series representation learning',
        'self-supervised pretraining limited labels regression seismic'],
 'Q2': ['earthquake magnitude estimation deep learning single station',
        'transfer learning earthquake magnitude cross region',
        'site effect station term magnitude machine learning'],
 'Q3': ['relative ranking calibration to absolute scale regression',
        'conformal calibration distribution shift regression',
        'shrinkage estimator noisy predictions calibration'],
 'Q4': ['domain generalization pretraining data diversity out-of-distribution',
        'multi-domain self-supervised pretraining transfer'],
 'Q5': ['seismic foundation model pretraining waveform',
        'SeisBench benchmark deep learning seismology',
        'transformer earthquake phase picking generalization'],
 'Q6': ['knowledge distillation robustness distribution shift',
        'deep ensembles distribution shift uncertainty'],
}

def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception as e:
            if k == tries - 1:
                return ''
            time.sleep(3 + 3 * k)
    return ''

def openalex(q, n=25):
    u = ('https://api.openalex.org/works?search=' + urllib.parse.quote(q) +
         f'&per-page={n}&filter=from_publication_date:2018-01-01')
    t = get(u)
    if not t:
        return []
    try:
        d = json.loads(t)
    except Exception:
        return []
    out = []
    for w in d.get('results', []):
        out.append(dict(src='openalex', title=(w.get('title') or ''),
                        year=w.get('publication_year'), doi=w.get('doi'),
                        cited=w.get('cited_by_count', 0),
                        venue=((w.get('primary_location') or {}).get('source') or {}).get('display_name'),
                        oa=(w.get('open_access') or {}).get('oa_url'),
                        query=q))
    return out

def crossref(q, n=20):
    u = ('https://api.crossref.org/works?query=' + urllib.parse.quote(q) +
         f'&rows={n}&filter=from-pub-date:2018-01-01,type:journal-article&sort=score')
    t = get(u)
    if not t:
        return []
    try:
        d = json.loads(t)
    except Exception:
        return []
    out = []
    for w in d.get('message', {}).get('items', []):
        out.append(dict(src='crossref', title=(w.get('title') or [''])[0],
                        year=(w.get('issued', {}).get('date-parts', [[None]])[0][0]),
                        doi=w.get('DOI'), cited=w.get('is-referenced-by-count', 0),
                        venue=(w.get('container-title') or [None])[0], oa=None, query=q))
    return out

def arxiv(q, n=20):
    u = ('http://export.arxiv.org/api/query?search_query=' +
         urllib.parse.quote('all:"' + q + '"') + f'&max_results={n}&sortBy=relevance')
    t = get(u)
    if not t:
        return []
    out = []
    for m in re.finditer(r'<entry>(.*?)</entry>', t, re.S):
        e = m.group(1)
        ti = re.search(r'<title>(.*?)</title>', e, re.S)
        idm = re.search(r'<id>(.*?)</id>', e, re.S)
        pub = re.search(r'<published>(\d{4})', e)
        out.append(dict(src='arxiv', title=(ti.group(1).strip().replace('\n', ' ') if ti else ''),
                        year=int(pub.group(1)) if pub else None,
                        doi=None, cited=None, venue='arXiv',
                        oa=(idm.group(1).strip() if idm else None), query=q))
    return out

all_hits = []
for qid, qs in QUERIES.items():
    for q in qs:
        for fn in (openalex, crossref, arxiv):
            hs = fn(q)
            for h in hs:
                h['qid'] = qid
            all_hits += hs
            print(qid, fn.__name__, q[:48], len(hs), flush=True)
            time.sleep(1.2)

def norm(t):
    return re.sub(r'[^a-z0-9]+', ' ', (t or '').lower()).strip()

by_title = {}
for h in all_hits:
    k = norm(h['title'])
    if not k or len(k) < 12:
        continue
    if k in by_title:
        by_title[k]['srcs'] = sorted(set(by_title[k]['srcs'] + [h['src']]))
        by_title[k]['cited'] = max(by_title[k].get('cited') or 0, h.get('cited') or 0)
        by_title[k]['qids'] = sorted(set(by_title[k]['qids'] + [h['qid']]))
        by_title[k]['oa'] = by_title[k].get('oa') or h.get('oa')
        by_title[k]['doi'] = by_title[k].get('doi') or h.get('doi')
    else:
        by_title[k] = dict(title=h['title'], year=h['year'], doi=h['doi'], cited=h.get('cited') or 0,
                           venue=h.get('venue'), oa=h.get('oa'), srcs=[h['src']], qids=[h['qid']])

uniq = list(by_title.values())
multi = [u for u in uniq if len(u['srcs']) >= 2]
uniq.sort(key=lambda u: (-(len(u['srcs'])), -(u['cited'] or 0)))
json.dump(dict(n_raw=len(all_hits), n_unique=len(uniq), n_multisource=len(multi),
               queries=QUERIES, hits=uniq), open(f'{OUT}/search_raw.json', 'w'), indent=1)
print('RAW', len(all_hits), 'UNIQUE', len(uniq), 'MULTI_SOURCE', len(multi), flush=True)
for u in uniq[:60]:
    print('%d src=%s cited=%s %s | %s' % (u['year'] or 0, ','.join(u['srcs']), u['cited'],
                                          (u['title'] or '')[:110], ','.join(u['qids'])), flush=True)