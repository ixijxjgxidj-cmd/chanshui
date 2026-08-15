"""第39C/D文献双源检索：Crossref发现 + arXiv题名/摘要核验。"""
import json, os, re, time, urllib.error, urllib.parse, urllib.request

OUT = '/root/5.6+chanshui1/outputs/t2_round39c_literature_crossref_arxiv'
UA = {'User-Agent': 'dizheng-research/1.0 (mailto:research@example.org)'}
MIN_INTERVAL = 3.5
CACHE = f'{OUT}/http_cache.json'
_cache = {}
_last = [0.0]
QUERIES = {
    'N1': ['masked autoencoders scalable learners', 'masked time series modeling'],
    'N2': ['self supervised pretraining transfer learning seismic'],
    'N3': ['domain generalization distribution shift time series'],
    'N4': ['calibration regression neural networks conformal prediction'],
    'N5': ['knowledge distillation deep ensembles out of distribution'],
    'N6': ['earthquake magnitude estimation deep learning seismic'],
}

def get(url):
    if url in _cache:
        return _cache[url]
    for attempt, pause in enumerate((15, 45, 90)):
        wait = MIN_INTERVAL - (time.time() - _last[0])
        if wait > 0: time.sleep(wait)
        _last[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                text = r.read().decode('utf-8', 'replace')
            _cache[url] = text
            os.makedirs(OUT, exist_ok=True)
            with open(CACHE, 'w', encoding='utf-8') as f: json.dump(_cache, f)
            return text
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 2: raise
            time.sleep(pause)

def norm_words(text):
    return set(re.sub(r'[^a-z0-9 ]', ' ', (text or '').lower()).split())

def sim(left, right):
    a, b = norm_words(left), norm_words(right)
    return len(a & b) / max(1, len(a | b))

def crossref_candidates(query, rows=30):
    url = ('https://api.crossref.org/works?query.bibliographic=' + urllib.parse.quote(query) +
           '&rows=%d&select=DOI,title,published,container-title,is-referenced-by-count,type' % rows)
    data = json.loads(get(url)).get('message', {}).get('items', [])
    out = []
    for item in data:
        title, doi = (item.get('title') or [''])[0].strip(), item.get('DOI', '').strip()
        if title and doi:
            out.append({'title': title, 'doi': doi,
                        'year': item.get('published', {}).get('date-parts', [[None]])[0][0],
                        'venue': (item.get('container-title') or [None])[0],
                        'cited': item.get('is-referenced-by-count'), 'type': item.get('type')})
    return out

def arxiv_by_title(title):
    query = 'http://export.arxiv.org/api/query?search_query=' + urllib.parse.quote('ti:"' + title[:120] + '"') + '&max_results=3'
    xml = get(query)
    candidates = []
    for entry in re.findall(r'<entry>(.*?)</entry>', xml, re.S):
        raw_title, summary, ident = (re.search(r'<title>(.*?)</title>', entry, re.S),
                                     re.search(r'<summary>(.*?)</summary>', entry, re.S),
                                     re.search(r'<id>(.*?)</id>', entry, re.S))
        if not raw_title or not summary: continue
        atitle = re.sub(r'\s+', ' ', raw_title.group(1)).strip()
        abstract = re.sub(r'\s+', ' ', summary.group(1)).strip()
        candidates.append((sim(title, atitle), {'title': atitle, 'abstract': abstract,
                          'url': ident.group(1).strip() if ident else ''}))
    if not candidates: return None
    score, record = max(candidates, key=lambda x: x[0])
    return record | {'title_sim': round(score, 3)} if score >= .75 else None

def build_records():
    records, seen = [], set()
    for group, queries in QUERIES.items():
        kept = 0
        for query in queries:
            for cr in crossref_candidates(query)[:12]:
                if kept >= 4: break
                if cr['doi'].lower() in seen: continue
                ax = arxiv_by_title(cr['title'])
                if not ax or len(ax['abstract']) < 400: continue
                seen.add(cr['doi'].lower()); kept += 1
                records.append(cr | {'group': group, 'query': query, 'crossref_title': cr['title'],
                    'arxiv_title': ax['title'], 'arxiv_url': ax['url'], 'title_sim': ax['title_sim'],
                    'abstract': ax['abstract'], 'abstract_len': len(ax['abstract']),
                    'sources': ['crossref', 'arxiv'], 'n_sources': 2})
            if kept >= 4: break
    return records

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    records = build_records()
    with open(f'{OUT}/readlist_crossref_arxiv.json', 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print('TOTAL', len(records), 'VERIFIED_MULTISOURCE', len(records), flush=True)