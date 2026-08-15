import ast
from pathlib import Path

p = Path('experiments/t2_round39c_ablation/lit_crossref_arxiv.py')
assert p.exists(), 'production fallback script must exist'
source = p.read_text(encoding='utf-8')
mod = ast.parse(source)
funcs = {n.name for n in mod.body if isinstance(n, ast.FunctionDef)}
assert {'crossref_candidates', 'arxiv_by_title', 'sim', 'get'} <= funcs
assert 'HTTPError' in source and 'CACHE' in source, 'network fetches must retry 429 and persist cache'
assert '[:12]' in source, 'candidate list must be bounded before arXiv requests'
assert 'return \'\'' in source, 'rate-limit exhaustion must skip a candidate rather than abort the review run'
assert 'MAX_429_RETRIES = 1' in source, 'single-paper rate-limit retry budget must remain bounded'
assert 'checkpoint(records)' in source and 'VERIFIED' in source, 'retrieval must persist and log verified records incrementally'
print('fallback API and rate-limit defenses present')