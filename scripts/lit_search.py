"""多通道文献检索与原文核验工具。

检索通道：
  - arxiv    : arXiv Atom API (export.arxiv.org)
  - crossref : Crossref REST API (api.crossref.org)
  - openalex : OpenAlex works API (api.openalex.org)
  - s2       : Semantic Scholar Graph API (api.semanticscholar.org)

核验通道 --verify：对每条命中记录，直接抓取 arXiv abs 页面或 DOI landing page，
把 HTTP 状态码、最终 URL、页面标题以及页面正文命中标题 token 的比例写入结果，
作为原文核验证据。这是浏览器 MCP 不可用时的等价可核验替代，必须在论文记录中
如实标注实际使用的通道，不得声称使用了未实际调用的工具。

本脚本只做网络读取与 JSON 落盘，不接触任何比赛数据，不参与训练。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

USER_AGENT = "dizheng-lit-search/1.0 (academic literature review)"
TIMEOUT = 45

def _get(url, headers=None, retries=3):
    """HTTP GET，返回 (status, body, final_url)；对限流与 5xx 做指数退避重试。"""
    hdrs = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        hdrs.update(headers)
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return int(resp.status), resp.read(), str(resp.geturl())
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read()
            except Exception:
                pass
            if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                last_err = exc
                time.sleep(2.0 * (attempt + 1))
                continue
            return int(exc.code), body, url
        except Exception as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
    raise RuntimeError("GET failed after %d attempts: %s: %s" % (retries, url, last_err))


@dataclass
class Record:
    channel: str
    title: str = ""
    authors: list = field(default_factory=list)
    year: Any = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    abstract: str = ""
    citations: Any = None
    verification: Any = None


def _clean(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()

def search_arxiv(query, max_results):
    params = urllib.parse.urlencode(
        {
            "search_query": "all:%s" % query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    status, body, _ = _get("http://export.arxiv.org/api/query?%s" % params)
    if status != 200:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ElementTree.fromstring(body)
    out = []
    for entry in root.findall("a:entry", ns):
        raw_id = _clean(entry.findtext("a:id", default="", namespaces=ns))
        arxiv_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
        published = _clean(entry.findtext("a:published", default="", namespaces=ns))
        year = int(published[:4]) if published[:4].isdigit() else None
        doi_el = entry.find("arxiv:doi", ns)
        authors = []
        for a in entry.findall("a:author", ns):
            authors.append(_clean(a.findtext("a:name", default="", namespaces=ns)))
        out.append(
            Record(
                channel="arxiv",
                title=_clean(entry.findtext("a:title", default="", namespaces=ns)),
                authors=authors,
                year=year,
                venue=_clean(entry.findtext("arxiv:journal_ref", default="", namespaces=ns)),
                doi=_clean(doi_el.text) if doi_el is not None else "",
                arxiv_id=arxiv_id,
                url=raw_id,
                abstract=_clean(entry.findtext("a:summary", default="", namespaces=ns)),
            )
        )
    return out

def search_crossref(query, max_results):
    sel = "DOI,title,author,issued,container-title,abstract,is-referenced-by-count,URL"
    params = urllib.parse.urlencode(
        {"query.bibliographic": query, "rows": max_results, "select": sel}
    )
    status, body, _ = _get("https://api.crossref.org/works?%s" % params)
    if status != 200:
        return []
    payload = json.loads(body.decode("utf-8", "replace"))
    out = []
    for item in payload.get("message", {}).get("items", []):
        titles = item.get("title") or []
        issued = (item.get("issued") or {}).get("date-parts") or [[None]]
        year = issued[0][0] if issued and issued[0] else None
        authors = []
        for a in item.get("author") or []:
            parts = [x for x in [a.get("given"), a.get("family")] if x]
            if parts:
                authors.append(_clean(" ".join(parts)))
        containers = item.get("container-title") or []
        abstract = re.sub(r"<[^>]+>", " ", item.get("abstract") or "")
        out.append(
            Record(
                channel="crossref",
                title=_clean(titles[0] if titles else ""),
                authors=authors,
                year=int(year) if isinstance(year, int) else None,
                venue=_clean(containers[0] if containers else ""),
                doi=_clean(item.get("DOI")),
                url=_clean(item.get("URL")),
                abstract=_clean(abstract),
                citations=item.get("is-referenced-by-count"),
            )
        )
    return out

def search_openalex(query, max_results):
    params = urllib.parse.urlencode({"search": query, "per-page": min(max_results, 50)})
    status, body, _ = _get("https://api.openalex.org/works?%s" % params)
    if status != 200:
        return []
    payload = json.loads(body.decode("utf-8", "replace"))
    out = []
    for item in payload.get("results", []):
        ids = item.get("ids") or {}
        doi = _clean(ids.get("doi")).replace("https://doi.org/", "")
        loc = (item.get("primary_location") or {}).get("source") or {}
        abstract = item.get("abstract") or ""
        if not abstract and item.get("abstract_inverted_index"):
            positions = []
            for word, idxs in item["abstract_inverted_index"].items():
                for i in idxs:
                    positions.append((i, word))
            abstract = " ".join(w for _, w in sorted(positions))
        authors = []
        for a in item.get("authorships") or []:
            authors.append(_clean((a.get("author") or {}).get("display_name")))
        out.append(
            Record(
                channel="openalex",
                title=_clean(item.get("title") or item.get("display_name")),
                authors=authors,
                year=item.get("publication_year"),
                venue=_clean(loc.get("display_name")),
                doi=doi,
                url=_clean(ids.get("doi") or item.get("id")),
                abstract=_clean(abstract),
                citations=item.get("cited_by_count"),
            )
        )
    return out


def search_s2(query, max_results):
    fields = "title,abstract,year,venue,authors,externalIds,citationCount,url"
    params = urllib.parse.urlencode(
        {"query": query, "limit": min(max_results, 100), "fields": fields}
    )
    url = "https://api.semanticscholar.org/graph/v1/paper/search?%s" % params
    try:
        status, body, _ = _get(url)
    except RuntimeError:
        return []
    if status != 200:
        return []
    payload = json.loads(body.decode("utf-8", "replace"))
    out = []
    for item in payload.get("data", []):
        ext = item.get("externalIds") or {}
        authors = []
        for a in item.get("authors") or []:
            authors.append(_clean(a.get("name")))
        out.append(
            Record(
                channel="s2",
                title=_clean(item.get("title")),
                authors=authors,
                year=item.get("year"),
                venue=_clean(item.get("venue")),
                doi=_clean(ext.get("DOI")),
                arxiv_id=_clean(ext.get("ArXiv")),
                url=_clean(item.get("url")),
                abstract=_clean(item.get("abstract")),
                citations=item.get("citationCount"),
            )
        )
    return out


SEARCHERS = {
    "arxiv": search_arxiv,
    "crossref": search_crossref,
    "openalex": search_openalex,
    "s2": search_s2,
}

def _title_tokens(title):
    return set(t for t in re.findall(r"[a-z0-9]+", title.lower()) if len(t) > 3)


def verify_record(rec):
    """抓取 arXiv abs 页或 DOI landing page，核对标题是否真实存在。"""
    target = ""
    if rec.arxiv_id:
        target = "https://arxiv.org/abs/%s" % rec.arxiv_id
    elif rec.doi:
        target = "https://doi.org/%s" % rec.doi
    elif rec.url:
        target = rec.url
    if not target:
        return {"ok": False, "reason": "no-resolvable-identifier"}
    hdr = {"Accept": "text/html,application/xhtml+xml"}
    try:
        status, body, final_url = _get(target, headers=hdr, retries=2)
    except RuntimeError as exc:
        return {"ok": False, "target": target, "reason": "fetch-failed: %s" % exc}
    text = body.decode("utf-8", "replace")
    m = re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
    page_title = _clean(re.sub(r"<[^>]+>", " ", m.group(1))) if m else ""
    plain = _clean(re.sub(r"<[^>]+>", " ", text)).lower()
    want = _title_tokens(rec.title)
    hit = sum(1 for t in want if t in plain)
    ratio = (float(hit) / len(want)) if want else 0.0
    return {
        "ok": bool(status == 200 and ratio >= 0.6),
        "target": target,
        "final_url": final_url,
        "http_status": status,
        "page_title": page_title[:300],
        "title_token_match_ratio": round(ratio, 4),
        "bytes": len(body),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def dedupe(records):
    seen = {}
    for rec in records:
        if rec.doi:
            key = rec.doi.lower()
        elif rec.arxiv_id:
            key = "arxiv:%s" % rec.arxiv_id
        else:
            key = rec.title.lower()[:120]
        if not key:
            continue
        cur = seen.get(key)
        if cur is None:
            seen[key] = rec
            continue
        if len(rec.abstract) > len(cur.abstract):
            cur.abstract = rec.abstract
        if not cur.doi and rec.doi:
            cur.doi = rec.doi
        if not cur.arxiv_id and rec.arxiv_id:
            cur.arxiv_id = rec.arxiv_id
        if cur.citations is None and rec.citations is not None:
            cur.citations = rec.citations
        if not cur.venue and rec.venue:
            cur.venue = rec.venue
        if cur.year is None and rec.year is not None:
            cur.year = rec.year
        if rec.channel not in cur.channel:
            cur.channel = "%s+%s" % (cur.channel, rec.channel)
    return list(seen.values())

def main(argv=None):
    ap = argparse.ArgumentParser(description="多通道文献检索与原文核验")
    ap.add_argument("--query", action="append", required=True, help="检索式，可重复")
    ap.add_argument("--channels", default="arxiv,crossref,openalex,s2")
    ap.add_argument("--max-results", type=int, default=15)
    ap.add_argument("--verify", action="store_true", help="抓取 abs/DOI 页做原文核验")
    ap.add_argument("--verify-limit", type=int, default=40)
    ap.add_argument("--min-year", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    bad = [c for c in channels if c not in SEARCHERS]
    if bad:
        ap.error("unknown channels: %s; allowed=%s" % (bad, list(SEARCHERS)))

    all_records = []
    log = []
    for query in args.query:
        for ch in channels:
            t0 = time.time()
            try:
                got = SEARCHERS[ch](query, args.max_results)
                err = ""
            except Exception as exc:
                got, err = [], str(exc)
            log.append(
                {
                    "query": query,
                    "channel": ch,
                    "n_results": len(got),
                    "error": err,
                    "elapsed_s": round(time.time() - t0, 2),
                    "queried_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            tail = (" ERR:" + err) if err else ""
            sys.stderr.write("[%s] %s -> %d%s\n" % (ch, query, len(got), tail))
            all_records.extend(got)
            time.sleep(1.0)

    merged = dedupe(all_records)
    if args.min_year:
        merged = [r for r in merged if (r.year or 0) >= args.min_year]
    merged.sort(key=lambda r: (-(r.citations or 0), -(r.year or 0)))

    if args.verify:
        for rec in merged[: args.verify_limit]:
            rec.verification = verify_record(rec)
            state = "OK" if (rec.verification or {}).get("ok") else "FAIL"
            sys.stderr.write("  verify[%s] %s\n" % (state, rec.title[:80]))
            time.sleep(0.6)

    n_ok = sum(1 for r in merged if (r.verification or {}).get("ok"))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "queries": args.query,
        "channels": channels,
        "search_log": log,
        "n_unique": len(merged),
        "n_verified_ok": n_ok,
        "records": [asdict(r) for r in merged],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stderr.write("wrote %s unique=%d verified_ok=%d\n" % (args.out, len(merged), n_ok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

def verify_doi_metadata(doi, want_title):
    """Crossref /works/{doi} 权威元数据核验（用于出版商页面有 JS/Cloudflare 墙时）。

    返回字段包含 Crossref 记录的标题、年份、期刊、作者数，以及与期望标题的
    token 命中率。这是书目级核验，不等于全文核验，必须在论文记录中分别标注。
    """
    if not doi:
        return {"ok": False, "reason": "no-doi"}
    url = "https://api.crossref.org/works/%s" % urllib.parse.quote(doi, safe="")
    try:
        status, body, final_url = _get(url, retries=3)
    except RuntimeError as exc:
        return {"ok": False, "target": url, "reason": "fetch-failed: %s" % exc}
    if status != 200:
        return {"ok": False, "target": url, "http_status": status}
    try:
        msg = json.loads(body.decode("utf-8", "replace"))["message"]
    except Exception as exc:
        return {"ok": False, "target": url, "reason": "parse-failed: %s" % exc}
    titles = msg.get("title") or []
    got = _clean(titles[0] if titles else "")
    issued = (msg.get("issued") or {}).get("date-parts") or [[None]]
    year = issued[0][0] if issued and issued[0] else None
    containers = msg.get("container-title") or []
    want = _title_tokens(want_title)
    have = _title_tokens(got)
    ratio = (len(want & have) / len(want)) if want else 0.0
    return {
        "ok": bool(ratio >= 0.6),
        "channel": "crossref-metadata",
        "target": url,
        "final_url": final_url,
        "http_status": status,
        "crossref_title": got[:300],
        "crossref_year": year,
        "crossref_container": _clean(containers[0] if containers else "")[:160],
        "crossref_n_authors": len(msg.get("author") or []),
        "crossref_cited_by": msg.get("is-referenced-by-count"),
        "title_token_match_ratio": round(ratio, 4),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }