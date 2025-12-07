#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, json, re, os, time
from datetime import datetime
from dateutil import parser as dtparser
import pytz, requests
from mcp.server.fastmcp import FastMCP

# ---------- stderr-only logs (never print to stdout) ----------
print(">>> football_fastmcp starting", file=sys.stderr, flush=True)

mcp = FastMCP("mcp-football")

# Map friendly league names to ESPN paths
LEAGUE_MAP = {
    "epl": "soccer/eng.1",          # English Premier League
    "ucl": "soccer/uefa.champions", # UEFA Champions League
    "laliga": "soccer/esp.1",
    "seriea": "soccer/ita.1",
    "bundesliga": "soccer/ger.1",
    "ligue1": "soccer/fra.1",
}

UA = {"User-Agent": "mcp-football/1.0 (+local)"}
DEBUG = os.environ.get("FOOTBALL_DEBUG", "0") == "1"

def dlog(msg: str):
    if DEBUG:
        print(f"[debug] {msg}", file=sys.stderr, flush=True)

def _today_london_iso() -> str:
    now_ldn = datetime.now(pytz.timezone("Europe/London"))
    return now_ldn.strftime("%Y-%m-%d")

# ---------------------- ESPN JSON (primary) ----------------------
def json_endpoint_for(league_key: str) -> str:
    league_path = LEAGUE_MAP.get(league_key.lower())
    if not league_path:
        raise ValueError(f"Unknown league '{league_key}'. Try one of: {', '.join(LEAGUE_MAP.keys())}")
    return f"https://site.api.espn.com/apis/v2/sports/{league_path}/scoreboard"

def fetch_results_json(league_key: str, date_iso: str):
    url = json_endpoint_for(league_key)
    yyyymmdd = date_iso.replace("-", "")

    # cache-bust param "_" avoids some ESPN edge caches
    resp = requests.get(
        url,
        params={"dates": yyyymmdd, "limit": 300, "_": int(time.time())},
        timeout=15,
        headers=UA,
    )
    status = resp.status_code
    dlog(f"JSON GET {url}?dates={yyyymmdd} -> {status}")
    if status == 404:
        return [], {"status": status, "events": 0}
    resp.raise_for_status()

    data = resp.json()
    events = data.get("events", []) or []
    results = []
    for ev in events:
        comps = ev.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        status = comp.get("status", {}).get("type", {}).get("shortDetail") or ev.get("status", {}).get("type", {}).get("shortDetail")
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        home = next((t for t in competitors if t.get("homeAway") == "home"), competitors[0])
        away = next((t for t in competitors if t.get("homeAway") == "away"), competitors[-1])

        def team_line(c):
            team = c.get("team", {}) or {}
            name = team.get("shortDisplayName") or team.get("displayName") or team.get("name")
            score = c.get("score")
            return name, score

        home_name, home_score = team_line(home)
        away_name, away_score = team_line(away)

        kickoff = None
        start = ev.get("date")
        if start:
            try:
                dt = dtparser.isoparse(start).astimezone(pytz.timezone("Europe/London"))
                kickoff = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                kickoff = start

        results.append({
            "kickoff_local": kickoff,
            "status": status,
            "home": home_name,
            "away": away_name,
            "home_score": home_score,
            "away_score": away_score,
        })
    return results, {"status": resp.status_code, "events": len(events)}

# ---------------------- ESPN schedule HTML (fallback) ----------------------
def schedule_url_for(league_key: str, date_iso: str) -> str:
    league_path = LEAGUE_MAP.get(league_key.lower())
    if not league_path:
        raise ValueError(f"Unknown league '{league_key}'. Try one of: {', '.join(LEAGUE_MAP.keys())}")
    yyyymmdd = date_iso.replace("-", "")
    league_code = league_path.split("/")[-1]  # e.g., "eng.1"
    return f"https://www.espn.com/soccer/schedule/_/league/{league_code}/date/{yyyymmdd}"

def try_bs4_parse(html: str):
    """
    Parse ESPN schedule HTML rows robustly:
    - Extract exactly two team names (home, away) in row order
    - Extract score if present (e.g., '2-1', '2 – 1')
    - Detect FT / Finished status
    """
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return None  # bs4 not installed

    soup = BeautifulSoup(html, "lxml")
    results = []

    # Helper: clean team name
    def clean(txt: str) -> str:
        txt = re.sub(r"\s+", " ", txt).strip()
        txt = txt.replace("\xa0", " ").replace("&nbsp;", " ").replace("&amp;", "&")
        return txt

    # Helper: extract (home, away) names in row order
    def get_names(tr) -> list[str]:
        # Prefer anchor tags that look like team/club links
        anchors = []
        for a in tr.find_all("a"):
            txt = a.get_text(" ", strip=True)
            if not txt:
                continue
            if txt.lower() in ("tickets", "details"):
                continue
            href = a.get("href", "")
            if "/team/" in href or "/club/" in href or "/_club" in href or "/_team" in href:
                anchors.append(clean(txt))

        # De-dup while preserving order
        seen = set()
        ordered = []
        for name in anchors:
            if name and name not in seen:
                ordered.append(name)
                seen.add(name)

        # If we didn't get two clean names from anchors, fall back to the first TD text
        if len(ordered) < 2:
            # ESPN often has a single cell that contains "Home vs Away"
            # Find the most text-dense cell
            tds = tr.find_all("td")
            if tds:
                dense = max(tds, key=lambda td: len(td.get_text("", strip=True)))
                block = clean(dense.get_text(" ", strip=True))
                # Split on vs or en dash/hyphen between words
                m = re.split(r"\s+vs\.?\s+|\s+v\s+|\s+\|\s+", block, maxsplit=1, flags=re.IGNORECASE)
                if len(m) == 2:
                    ordered = [clean(m[0]), clean(m[1])]
                else:
                    # final fallback: try to split around the score
                    score_rx = re.compile(r"\b\d+\s*[-–]\s*\d+\b")
                    parts = score_rx.split(block)
                    if len(parts) == 2:
                        left, right = clean(parts[0]), clean(parts[1])
                        # heuristics to keep the last/first few words
                        left = " ".join(left.split()[-4:]) or left
                        right = " ".join(right.split()[:4]) or right
                        ordered = [left, right]

        # Trim to exactly two
        if len(ordered) >= 2:
            return ordered[:2]
        return []

    # Walk table rows
    for tr in soup.select("table tbody tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        row_text = " ".join(td.get_text(" ", strip=True) for td in tds)
        if not row_text:
            continue

        # Find score and status
        score_m = re.search(r"\b(\d+)\s*[-–]\s*(\d+)\b", row_text)
        is_ft = bool(re.search(r"\bFT\b|\bFull\s*Time\b|\bFinished\b", row_text, re.IGNORECASE))
        is_vs = bool(re.search(r"\bvs\.?\b|\bv\b", row_text, re.IGNORECASE))

        names = get_names(tr)

        # Finished/result rows
        if score_m and (is_ft or not is_vs):
            if len(names) < 2:
                # As a last resort, try to split by score inside the largest cell
                dense = max(tds, key=lambda td: len(td.get_text("", strip=True)))
                block = clean(dense.get_text(" ", strip=True))
                parts = re.split(r"\b\d+\s*[-–]\s*\d+\b", block, maxsplit=1)
                if len(parts) == 2:
                    left = " ".join(parts[0].split()[-4:])
                    right = " ".join(parts[1].split()[:4])
                    names = [clean(left), clean(right)]
            if len(names) >= 2:
                results.append({
                    "kickoff_local": None,
                    "status": "FT" if is_ft else "Result",
                    "home": names[0],
                    "away": names[1],
                    "home_score": score_m.group(1),
                    "away_score": score_m.group(2),
                })
            continue

        # Scheduled fixtures (no score yet)
        if is_vs and not score_m and len(names) >= 2:
            results.append({
                "kickoff_local": None,
                "status": "Scheduled",
                "home": names[0],
                "away": names[1],
                "home_score": None,
                "away_score": None,
            })

    return results


def fetch_results_html_fallback(league_key: str, date_iso: str):
    url = schedule_url_for(league_key, date_iso)
    try:
        resp = requests.get(url, timeout=15, headers=UA)
        status = resp.status_code
        html = resp.text
    except Exception as e:
        print(f"[fallback] schedule fetch failed: {e}", file=sys.stderr, flush=True)
        return [], {"status": None, "len": 0}

    dlog(f"HTML GET {url} -> {status}, len={len(html)}")

    results = try_bs4_parse(html)
    if results is None:  # bs4 missing — Regex-only fallback
        text = re.sub(r"\s+", " ", html)
        pattern = re.compile(
            r">([^<]{2,80}?)<[^>]*?>\s*(\d+)\s*[-–]\s*(\d+)\s*<[^>]*?>\s*([^<]{2,80}?)<[^>]*?>\s*F(?:T|ull Time)?",
            re.IGNORECASE
        )
        results = []
        for m in pattern.finditer(text):
            home_name = re.sub(r"&nbsp;", " ", re.sub(r"&amp;", "&", m.group(1).strip()))
            home_score = m.group(2).strip()
            away_score = m.group(3).strip()
            away_name = re.sub(r"&nbsp;", " ", re.sub(r"&amp;", "&", m.group(4).strip()))
            results.append({
                "kickoff_local": None,
                "status": "FT",
                "home": home_name,
                "away": away_name,
                "home_score": home_score,
                "away_score": away_score,
            })
    return results, {"status": status, "len": len(html)}

# ---------------------- Combined fetch ----------------------
def fetch_results(league_key: str, date_iso: str):
    # 1) JSON first
    try:
        primary, meta = fetch_results_json(league_key, date_iso)
        if primary:
            return primary, {"source": "json", **meta}
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        dlog(f"JSON HTTPError {code}: {e}")
        if code != 404:
            # Only fall back on 404/empty; otherwise re-raise
            pass
    except Exception as e:
        dlog(f"JSON fetch error: {e}")

    # 2) HTML fallback
    fallback, meta2 = fetch_results_html_fallback(league_key, date_iso)
    return fallback, {"source": "html", **meta2}

# ----------------------------- Tools -----------------------------
@mcp.tool()
def ping() -> str:
    """Simple health check."""
    return "pong"

@mcp.tool()
def debug_sources(league: str = "epl", date: str | None = None) -> str:
    """
    Diagnostic: status/event counts for JSON & HTML sources (no scraping/formatting).
    """
    league = (league or "epl").lower()
    date_str = date or _today_london_iso()
    try:
        _ = dtparser.isoparse(date_str)
        date_iso = date_str[:10]
    except Exception as e:
        return json.dumps({"error": f"bad date: {date} -> {e}"}, indent=2)

    # JSON probe
    try:
        _, jmeta = fetch_results_json(league, date_iso)
    except requests.HTTPError as e:
        jmeta = {"status": getattr(e.response, "status_code", None), "error": str(e)}
    except Exception as e:
        jmeta = {"status": None, "error": str(e)}

    # HTML probe
    url = schedule_url_for(league, date_iso)
    try:
        r = requests.get(url, timeout=15, headers=UA)
        hmeta = {"status": r.status_code, "len": len(r.text)}
    except Exception as e:
        hmeta = {"status": None, "error": str(e)}

    return json.dumps({"date": date_iso, "league": league, "json": jmeta, "html": hmeta, "schedule_url": url}, indent=2)

@mcp.tool()
def get_results(league: str = "epl", date: str | None = None, format: str = "markdown") -> str:
    """
    Get football scores/results for a league and date (Europe/London local date).
    league: epl, ucl, laliga, seriea, bundesliga, ligue1
    date: YYYY-MM-DD (defaults to today in Europe/London)
    format: 'markdown' or 'json'
    """
    league = (league or "epl").lower()
    date_str = date or _today_london_iso()

    try:
        _ = dtparser.isoparse(date_str)
        date_iso = date_str[:10]
    except Exception:
        raise ValueError("Invalid 'date'. Use YYYY-MM-DD.")

    results, meta = fetch_results(league, date_iso)

    if (format or "").lower() == "json":
        return json.dumps({"league": league, "date": date_iso, "source": meta.get("source"), "results": results}, ensure_ascii=False, indent=2)

    if not results:
        return f"**{league.upper()} — {date_iso}**\nNo fixtures or results found. _(source: {meta.get('source')}, meta: {json.dumps(meta)})_"

    lines = [f"**{league.upper()} — {date_iso}** _(source: {meta.get('source')})_"]
    for r in results:
        ko = f"({r.get('kickoff_local')}) " if r.get("kickoff_local") else ""
        hs, as_ = r.get("home_score"), r.get("away_score")
        have_scores = (hs is not None and as_ is not None)
        score = f"{hs}–{as_}" if have_scores else "vs"
        status = r.get("status") or ""
        lines.append(f"- {ko}{r['home']} {score} {r['away']} — {status}")
    return "\n".join(lines)

# ----------------------------- Entry -----------------------------
if __name__ == "__main__":
    print(">>> calling mcp.run()", file=sys.stderr, flush=True)
    mcp.run()
    print(">>> mcp.run() returned", file=sys.stderr, flush=True)
