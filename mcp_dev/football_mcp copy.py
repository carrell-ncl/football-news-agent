import json
from datetime import datetime
from dateutil import parser as dtparser
import pytz
import requests

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-football")

LEAGUE_MAP = {
    "epl": "soccer/eng.1",          # Premier League
    "ucl": "soccer/uefa.champions", # Champions League
    "laliga": "soccer/esp.1",
    "seriea": "soccer/ita.1",
    "bundesliga": "soccer/ger.1",
    "ligue1": "soccer/fra.1",
}

def _today_london_iso() -> str:
    now_ldn = datetime.now(pytz.timezone("Europe/London"))
    return now_ldn.strftime("%Y-%m-%d")

def fetch_results(league_key: str, date_iso: str):
    league_key = (league_key or "epl").lower()
    league_path = LEAGUE_MAP.get(league_key)
    if not league_path:
        raise ValueError(f"Unknown league '{league_key}'. Try one of: {', '.join(LEAGUE_MAP.keys())}")

    url = f"https://site.api.espn.com/apis/v2/sports/{league_path}/scoreboard"
    yyyymmdd = date_iso.replace("-", "")

    resp = requests.get(url, params={"dates": yyyymmdd, "limit": 300}, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    events = data.get("events", [])
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
            team = c.get("team", {})
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
    return results

@mcp.tool()
def get_results(league: str = "epl", date: str | None = None, format: str = "markdown") -> str:
    """
    Get football scores/results for a league and date (Europe/London local date).
    league: one of epl, ucl, laliga, seriea, bundesliga, ligue1
    date: YYYY-MM-DD (defaults to today in Europe/London)
    format: 'markdown' or 'json'
    """
    date_str = date or _today_london_iso()

    # Validate date
    try:
        _ = dtparser.isoparse(date_str)
        date_iso = date_str[:10]
    except Exception:
        raise ValueError("Invalid 'date'. Use YYYY-MM-DD.")

    results = fetch_results(league, date_iso)

    if format.lower() == "json":
        return json.dumps({"league": league, "date": date_iso, "results": results}, ensure_ascii=False, indent=2)

    # markdown
    if not results:
        return f"**{league.upper()} — {date_iso}**\nNo fixtures or results found."

    lines = [f"**{league.upper()} — {date_iso}**"]
    for r in results:
        ko = f"({r['kickoff_local']}) " if r['kickoff_local'] else ""
        have_scores = r['home_score'] is not None and r['away_score'] is not None
        score = f"{r['home_score']}–{r['away_score']}" if have_scores else "vs"
        lines.append(f"- {ko}{r['home']} {score} {r['away']} — {r['status']}")
    return "\n".join(lines)

if __name__ == "__main__":
    mcp.run()
