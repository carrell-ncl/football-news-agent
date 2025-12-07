import argparse
import hashlib
import os
import sqlite3
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup

from datetime import timedelta
from email.utils import parsedate_to_datetime

DEFAULT_UA = "NUFC-News-Agent/1.0 (personal use)"



def is_too_old(published_dt_utc: datetime | None, max_age_hours: int | None) -> bool:
    if not max_age_hours:
        return False
    if published_dt_utc is None:
        # If no date is available, treat as fresh (or flip to True if you prefer strict)
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    return published_dt_utc < cutoff

def _rss_entry_published_dt(entry) -> datetime | None:
    """
    Try multiple fields: published/updated (RFC822) → datetime (UTC).
    """
    for key in ("published", "updated", "dc_date"):
        val = entry.get(key)
        if not val:
            continue
        try:
            dt = parsedate_to_datetime(val)
            # normalize to UTC if tz-aware, else assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            pass
    # Some feeds expose structured *_parsed
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            try:
                dt = datetime(*t[:6], tzinfo=timezone.utc)
                return dt
            except Exception:
                pass
    return None

def _html_article_published_dt(html: str) -> datetime | None:
    """
    Best-effort parse from common meta tags on the article page.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        # common patterns
        candidates = [
            ("meta", {"property": "article:published_time"}),
            ("meta", {"name": "article:published_time"}),
            ("meta", {"name": "pubdate"}),
            ("meta", {"itemprop": "datePublished"}),
            ("time", {"itemprop": "datePublished"}),
            ("time", {"datetime": True}),
        ]
        for tag, attrs in candidates:
            el = soup.find(tag, attrs)
            if not el:
                continue
            iso = el.get("content") or el.get("datetime")
            if iso:
                try:
                    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                    return dt.astimezone(timezone.utc)
                except Exception:
                    pass
        return None
    except Exception:
        return None


# ----------------------------
# Utilities
# ----------------------------
def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")

import json

def overall_10_bullet_summary(endpoint: str, model: str, bullets: list[str]) -> str:
    prompt = (
        "You are a concise football news summarizer.\n"
        "From the following bullet summaries about Newcastle United, write EXACTLY 10 bullets.\n"
        "- Be specific (players, injuries, transfers, quotes, fixtures).\n"
        "- Remove duplicates, merge overlaps.\n"
        "- Max ~20 words per bullet.\n"
        "- No preamble, no numbering, just 10 dash-led bullets.\n\n"
        "INPUT BULLETS:\n" + "\n".join(f"- {b.strip()}" for b in bullets if b.strip())
        + "\n\nOUTPUT (exactly 10 bullets):\n"
    )
    try:
        r = requests.post(
            f"{endpoint.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 400}},
            timeout=90
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"(overall summary error: {e})"


def ensure_db(path: str):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen (
            id TEXT PRIMARY KEY,
            source TEXT,
            url TEXT,
            title TEXT,
            first_seen_utc TEXT
        )
    """)
    # add columns if missing
    cols = {r[1] for r in conn.execute("PRAGMA table_info(seen)")}
    if "sentiment_score" not in cols:
        conn.execute("ALTER TABLE seen ADD COLUMN sentiment_score REAL")
    if "sentiment_label" not in cols:
        conn.execute("ALTER TABLE seen ADD COLUMN sentiment_label TEXT")
    conn.commit()
    return conn

def notify_console(item, summary: str = "", sentiment=None):
    sep = "-" * 80
    print(sep)
    print(item["title"])
    print(item["url"])
    if sentiment is not None:
        sc, lab, note = sentiment
        print(f"Sentiment: {lab} ({sc:+.2f}) {note and '- ' + note if note else ''}")
    if summary:
        print("\nSummary:\n" + summary)



def _safe_json_extract(text: str):
    try:
        return json.loads(text)
    except Exception:
        # Try to extract a JSON object if the model added extra text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                return None
    return None

def sentiment_via_ollama(endpoint: str, model: str, text: str, num_predict: int = 80, keep_alive: str | None = None):
    """
    Returns (score, label, reasons) where score is float in [-1, 1].
    """
    prompt = (
        "You are rating football news sentiment about Newcastle United.\n"
        "Return STRICT JSON ONLY with keys: sentiment (float -1..1), label ('negative'|'neutral'|'positive'), reasons (string).\n\n"
        f"Text:\n{text}\n\n"
        "JSON:"
    )
    options = {"num_predict": num_predict, "temperature": 0.2}
    if keep_alive:
        options["keep_alive"] = keep_alive
    try:
        r = requests.post(
            f"{endpoint.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": options},
            timeout=60
        )
        r.raise_for_status()
        data = r.json()
        payload = _safe_json_extract(data.get("response", "").strip())
        if not payload:
            return 0.0, "neutral", "(could not parse sentiment JSON)"
        score = float(payload.get("sentiment", 0.0))
        score = max(-1.0, min(1.0, score))
        label = str(payload.get("label", "neutral"))
        reasons = str(payload.get("reasons", ""))
        return score, label, reasons
    except Exception as e:
        return 0.0, "neutral", f"(sentiment error: {e})"

# Optional: ultra-fast rule-based sentiment (no LLM). pip install vaderSentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except Exception:
    _vader = None

def sentiment_via_vader(text: str):
    if _vader is None:
        return 0.0, "neutral", "(vader not installed)"
    s = _vader.polarity_scores(text)["compound"]  # already -1..1
    label = "positive" if s > 0.05 else "negative" if s < -0.05 else "neutral"
    return float(s), label, "(vader)"


def hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def save_summary_to_file(filename, item, summary):
    """Append the summary of an article to a text file."""
    with open(filename, "a", encoding="utf-8") as f:
        f.write("-" * 80 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {item['source']}\n")
        f.write(f"Title: {item['title']}\n")
        f.write(f"URL: {item['url']}\n\n")
        f.write("Summary:\n")
        f.write(summary.strip() + "\n\n")



def ensure_db(path: str):
    conn = sqlite3.connect(path)
    # Create table if needed
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen (
            id TEXT PRIMARY KEY,
            source TEXT,
            url TEXT,
            title TEXT,
            first_seen_utc TEXT
        )
    """)

    # ---- add columns if missing (safe on reruns) ----
    cols = {r[1] for r in conn.execute("PRAGMA table_info(seen)")}
    if "sentiment_score" not in cols:
        conn.execute("ALTER TABLE seen ADD COLUMN sentiment_score REAL")
    if "sentiment_label" not in cols:
        conn.execute("ALTER TABLE seen ADD COLUMN sentiment_label TEXT")
    if "published_at_utc" not in cols:
        conn.execute("ALTER TABLE seen ADD COLUMN published_at_utc TEXT")

    conn.commit()
    return conn



def already_seen(conn, uid: str) -> bool:
    cur = conn.execute("SELECT 1 FROM seen WHERE id=?", (uid,))
    return cur.fetchone() is not None


def mark_seen(conn, uid: str, source: str, url: str, title: str, published_at_utc: datetime | None = None):
    conn.execute(
        "INSERT OR IGNORE INTO seen (id, source, url, title, first_seen_utc, published_at_utc) VALUES (?,?,?,?,?,?)",
        (uid, source, url, title, datetime.now(timezone.utc).isoformat(),
         published_at_utc.isoformat() if published_at_utc else None),
    )
    conn.commit()



# ----------------------------
# Fetchers
# ----------------------------
def fetch_rss(source: dict, headers: dict):
    d = feedparser.parse(source["url"])
    items = []
    for e in d.entries:
        link = e.get("link")
        title = e.get("title", "(no title)")
        summary = e.get("summary", "")
        published_dt = _rss_entry_published_dt(e)  # NEW
        items.append({
            "title": title,
            "url": link,
            "summary": summary,
            "source": source["name"],
            "published_at_utc": published_dt,  # NEW
        })
    return items



def fetch_html(source: dict, headers: dict) -> List[Dict[str, Any]]:
    resp = requests.get(source["url"], headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    selector = source.get("selector") or "article a"
    base = source["url"]
    allow_domains = set(source.get("allow_domains", []))

    items = []
    for i, a in enumerate(soup.select(selector)):
        print(f"Processing Article: {i+1}")
        href = a.get("href")
        title = (a.get_text(strip=True) or "").strip()
        if not href:
            continue
        url = urljoin(base, href)
        if allow_domains:
            if urlparse(url).hostname not in allow_domains:
                continue
        items.append({
            "title": title or url,
            "url": url,
            "summary": "",
            "source": source["name"],
        })
    return items


# ----------------------------
# Content fetching & filtering
# ----------------------------
def get_article_text(url: str, headers: dict, max_chars: int = 8000):
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for sel in ["article", "main", "div[itemprop='articleBody']", "#content"]:
            el = soup.select_one(sel)
            if el:
                text = el.get_text("\n", strip=True)
                return text[:max_chars]
        return soup.get_text("\n", strip=True)[:max_chars]
    except Exception as e:
        return f"(error fetching full text: {e})"


def passes_keywords(text: str, keywords):
    if not keywords:
        return True
    text_l = text.lower()
    return any(k.lower() in text_l for k in keywords)


# ----------------------------
# Summarization via Ollama
# ----------------------------
def ollama_summarize(endpoint: str, model: str, text: str, max_tokens: int = 256) -> str:
    try:
        payload = {
            "model": model,
            "prompt": f"Summarize in 3-5 bullets (focus: Newcastle United):\n\n{text}\n\n",
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        r = requests.post(f"{endpoint}/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")
    except Exception as e:
        return f"(local summarizer error: {e})"


# ----------------------------
# Notifications
# ----------------------------
# def notify_console(item, summary: str = ""):
#     sep = "-" * 80
#     print(sep)
#     print(item["title"])
#     print(item["url"])
#     if summary:
#         print("\nSummary:\n" + summary)


def notify_email(cfg_email: dict, subject: str, body: str):
    if not cfg_email.get("enabled"):
        return
    import smtplib
    from email.mime.text import MIMEText

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = cfg_email["from_addr"]
    msg["To"] = cfg_email["to_addr"]

    with smtplib.SMTP(cfg_email["smtp_host"], cfg_email["smtp_port"]) as s:
        s.starttls()
        s.login(cfg_email["username"], cfg_email["password"])
        s.send_message(msg)


# ----------------------------
# Main loop
# ----------------------------
def run_once(cfg):
    headers = {"User-Agent": cfg.get("user_agent") or DEFAULT_UA}
    conn = ensure_db(cfg.get("state_db", "news_state.sqlite"))
    keywords = cfg.get("keywords", [])

    items_new = []

    max_age_hours = (cfg.get("windows") or {}).get("max_age_hours")

    for src in cfg.get("sources", []):
        src_type = src.get("type", "rss").lower()
        try:
            if src_type == "rss":
                items = fetch_rss(src, headers)
                # Filter early using feed times (cheap)
                if max_age_hours:
                    items = [it for it in items if not is_too_old(it.get("published_at_utc"), max_age_hours)]
            else:
                items = fetch_html(src, headers)
        except Exception as e:
            log(f"Error fetching {src['name']}: {e}")
            continue

        time.sleep(cfg.get("request_interval_seconds", 2))

        for it in items:
            # For HTML sources (or RSS items with missing date), fetch page and extract published time before heavy work
            published_at_utc = it.get("published_at_utc")
            full_text_html = None

            if published_at_utc is None and max_age_hours:
                # Fetch once to check meta date (and reuse for content)
                try:
                    r = requests.get(it["url"], headers=headers, timeout=20)
                    r.raise_for_status()
                    full_text_html = r.text
                    published_at_utc = _html_article_published_dt(full_text_html)
                    if is_too_old(published_at_utc, max_age_hours):
                        continue  # too old → skip
                except Exception:
                    # If we can't get the page/date, you can choose to skip or keep. We'll keep by default.
                    pass

            uid = hash_str(f"{it['source']}::{it['url']}::{it['title']}")
            if already_seen(conn, uid):
                continue

            # Get content (reuse the HTML if we already fetched it)
            if full_text_html is not None:
                soup = BeautifulSoup(full_text_html, "html.parser")
                text = None
                for sel in ["article", "main", "div[itemprop='articleBody']", "#content"]:
                    el = soup.select_one(sel)
                    if el:
                        text = el.get_text("\n", strip=True)
                        break
                full_text = (text or soup.get_text("\n", strip=True))[:cfg.get("summarizer", {}).get("max_chars", 4000)]
            else:
                full_text = get_article_text(
                    it["url"], headers,
                    max_chars=cfg.get("summarizer", {}).get("max_chars", 4000)
                )

            content_for_filter = (" ".join([it.get("title", ""), it.get("summary", ""), full_text])).strip()
            if not passes_keywords(content_for_filter, keywords):
                continue

            mark_seen(conn, uid, it["source"], it["url"], it["title"])

            summary = ""
            sum_cfg = cfg.get("summarizer", {})
            if sum_cfg.get("enabled"):
                print(f"Summarising with Llama LLM")
                summary = ollama_summarize(
                    sum_cfg.get("endpoint", "http://localhost:11434"),
                    sum_cfg.get("model", "llama3.2:3b"),
                    full_text
                )

            # Carry forward values we need later (avoid scope bugs)
            items_new.append((it, summary, uid, full_text))

    # After the loop that writes individual summaries…
    overall_cfg = cfg.get("overall_summary", {})
    if overall_cfg.get("enabled", True):
        endpoint = overall_cfg.get("endpoint", cfg.get("summarizer", {}).get("endpoint", "http://localhost:11434"))
        model = overall_cfg.get("model",  cfg.get("summarizer", {}).get("model", "llama3.2:3b"))
        outfile = overall_cfg.get("output_file", "llm_summary_overall.txt")

        # gather this run's summaries (skip empties)
        run_bullets = [summary for _, summary, _, _ in items_new if summary and summary.strip()]
        if run_bullets:
            overall = overall_10_bullet_summary(endpoint, model, run_bullets)
            with open(outfile, "a", encoding="utf-8") as f:
                f.write("-" * 80 + "\n")
                f.write(f"Overall (10 bullets) @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(overall + "\n\n")



    for it, summary, uid, full_text in items_new:
        sentiment_tuple = None
        if cfg.get("sentiment", {}).get("enabled"):
            sent_cfg = cfg["sentiment"]
            sent_text_choice = (sent_cfg.get("input", "summary") or "summary").lower()
            text_for_sentiment = summary if sent_text_choice == "summary" else full_text
            if sent_cfg.get("provider", "ollama") == "vader":
                sentiment_tuple = sentiment_via_vader(text_for_sentiment)
            else:
                sentiment_tuple = sentiment_via_ollama(
                    endpoint=sent_cfg.get("endpoint", "http://localhost:11434"),
                    model=sent_cfg.get("model", "llama3.2:3b"),
                    num_predict=int(sent_cfg.get("num_predict", 80)),
                    keep_alive=sent_cfg.get("keep_alive")
                )

        if sentiment_tuple:
            sc, lab, _ = sentiment_tuple
            conn.execute(
                "UPDATE seen SET sentiment_score=?, sentiment_label=? WHERE id=?",
                (sc, lab, uid)
            )
            conn.commit()

        if cfg.get("notify", {}).get("console", True):
            notify_console(it, summary, sentiment_tuple)

        save_summary_to_file(cfg.get("output_file", "summaries.txt"), it, summary + (
            f"\n\n[Sentiment] {sentiment_tuple[1]} ({sentiment_tuple[0]:+.2f})" if sentiment_tuple else ""
        ))

        # (Optional) email notify unchanged


    # ... existing email code unchanged


        # Save to text file
        save_summary_to_file("llm_summary.txt", it, summary)

        email_cfg = cfg.get("notify", {}).get("email", {})
        if email_cfg.get("enabled"):
            body = f"{it['title']}\n{it['url']}\n\n{summary}"
            try:
                notify_email(email_cfg, f"[NUFC] {it['title']}", body)
            except Exception as e:
                log(f"Email notify failed: {e}")

    log(f"Checked {len(cfg.get('sources', []))} source(s). New items: {len(items_new)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--once", action="store_true", help="Run a single check and exit")
    ap.add_argument("--every", type=int, help="Run forever; seconds between checks")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.once:
        run_once(cfg)
        return

    interval = args.every or 900
    while True:
        try:
            run_once(cfg)
        except Exception as e:
            log(f"Run failed: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
