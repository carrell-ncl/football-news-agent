#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCP server for your existing SQLite news table with columns:

id TEXT (PK-ish), source TEXT, url TEXT, title TEXT,
first_seen_utc TEXT, sentiment_score REAL, sentiment_label TEXT, published_at_utc TEXT

No migrations. Reads only (safe). Works with SQLite by default; optional MySQL.
If SEEN_TABLE is not set for SQLite, we autodetect the table that has these columns.
"""

import os, sys, json, re
from typing import Optional, List, Tuple

from mcp.server.fastmcp import FastMCP

print(">>> news_mcp starting", file=sys.stderr, flush=True)
mcp = FastMCP("mcp-news")

# -------- Config (env) --------
DB_DRIVER = os.getenv("DB_DRIVER", "sqlite").lower()
DB_PATH   = os.getenv("DB_PATH", "/Users/carrell/Desktop/dev/agentic_news/news_state.sqlite")

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "")

# Optional explicit table name. If empty and SQLite, we try to detect it.
EXPLICIT_TABLE = os.getenv("SEEN_TABLE", "").strip()

# Expected schema (column names)
COLS = [
    "id", "source", "url", "title",
    "first_seen_utc", "sentiment_score", "sentiment_label", "published_at_utc"
]
COLSET = set(COLS)

# -------- DB wrapper --------
class DB:
    def __init__(self):
        self.driver = DB_DRIVER
        if self.driver == "sqlite":
            import sqlite3
            self.sqlite3 = sqlite3
            self.conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)  # autocommit
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA foreign_keys=ON;")
            self.table = EXPLICIT_TABLE or self._detect_sqlite_table()
            if not self.table:
                raise RuntimeError(
                    "Could not detect a table with the expected columns. "
                    "Set SEEN_TABLE env to your table name."
                )
            print(f">>> sqlite path={DB_PATH} table={self.table}", file=sys.stderr, flush=True)
        elif self.driver == "mysql":
            try:
                import pymysql  # requires `pip install pymysql` (+ cryptography for caching_sha2_password)
            except ImportError as e:
                print("Missing dependency: pymysql (pip install pymysql)", file=sys.stderr, flush=True)
                raise
            self.pymysql = pymysql
            self.conn = pymysql.connect(
                host=MYSQL_HOST, port=MYSQL_PORT,
                user=MYSQL_USER, password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE, autocommit=True, charset="utf8mb4"
            )
            # With MySQL you MUST set SEEN_TABLE explicitly.
            self.table = EXPLICIT_TABLE
            if not self.table:
                raise RuntimeError("For MySQL, set SEEN_TABLE to your table name.")
            print(f">>> mysql host={MYSQL_HOST} db={MYSQL_DATABASE} table={self.table}", file=sys.stderr, flush=True)
        else:
            raise ValueError(f"Unsupported DB_DRIVER: {self.driver}")

        self._validate_schema()

    # --- helpers ---
    def _detect_sqlite_table(self) -> Optional[str]:
        cur = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        for t in tables:
            try:
                cols = self.conn.execute(f"PRAGMA table_info({t})").fetchall()
                names = {c[1] for c in cols}  # c[1] is column name
                if COLSET.issubset(names):
                    return t
            except Exception:
                pass
        return None

    def _validate_schema(self):
        # Ensure the chosen table has at least the required columns
        if self.driver == "sqlite":
            cols = self.conn.execute(f"PRAGMA table_info({self.table})").fetchall()
            names = {c[1] for c in cols}
        else:
            with self.conn.cursor() as cur:
                cur.execute(f"SHOW COLUMNS FROM {self.table}")
                names = {row[0] for row in cur.fetchall()}
        missing = COLSET - names
        if missing:
            raise RuntimeError(f"Table '{self.table}' is missing columns: {', '.join(sorted(missing))}")

    def query(self, sql: str, params: Tuple = ()) -> List[Tuple]:
        if self.driver == "sqlite":
            cur = self.conn.execute(sql, params)
            return cur.fetchall()
        else:
            with self.conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()

db = DB()

# -------- Shared SQL bits --------
ORDER_ALLOWED = {"asc", "desc"}

def _order_dir(dir_str: Optional[str]) -> str:
    if not dir_str:
        return "DESC"
    d = dir_str.strip().lower()
    return "ASC" if d == "asc" else "DESC"

# -------- Tools --------
@mcp.tool()
def ping() -> str:
    """Health check."""
    return "pong"

@mcp.tool()
def latest() -> str:
    """
    Return the most recent article by COALESCE(first_seen_utc, published_at_utc) DESC.
    """
    sql = f"""
    SELECT id, source, url, title, first_seen_utc, sentiment_score, sentiment_label, published_at_utc
    FROM {db.table}
    ORDER BY COALESCE(first_seen_utc, published_at_utc) DESC
    LIMIT 1
    """
    rows = db.query(sql)
    if not rows:
        return json.dumps({"found": False}, ensure_ascii=False, indent=2)
    (id_, source, url, title, first_seen, score, label, published) = rows[0]
    return json.dumps({
        "found": True,
        "article": {
            "id": id_,
            "source": source,
            "url": url,
            "title": title,
            "first_seen_utc": first_seen,
            "sentiment_score": float(score) if score is not None else None,
            "sentiment_label": label,
            "published_at_utc": published
        }
    }, ensure_ascii=False, indent=2)

@mcp.tool()
def get(url: str) -> str:
    """
    Get one article by URL.
    """
    if not url:
        raise ValueError("url is required")
    sql = f"""
    SELECT id, source, url, title, first_seen_utc, sentiment_score, sentiment_label, published_at_utc
    FROM {db.table}
    WHERE url = ?
    LIMIT 1
    """ if db.driver == "sqlite" else f"""
    SELECT id, source, url, title, first_seen_utc, sentiment_score, sentiment_label, published_at_utc
    FROM {db.table}
    WHERE url = %s
    LIMIT 1
    """
    rows = db.query(sql, (url,))
    if not rows:
        return json.dumps({"found": False, "url": url}, ensure_ascii=False, indent=2)
    (id_, source, url, title, first_seen, score, label, published) = rows[0]
    return json.dumps({
        "found": True,
        "article": {
            "id": id_,
            "source": source,
            "url": url,
            "title": title,
            "first_seen_utc": first_seen,
            "sentiment_score": float(score) if score is not None else None,
            "sentiment_label": label,
            "published_at_utc": published
        }
    }, ensure_ascii=False, indent=2)

@mcp.tool()
def list(limit: int = 50, since: Optional[str] = None, source: Optional[str] = None,
         search: Optional[str] = None, order: Optional[str] = "desc") -> str:
    """
    List recent articles.
    - limit: 1..500
    - since: filter by COALESCE(first_seen_utc, published_at_utc) >= since (YYYY-MM-DD or full timestamp)
    - source: exact source filter (e.g., 'bbc')
    - search: substring match on title (case-insensitive)
    - order: 'asc' | 'desc' (default desc)
    """
    limit = max(1, min(int(limit), 500))
    order_dir = _order_dir(order)

    where = []
    params: List = []

    if since:
        where.append("COALESCE(first_seen_utc, published_at_utc) >= ?")
        params.append(since)

    if source:
        where.append("source = ?")
        params.append(source)

    if search:
        if db.driver == "sqlite":
            where.append("LOWER(title) LIKE ?")
            params.append(f"%{search.lower()}%")
        else:
            where.append("LOWER(title) LIKE ?")
            params.append(f"%{search.lower()}%")

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
    SELECT id, source, url, title, first_seen_utc, sentiment_score, sentiment_label, published_at_utc
    FROM {db.table}
    {where_sql}
    ORDER BY COALESCE(first_seen_utc, published_at_utc) {order_dir}
    LIMIT ?
    """
    params.append(limit)

    rows = db.query(sql, tuple(params))
    items = []
    for (id_, source, url, title, first_seen, score, label, published) in rows:
        items.append({
            "id": id_,
            "source": source,
            "url": url,
            "title": title,
            "first_seen_utc": first_seen,
            "sentiment_score": float(score) if score is not None else None,
            "sentiment_label": label,
            "published_at_utc": published
        })
    return json.dumps({"count": len(items), "items": items}, ensure_ascii=False, indent=2)

@mcp.tool()
def show_latest(markdown: bool = True) -> str:
    """
    Show the most recent article in a friendly format.
    Natural prompts that map here:
      - "show me the latest news article"
      - "what's the newest story?"
    """
    raw = json.loads(latest())
    if not raw.get("found"):
        return "No articles found."
    a = raw["article"]
    if not markdown:
        return json.dumps(a, ensure_ascii=False, indent=2)

    lines = [
        f"### 📰 {a['title']}",
        f"- Source: **{a['source']}**",
        f"- First seen: {a['first_seen_utc'] or a['published_at_utc']}",
        f"- Sentiment: {a['sentiment_label']} ({a['sentiment_score']})",
        f"- URL: {a['url']}",
    ]
    return "\n".join(lines)

@mcp.tool()
def find_news(query: str = "", source: str | None = None, limit: int = 5) -> str:
    """
    Search recent articles by title (and optional source).
    Natural prompts:
      - "find news about aston villa"
      - "show 3 bbc stories on oil"
    """
    res = json.loads(list(limit=limit, search=query, source=source))
    if not res.get("items"):
        return f"No matches for '{query}'."
    out = ["### Results"]
    for a in res["items"]:
        out.append(f"- **{a['title']}** — {a['source']} • {a['first_seen_utc'] or a['published_at_utc']}\n  {a['url']}")
    return "\n".join(out)

@mcp.tool()
def show_latest_from(source: str, limit: int = 3) -> str:
    """
    Show latest N articles from a source.
    Natural prompt: "show the latest 3 from bbc"
    """
    res = json.loads(list(limit=limit, source=source))
    if not res.get("items"):
        return f"No recent items from '{source}'."
    out = [f"### Latest from {source}"]
    for a in res["items"]:
        out.append(f"- **{a['title']}** • {a['first_seen_utc'] or a['published_at_utc']}\n  {a['url']}")
    return "\n".join(out)


# -------- Entry --------
if __name__ == "__main__":
    print(f">>> driver={DB_DRIVER}", file=sys.stderr, flush=True)
    if DB_DRIVER == "sqlite":
        print(f">>> db={DB_PATH}", file=sys.stderr, flush=True)
    else:
        print(f">>> mysql host={MYSQL_HOST} db={MYSQL_DATABASE}", file=sys.stderr, flush=True)
    mcp.run()
