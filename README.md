sqlite3 news_state.sqlite

-- Make a full copy with data
CREATE TABLE IF NOT EXISTS seen_backup AS
SELECT * FROM seen
limit 1;

-- “Truncate” in SQLite = delete all rows
DELETE FROM seen;

-- Optionally reclaim disk space
VACUUM;
.quit



python news_agent.py --config config.yaml --once --overall-after

python news_agent.py --config config.yaml --once --overall-only


    "seen": {
      "command": "/Users/carrell/opt/anaconda3/envs/agentic_news/bin/python",
      "args": ["/Users/carrell/Desktop/dev/agentic_news/mcp_dev/seen_mcp.py"],
      "env": {
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "DB_DRIVER": "mysql",
        "MYSQL_HOST": "127.0.0.1",
        "MYSQL_PORT": "3306",
        "MYSQL_USER": "mcp_user",
        "MYSQL_PASSWORD": "ChangeMeStrong!123",
        "MYSQL_DATABASE": "agentic_seen"
      }
    }