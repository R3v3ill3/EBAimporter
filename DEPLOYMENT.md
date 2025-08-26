## Deployment guide (Railway + Supabase Postgres)

This project uses SQLAlchemy + psycopg2 and now supports SSL, keepalives, timeouts, and IPv4 preference. If your web UI fails to load with database errors, follow these steps.

### 1) Required environment variables (Railway)

Set these in Railway project settings (Variables):

- DB_URL: Your full Postgres URL
  - Example: `postgresql://<user>:<password>@db.<ref>.supabase.co:5432/postgres`
- DB_SSLMODE: `require`
- DB_PREFER_IPV4: `true`  (forces IPv4 to avoid IPv6 egress issues)
- DB_CONNECT_TIMEOUT: `10` (optional)
- DB_POOL_SIZE: `5` (recommended for small instances)
- DB_MAX_OVERFLOW: `5`
- DB_HOSTADDR: optional IPv4 literal for the DB host (e.g. `1.2.3.4`). If set, the app will connect to this address directly while keeping the hostname for TLS SNI via libpq's `host` field.

The `Procfile` already binds Uvicorn to `0.0.0.0` and uses `PORT`, so no extra web vars are required.

### 2) Supabase database checks

- Ensure the database is running and the credentials in `DB_URL` are correct
- Supabase requires SSL; keep `DB_SSLMODE=require`
- Verify port `5432` is open and correct for your project
- If you use network restrictions, allow Railway egress IPs or disable the allowlist

### 3) Connectivity validation

From a local machine (or any environment with psql):

```bash
psql "sslmode=require host=db.<ref>.supabase.co port=5432 dbname=postgres user=<user> password=<password>"
```

If this works locally but fails on Railway with IPv6 addresses in logs, ensure `DB_PREFER_IPV4=true` is set. If DNS still resolves to IPv6 only from within the container, set `DB_HOSTADDR` to the IPv4 address of your Supabase DB (you can resolve it locally: `dig +short A db.<ref>.supabase.co`).

### 4) Runtime behavior

- On startup, the app attempts to initialize the DB and will log errors but still start
- The dashboard renders in degraded mode when the DB is unreachable
- The `/api/health` endpoint returns 503 with an error message if the DB is down

### 5) Tuning

- Adjust `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, and `DB_POOL_RECYCLE` for your plan limits
- Increase `DB_CONNECT_TIMEOUT` on slow networks

