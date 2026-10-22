-- Database roles for the monitoring store, applied by `make up` (idempotent).
-- forecast: the application's role (it owns the tables); its password comes from .env.
-- grafana_reader: what Grafana connects as. SELECT on the monitoring tables only, no
-- superuser rights, so a dashboard query cannot write, read files or run programs.
\set ON_ERROR_STOP on
ALTER ROLE forecast PASSWORD :'app_password';
SELECT 'CREATE ROLE grafana_reader' WHERE NOT EXISTS
  (SELECT FROM pg_roles WHERE rolname = 'grafana_reader') \gexec
ALTER ROLE grafana_reader LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD :'reader_password';
GRANT CONNECT ON DATABASE forecast TO grafana_reader;
GRANT USAGE ON SCHEMA public TO grafana_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE forecast IN SCHEMA public GRANT SELECT ON TABLES TO grafana_reader;
