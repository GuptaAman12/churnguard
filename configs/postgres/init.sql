-- Runs only when the PostgreSQL data directory is first initialised.
-- MLflow uses the default POSTGRES_DB; Prefect gets its own database.
CREATE DATABASE prefect;
