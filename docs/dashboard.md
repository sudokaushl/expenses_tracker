# Expenses Tracker Grafana Dashboard

Local Grafana dashboards for account and credit-card transactions. The Account Overview uses `expenses.acc_labelled`; the Credit Card Overview uses `expenses.cc_categorized`. Grafana reads the database through the MotherDuck DuckDB datasource plugin; the database mount and datasource are read-only. Both dashboards share the same datasource, visual theme, date controls, and filter conventions, and each links to the other.

## Start

From repository root, run:

```sh
docker compose --env-file .env -f dashboard/docker-compose.yml up -d --build
```

Copy `.env.example` to `.env` and set `delta_dir`; Compose mounts `${delta_dir}/schema/expenses.db` into Grafana. The image pins Grafana 11.6.1 and DuckDB plugin v0.4.6, verifying the plugin SHA-256 during build. A health check polls Grafana's health endpoint.

Open [http://localhost:3001](http://localhost:3001). Port 3001 avoids conflicting with the existing Grafana container on port 3000. First login is `admin` / `admin`; set a new password when Grafana prompts.

## Dashboard controls

- Time range picker controls date coverage.
- Account Overview filters by payment mode, category, subcategory, and merchant/source; Credit Card Overview uses the same controls with card issuer, category, subcategory, and merchant/source.
- Minimum and maximum amount text fields filter transaction amounts; each dashboard also includes a monthly budget input.
- Account panels cover income, spend, cash flow, budget, categories, and recurring trends. Credit Card panels cover purchase/payment mix, daily and monthly trends, category and issuer distributions, merchant rankings, amount bands, recent transactions, data quality, budget, and year-over-year purchase trends.
- Use the dashboard link at the top to switch between account and credit-card views while carrying time range and filter variables.
- Dashboard refreshes every five minutes; use the time picker refresh controls to change cadence.

Amounts use INR formatting. `trans_type = 'credit'` represents credits or repayments and `trans_type = 'debit'` represents purchases. Account visualizations query `acc_labelled`; credit-card panels query `cc_categorized` so they show the current prediction output including manually preserved labels.

DuckDB supports limited cross-process concurrency. Stop Grafana before running ETL jobs that write to this database; restart it after updates.

## Stop / restart

```sh
docker compose --env-file .env -f dashboard/docker-compose.yml down

docker compose --env-file .env -f dashboard/docker-compose.yml up -d
```

Grafana settings persist in the `grafana-data` Docker volume. `down` keeps that volume. To remove dashboard settings too, run `docker compose --env-file .env -f dashboard/docker-compose.yml down -v`.

## Files

- `dashboard/Dockerfile` — Ubuntu-based Grafana image and pinned DuckDB plugin.
- `dashboard/docker-compose.yml` — localhost-only port, read-only database mount, and provisioning mounts.
- `dashboard/provisioning/` — provisioned datasource and dashboard provider.
- `dashboard/dashboards/acc.json` — account dashboard layout, filters, and panel SQL.
- `dashboard/dashboards/cc.json` — credit-card dashboard layout, filters, and panel SQL.
- `data/queries.sql` — standalone query reference.
