# Expenses Tracker

Local pipeline for importing bank statements, tokenizing transaction descriptions, normalizing sources, training category predictions, and exploring transactions in Grafana.

## Project layout

- `pipeline/`: importable application package containing pipeline stages, configuration, and shared utilities.
- `tests/`: automated regression tests.
- `data/`: SQL query reference.
- `dashboard/`: Grafana provisioning, dashboard JSON, Dockerfile, and Compose service.
- `docs/`: technical notes and dashboard setup guide.

## Statement types

The pipeline handles account statements (Axis CSV) and credit card statements (Axis xlsx, ICICI pdf). Every stage takes `--statement account|creditcard` (default `account`) and uses separate tables: `acc_raw`, `acc_curated`, `acc_tokenized`, `acc_normalized`, `acc_categorized` and the `cc_*` equivalents, plus `{acc,cc}_transactions` and `{acc,cc}_transaction_tokens` Chroma collections.

Credit card statements are first converted to CSV with `expenses-tracker extract --statement creditcard --bank axis|icici`. Token mappings and category labels are learned from account data and reused for credit cards, so run the account stages first. See `docs/commands.txt` for the full command sequence.

All processing is local; the optional LLM steps only accept a localhost `LOCAL_LLM_BASE_URL`.
- `artifacts/`, `logs/`: local generated model outputs and runtime logs, ignored by Git; use a private artifact store or backup for durable history.

## Configuration

Copy `.env.example` to `.env` and set `root_path` and `delta_dir`. The configured data directory should contain `schema/expenses.db`. Do not commit `.env`, credentials, statement CSVs, or database files.
Copy `config.ini.example` to the Git-ignored `config.ini` before running the pipeline. `BANK_NAMES`, `PAYMENT_MODES`, and `JUNK_WORDS` are loaded from that file; keep account-specific payee names only in your local `config.ini`.

## Common tasks

- Run tests: `python -m unittest discover -s tests`
- Show pipeline commands: `python main.py --help`
- Run statement ingestion: `python main.py ingest --mode incremental`
- Tokenize transactions: `python main.py tokenize --mode incremental`
- Normalize sources: `python main.py canonicalize`
- Predict categories: `python main.py predict --no-write`
- Generate embeddings: `python main.py embed`
- Install the fully locked environment: `uv sync --extra all`
- Install base dependencies with pip: `pip install -e .`
- Install ML/embedding and local-LLM extras with pip: `pip install -e '.[ml,llm]'`
- Start Grafana from the repository root: `docker compose --env-file .env -f dashboard/docker-compose.yml up -d --build`
- Stop Grafana: `docker compose --env-file .env -f dashboard/docker-compose.yml down`

See [docs/dashboard.md](docs/dashboard.md) for dashboard details, including candidate recurring-income sources.

## Amazon Pay history capture

Install the project's dependencies and Playwright's Chromium browser (`uv run playwright install chromium`), then run `uv run python scripts/amazon_transactions.py` from the project root. Log in in the browser, open the transaction history, and press Enter in the terminal. The script keeps responses in memory and writes one CSV to `amazon_pay_responses/transactions.csv` when the download completes; it creates no JSON files. Use `--output` to choose another path or `--overwrite` to replace an existing CSV.

Optionally add `--start-date YYYY-MM-DD`, `--end-date YYYY-MM-DD`, or both. Bounds are inclusive calendar dates in India time; the CSV retains Amazon's original timestamps. With no dates all transactions are included. Filtering is performed locally after fetching pages, so the script still scrolls all history. Every matching transaction is included, including duplicates returned by Amazon. If scrolling stalls, a response fails, or a date needed for filtering is invalid, the script reports the error and does not write the CSV. Existing JSON captures are not read or deleted. The CSV contains sensitive financial information; do not share or commit it or put session cookies or credentials in the script.
