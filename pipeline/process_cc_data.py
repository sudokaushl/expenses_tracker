"""Load extracted credit card CSVs into DuckDB and append them to the shared curated table. Offline only."""

from __future__ import annotations

import hashlib
import os
from glob import glob

import pandas as pd

from . import config
from .utils import check_table_exists, create_duckdb_connection, get_logger, load_data_with_check

logger = get_logger()


def _read_bank_csvs(bank: str) -> pd.DataFrame:
    csv_dir = os.path.join(config.landing_dir("creditcard", bank), "csv")
    files = sorted(glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No extracted CSVs in {csv_dir}; run the extract command first")

    frames = []
    for path in files:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        df["trans_dt"] = pd.to_datetime(df["trans_dt"], errors="raise")
        df["trans_amt"] = pd.to_numeric(df["trans_amt"], errors="raise")
        # Counter keeps genuine same-day repeats distinct; overlapping files still collapse to identical ids
        key = ["trans_dt", "ref_num", "trans_details", "trans_amt", "trans_type"]
        df["_seq"] = df.groupby(key, dropna=False).cumcount()
        frames.append(df)
        logger.info("Read %s rows from %s", len(df), os.path.basename(path))

    df = pd.concat(frames, ignore_index=True)
    df["bank"] = bank
    df["trans_id"] = df.apply(
        lambda r: hashlib.md5(
            "|".join(str(r[c]) for c in ("bank", "trans_dt", "ref_num", "trans_details", "trans_amt", "trans_type", "_seq")).encode()
        ).hexdigest(),
        axis=1,
    )
    return df.drop(columns="_seq")


def main(ingest_mode: str, bank: str) -> None:
    banks = config.BANKS["creditcard"] if bank == "all" else (bank,)
    t = config.tables("creditcard")
    con = None
    try:
        logger.info("=============== Starting Credit Card Ingestion =============")
        con = create_duckdb_connection(config.DATABASE_PATH)

        df = pd.concat([_read_bank_csvs(b) for b in banks], ignore_index=True)
        df["particulars"] = df["trans_details"]
        df["trans_mode"] = "cc_" + df["bank"]
        df["trans_type"] = df["trans_type"].str.lower()  # curated table uses 'debit'/'credit'
        con.register("cc_df", df)

        if ingest_mode == config.HISTORICAL:
            # Replace only the selected banks so other banks' rows survive
            bank_list = ",".join(f"'{b}'" for b in banks)
            if check_table_exists(con, config.SCHEMA_NAME, t.raw):
                con.execute(f"DELETE FROM {config.SCHEMA_NAME}.{t.raw} WHERE bank IN ({bank_list})")
            if check_table_exists(con, config.SCHEMA_NAME, t.curated):
                con.execute(f"DELETE FROM {config.SCHEMA_NAME}.{t.curated} WHERE trans_mode IN ({','.join(repr('cc_' + b) for b in banks)})")

        raw_query = (
            "SELECT trans_id, bank, trans_dt::DATE AS trans_dt, ref_num, trans_details, trans_amt, trans_type FROM cc_df"
        )
        load_data_with_check(con, config.SCHEMA_NAME, t.raw, raw_query, config.DT_COL, mode="append")

        # Column order must match the curated table (inserts are positional)
        curated_query = (
            "SELECT trans_id, trans_dt::DATE AS trans_dt, particulars, trans_mode, trans_amt, trans_type FROM cc_df"
        )
        load_data_with_check(con, config.SCHEMA_NAME, t.curated, curated_query, config.DT_COL, mode="append")
        logger.info("=============== Credit Card Ingestion Complete =============")
    except Exception:
        logger.error("Error in credit card ingestion", exc_info=True)
        raise
    finally:
        if con is not None:
            con.close()
