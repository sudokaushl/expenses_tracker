"""Extract credit card statements (Axis xlsx, ICICI pdf) into normalized CSVs. Runs fully offline."""

from __future__ import annotations

import csv
import os
from glob import glob
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger

logger = get_logger()

OUTPUT_COLUMNS = ["trans_dt", "ref_num", "trans_details", "trans_amt", "trans_type"]


def _to_amount(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def process_axis_cc_raw(x_file: str) -> pd.DataFrame:
    df = pd.read_excel(x_file, sheet_name="Transactions Summary", header=6)
    df = df.rename(columns={
        "Date": "trans_dt",
        "Transaction Details": "trans_details",
        "Amount (INR)": "trans_amt",
        "Debit/Credit": "trans_type",
    })
    df["trans_dt"] = pd.to_datetime(df["trans_dt"], format="%d %b '%y", errors="coerce")
    df = df.dropna(subset=["trans_dt"]).copy()
    df["trans_amt"] = _to_amount(df["trans_amt"]).abs()
    df["trans_type"] = df["trans_type"].astype(str).str.strip().str.title()
    df["ref_num"] = pd.NA
    return df[OUTPUT_COLUMNS]


def process_icici_cc_raw(x_file: str) -> pd.DataFrame:
    import tabula  # needs a local Java runtime; does not use the network

    dfs = tabula.read_pdf(x_file, pages="all", multiple_tables=False, pandas_options={"skiprows": 1})
    if not dfs:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = dfs[0].rename(columns={
        "Date": "trans_dt",
        "Ref. Number": "ref_num",
        "Transaction Details": "trans_details",
        "Amount(in ` )": "trans_amt",
    })
    df["trans_dt"] = pd.to_datetime(df["trans_dt"], format="%d-%b-%y", errors="coerce")
    df = df.dropna(subset=["trans_dt"]).copy()
    # Statement marks credits with a "CR" suffix or a negative sign
    raw_amt = df["trans_amt"].astype(str)
    is_credit = raw_amt.str.contains(r"CR|^\s*-", case=False, regex=True)
    amt = _to_amount(raw_amt.str.replace(r"(?i)\s*CR\s*$", "", regex=True))
    df["trans_amt"] = amt.abs()
    df["trans_type"] = np.where(is_credit | (amt < 0), "Credit", "Debit")
    return df[OUTPUT_COLUMNS]


PARSERS: Dict[Tuple[str, str], Tuple[Callable[[str], pd.DataFrame], str]] = {
    ("creditcard", "axis"): (process_axis_cc_raw, "xlsx"),
    ("creditcard", "icici"): (process_icici_cc_raw, "pdf"),
}


def extract_statements(statement: str, bank: str) -> list[str]:
    """Parse every raw statement for the bank and write one CSV per file; returns output paths."""
    key = (statement, bank)
    if key not in PARSERS:
        raise ValueError(f"Unsupported statement/bank combination: {statement}/{bank}")
    parser, ext = PARSERS[key]

    base = config.landing_dir(statement, bank)
    raw_dir, out_dir = os.path.join(base, "raw"), os.path.join(base, "csv")
    files = sorted(glob(os.path.join(raw_dir, f"*.{ext}")))
    if not files:
        raise FileNotFoundError(f"No .{ext} files found in {raw_dir}")
    os.makedirs(out_dir, exist_ok=True)

    outputs = []
    for path in files:
        name = os.path.basename(path)
        df = parser(path)
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()  # embedded newlines break rows
        out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".csv")
        df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL, doublequote=True)
        logger.info("Extracted %s rows from %s", len(df), name)  # counts only, no transaction data
        outputs.append(out_path)
    return outputs
