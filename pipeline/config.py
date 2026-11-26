# Import libraries
import os
import pandas as pd
import configparser
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import List, Set, TypeVar, Tuple
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# --- General Configurations ---
_delta_dir = os.getenv("delta_dir") or os.getenv("DELTA_DIR")
if not _delta_dir:
    raise RuntimeError(
        f"Missing required 'delta_dir'; set it in {PROJECT_ROOT / '.env'} or the environment."
    )
ROOT_PATH: str = str(
    Path(os.getenv("root_path") or os.getenv("ROOT_PATH") or _delta_dir).expanduser().resolve()
)
DELTA_DIR: str = str(Path(_delta_dir).expanduser().resolve())
    
# --- Job Configurations ---
JOB_NAME: str = "expenses_etl"
LOG_LEVEL: str = "INFO"
RAW_FILE_FORMAT: str = "csv"
PROCESSED_FILE_FORMAT: str = "csv"
RAW_FILE_START_IDX: int = None
RAW_FILE_END_IDX: int = None
RAW_FILE_START_IDENTIFIER = "Tran Date,CHQNO,PARTICULARS,DR,CR,BAL,SOL"
# RAW_FILE_START_IDENTIFIER = None
RAW_FILE_END_IDENTIFIER = "\n"


# --- Pandas Configurations ---
PANDAS_DF = TypeVar("pandas.core.frame.DataFrame")
DEFAULT_MISSING: Set[str] = set(pd._libs.parsers.STR_NA_VALUES) | {" "}


# --- Path Configurations ---
class PATHS:
    LANDING_DIR = {"account": "landing_zone/statements/acc", "creditcard": "landing_zone/statements/cc"}
    PROCESSED_CSV_DIR: str = "processed_zone/statements/acc/axis"
    TEMP_DIR: str = os.path.join(DELTA_DIR, "temp", "expenses_etl")
    SCHEMA_DIR: str = "schema"
    ARTIFACT_DIR: str = str(PROJECT_ROOT / "artifacts")


# --- Database Configurations ---
SCHEMA_NAME: str = "expenses"
DATABASE_PATH: str = os.path.join(DELTA_DIR, PATHS.SCHEMA_DIR, f"{SCHEMA_NAME}.db")
CHROMA_DB_PATH: str = os.path.join(DELTA_DIR, PATHS.SCHEMA_DIR, "chroma_db")

@dataclass(frozen=True)
class StatementTables:
    raw: str
    curated: str
    tokens: str
    normalized: str
    categorized: str
    txn_collection: str
    token_collection: str


STATEMENTS = ("account", "creditcard")
BANKS = {"account": ("axis",), "creditcard": ("axis", "icici")}
_PREFIX = {"account": "acc", "creditcard": "cc"}


def tables(statement: str) -> StatementTables:
    """Table and Chroma collection names for one statement type (acc_* / cc_*)."""
    p = _PREFIX[statement]
    return StatementTables(
        raw=f"{p}_raw",
        curated=f"{p}_curated",
        tokens=f"{p}_tokenized",
        normalized=f"{p}_normalized",
        categorized=f"{p}_categorized",
        txn_collection=f"{p}_transactions",
        token_collection=f"{p}_transaction_tokens",
    )


def landing_dir(statement: str, bank: str) -> str:
    """Landing folder holding the bank's raw/ and csv/ subfolders."""
    return os.path.join(DELTA_DIR, PATHS.LANDING_DIR[statement], bank)


def labelled_table(statement: str) -> str:
    """Label table for one statement type (acc_labelled / cc_labelled)."""
    prefix = _PREFIX[statement]
    return f"{prefix}_labelled"

RAW_COLUMNS: List[str] = ["trans_dt", "cheque_no", "particulars", "debit", "credit", "balance", "sol"]
CURATED_COLUMNS: List[str] = ["trans_id", "trans_dt", "particulars", "trans_type", "trans_mode", "trans_amt", "trans_entity"]
DROP_COLUMNS: List[str] = ["cheque_no", "debit", "credit", "balance", "sol"]
HASH_COLUMN: str = """md5(to_json(struct_pack(
    trans_dt := CAST(trans_dt AS DATE),
    cheque_no := cheque_no,
    particulars := particulars,
    debit := debit,
    credit := credit,
    balance := balance,
    sol := sol
))) AS trans_id"""
DT_COL: str = "trans_dt"
CORPUS_COL: str = "particulars"


# --- Model Configurations ---
NLTK_MODELS: List[str] = [
    "words",
    "punkt",
    "maxent_ne_chunker",
    "averaged_perceptron_tagger"
]

def _read_config_list(section: str) -> List[str]:
    """Read a newline-separated list from local config.ini or its safe template."""
    config_path = PROJECT_ROOT / "config.ini"
    if not config_path.is_file():
        config_path = PROJECT_ROOT / "config.ini.example"

    parser = configparser.ConfigParser()
    loaded = parser.read(config_path, encoding="utf-8")
    if not loaded or not parser.has_option(section, "values"):
        raise RuntimeError(f"Missing [{section}] values in {config_path}")
    return [
        value.strip().lower()
        for line in parser.get(section, "values").splitlines()
        for value in line.split(",")
        if value.strip()
    ]


BANK_NAMES = _read_config_list("BANK_NAMES")
PAYMENT_MODES = _read_config_list("PAYMENT_MODES")
JUNK_WORDS = _read_config_list("JUNK_WORDS")
FILTER_WORDS = JUNK_WORDS + PAYMENT_MODES
TOTAL_NUM_DAYS: int = 365 * 2

# Ingestion mode
INCREMENTAL: str = "incremental"
HISTORICAL: str = "historical"
    

# For Canonicalizer
CN_TRAIN = True
CN_NORMALIZE = True
CN_INCREMENTAL = False
MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.75
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_MIN_MARGIN = 0.5
MIN_TRAINING_SAMPLES = 3  # Lowered to include sparse entities like Rapido, Zerodha
MAX_TRAINING_SAMPLES = 10  # Cap to prevent one category dominating

# Statistical gates for token mapping
ENTROPY_THRESHOLD = 0.8   # Lower = stricter (high entropy = spread across many categories)
DOMINANCE_THRESHOLD = 0.75  # Higher = stricter (low dominance = split between categories)
MIN_TOKEN_COUNT = 2  # Minimum occurrences to consider a token

# Category / subcategory predictor: below these confidences a row is flagged (or sent to the LLM)
CATEGORY_CONF_THRESHOLD = 0.7
SUBCATEGORY_CONF_THRESHOLD = 0.5

ARTIFACT_DIR = PATHS.ARTIFACT_DIR


# For LLM
class LLM_CONFIG:
    LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3")
    LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "ollama")  # Placeholder for local
    if urlparse(LOCAL_LLM_BASE_URL).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError("LOCAL_LLM_BASE_URL must point to localhost; transaction data must not leave this machine.")
    TEMPERATURE = 0.0  # Deterministic for extraction tasks
    MAX_TOKENS = 512
    SIMILARITY_THRESHOLD = 0.8  # For semantic fallback
    ENABLE_FALLBACK = True  # Opt in to local LLM fallback for unmatched entities
