# Technical Documentation: Expenses Tracker Pipeline

## Table of Contents

1. [Overall System Architecture](#1-overall-system-architecture)
2. [Module-by-Module Detailed Analysis](#2-module-by-module-detailed-analysis)
3. [Data Flow Diagram](#3-data-flow-diagram)
4. [Purpose of Each Step](#4-purpose-of-each-step)
5. [Technology Stack Summary](#5-technology-stack-summary)

---

## 1. Overall System Architecture

### Overview

The Expenses Tracker Pipeline is an end-to-end data processing system that transforms raw bank statement CSV files into semantic vector embeddings suitable for similarity search, clustering, and analytics. The system processes banking transaction data through five distinct stages: ingestion, curation, tokenization, canonicalization, and vector embedding generation.

> **Statement types.** Every stage takes `--statement account|creditcard` and reads/writes its own tables. The diagrams and tables below show the account flow with the original `expenses_*` names; read them as `acc_*` (for example `expenses_raw` is now `acc_raw`). Credit card data follows the same stages with `cc_*` tables, preceded by an `extract` step that converts Axis xlsx / ICICI pdf statements to CSV (`pipeline/statement_parsers.py`) and a dedicated ingest (`pipeline/process_cc_data.py`). Category prediction uses `acc_labelled` for account data and `cc_labelled` for credit-card data. Canonicalization artifacts remain shared and are trained from account labels.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        RAW DATA SOURCES                                             │
│                        (Axis Bank CSV Statements in landing_zone/)                                  │
└───────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: PROCESS INPUT DATA (process_input_data.py)                                                │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐                                   │
│  │  skip_lines()   │→ │ TransactionParser│→ │ load_raw_data()    │                                   │
│  │  - Extracts     │  │  - Extracts      │  │  - Registers DF    │                                   │
│  │    table from   │  │    trans_mode    │  │    to DuckDB       │                                   │
│  │    raw CSV      │  │    from          │  │  - Creates         │                                   │
│  │                 │  │    particulars   │  │    expenses_raw    │                                   │
│  └─────────────────┘  └──────────────────┘  └─────────────────────┘                                   │
│                                                                                                       │
│  Output: DuckDB table → expenses_raw (raw columns: trans_dt, cheque_no, particulars, etc.)          │
└───────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: CURATION (process_input_data.py: load_curated_data())                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐                   │
│  │  - Merge debit/credit columns into trans_amt                                 │                   │
│  │  - Extract transaction type (debit/credit)                                    │                   │
│  │  - Parse transaction mode from particulars                                    │                   │
│  └──────────────────────────────────────────────────────────────────────────────┘                   │
│  Output: DuckDB table → expenses_curated (columns: trans_id, trans_dt, particulars,                 │
│          trans_type, trans_mode, trans_amt, trans_entity)                                           │
└───────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: TOKENIZATION (tokenization.py)                                                             │
│  ┌─────────────────┐  ┌──────────────────────┐  ┌─────────────────────────┐                          │
│  │ get_source_data│→ │ normalize_tokens_   │→ │ generate_tokens_with_  │                          │
│  │  - Fetches     │  │ fuzzy()             │  │ tags()                 │                          │
│  │    from curated│  │  - Bank name        │  │  - Regex tokenization  │                          │
│  │    table       │  │    normalization    │  │  - Fuzzy filtering     │                          │
│  │                │  │  - Prefix/substring │  │  - NLTK tokenization   │                          │
│  │                │  │    matching         │  │  - Deduplication       │                          │
│  └─────────────────┘  └──────────────────────┘  └─────────────────────────┘                          │
│  Output: DuckDB table → expenses_tokenized (adds: tokenized column with normalized tokens)          │
└───────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: CANONICALIZATION (canonicalizer.py)                                                        │
│  ┌─────────────────┐  ┌──────────────────────┐  ┌─────────────────────────┐                          │
│  │ load_training_  │→ │ build_token_mapping │→ │ normalize_new_data()   │                          │
│  │ data()         │  │  - Learns token→    │  │  - Semantic similarity │                          │
│  │  - Loads       │  │    canonical        │  │    using Sentence-     │                          │
│  │    labeled     │  │    mappings from    │  │    Transformers        │                          │
│  │    data        │  │    training data    │  │  - Produces normalized │                          │
│  │                │  │  - Incremental       │  │    trans_src           │                          │
│  │                │  │    updates          │  │                        │                          │
│  └─────────────────┘  └──────────────────────┘  └─────────────────────────┘                          │
│  Output: DuckDB table → expenses_normalized (adds: normalized_trans_src)                            │
│          Artifacts: artifacts/v{N}/token_to_canonical_supervised.json                               │
└───────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: VECTOR EMBEDDING (vector_embedding.py)                                                    │
│  ┌─────────────────┐  ┌──────────────────────┐  ┌─────────────────────────┐                          │
│  │ load_tokenized_ │→ │ create_transaction_  │→ │ create_token_embeddings│                          │
│  │ data()         │  │ embeddings()         │  │  - Batch encoding      │                          │
│  │  - Loads from  │  │  - Transaction-level│  │  - Token-level         │                          │
│  │    tokenized   │  │    embeddings        │  │    embeddings          │                          │
│  │    table       │  │  - ChromaDB storage │  │  - ChromaDB storage    │                          │
│  └─────────────────┘  └──────────────────────┘  └─────────────────────────┘                          │
│  Output: ChromaDB Collections → expenses_transactions, expenses_transaction_tokens                  │
└───────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        FINAL OUTPUT                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  ChromaDB Vector Store                                                                  │  │
│  │  - expenses_transactions: 384-dim embeddings per transaction (all-MiniLM-L6-v2)        │  │
│  │  - expenses_transaction_tokens: 384-dim embeddings per token                           │  │
│  │  - Metadata: trans_mode, trans_type, trans_amt, trans_id                               │  │
│  └───────────────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

| Stage | Input | Output | Storage |
|-------|-------|--------|---------|
| 1. Process Input | Raw CSV files | Raw DataFrame | DuckDB: `acc_raw` / `cc_raw` |
| 2. Curation | `acc_raw` / `cc_raw` | Curated DataFrame | DuckDB: `acc_curated` / `cc_curated` |
| 3. Tokenization | `*_curated` | Tokenized DataFrame | DuckDB: `acc_tokenized` / `cc_tokenized` |
| 4. Canonicalization | `*_tokenized` | Normalized DataFrame | DuckDB: `acc_normalized` / `cc_normalized` + Artifacts |
| 5. Vector Embedding | `*_tokenized` | Vector Embeddings | ChromaDB |

---

## 2. Module-by-Module Detailed Analysis

### 2.1 config.py

**File Location:** `pipeline/config.py`

#### Purpose

Central configuration module that defines all system-wide constants, paths, database schemas, and classification lists. This module serves as the single source of truth for all configuration values, enabling consistent settings across all pipeline stages.

#### Architecture

The module defines configuration through multiple approaches:

1. **Environment Variables** (loaded via `dotenv`)
   - `ROOT_PATH`: Base working directory
   - `DELTA_DIR`: Delta lake storage root
   - `LOCAL_LLM_BASE_URL`: Local LLM endpoint
   - `LOCAL_LLM_API_KEY`: API key placeholder

2. **Static Constants** (defined directly)
   - Job configuration (`JOB_NAME`, `LOG_LEVEL`)
   - File formats and identifiers
   - Processing dates and thresholds

3. **Classes** (structured configuration)
   - `PATHS`: Directory paths (`LANDING_DIR` per statement type), processed, temp, schema, artifacts
   - `tables(statement)`: Table and Chroma collection names for `account` (`acc_*`) or `creditcard` (`cc_*`)
   - `LLM_CONFIG`: LLM connection parameters

4. **Classification Lists** (domain-specific)
   - `PAYMENT_MODES`: 22 payment method identifiers
   - `BANK_NAMES`: 50+ bank and financial service names
   - `JUNK_WORDS`: Words to filter from tokens
   - `FILTER_WORDS`: Combined junk + payment modes

#### Key Data Structures

```python
# Path configuration class
class PATHS:
    LANDING_DIR = {"account": "landing_zone/statements/acc", "creditcard": "landing_zone/statements/cc"}
    PROCESSED_CSV_DIR: str = "processed_zone/statements/acc/axis"
    TEMP_DIR: str = "/temp/expenses_etl"
    SCHEMA_DIR: str = "schema"
    ARTIFACT_DIR: str = "artifacts"

# Table names per statement type
def tables(statement: str) -> StatementTables:
    # account    -> acc_raw, acc_curated, acc_tokenized, acc_normalized, acc_categorized
    # creditcard -> cc_raw,  cc_curated,  cc_tokenized,  cc_normalized,  cc_categorized
    ...
# Line 52-57: Column specifications
RAW_COLUMNS: List[str] = ["trans_dt", "cheque_no", "particulars", "debit", "credit", "balance", "sol"]
CURATED_COLUMNS: List[str] = ["trans_id", "trans_dt", "particulars", "trans_type", "trans_mode", "trans_amt", "trans_entity"]
```

#### Techniques Used

- **Environment-based Configuration**: Uses `python-dotenv` for runtime configuration
- **Type Safety**: Uses `TypeVar` for DataFrame typing
- **Lazy Loading**: Configuration accessed at module import time

#### Tools/Libraries Used

- `os`: Path manipulation
- `pandas`: DataFrame type definitions
- `typing`: Type annotations
- `dotenv`: Environment variable loading

#### Key Functions/Constants and Signatures

| Item | Type | Location | Description |
|------|------|-----------|-------------|
| `ROOT_PATH` | str | :line:`10` | Base directory from environment |
| `DELTA_DIR` | str | :line:`11` | Delta storage root from environment |
| `JOB_NAME` | str | :line:`14` | ETL job identifier |
| `PATHS` | class | :line:`32` | Directory path constants |
| `DATABASE_PATH` | str | :line:`42` | DuckDB database file path |
| `CHROMA_DB_PATH` | str | :line:`43` | ChromaDB storage directory |
| `PAYMENT_MODES` | List[str] | :line:`68-71` | 22 payment mode identifiers |
| `BANK_NAMES` | List[str] | :line:`73-81` | 50+ bank name patterns |
| `JUNK_WORDS` | List[str] | :line:`84-87` | Words to filter |
| `FILTER_WORDS` | List[str] | :line:`88` | Combined filter list |
| `MODEL_NAME` | str | :line:`100` | SentenceTransformer model |
| `SIMILARITY_THRESHOLD` | float | :line:`101` | Semantic similarity cutoff |
| `LLM_CONFIG` | class | :line:`106-111` | LLM endpoint configuration |

---

### 2.2 utils.py

**File Location:** `pipeline/utils.py`

#### Purpose

Provides cross-cutting utility functions for logging, file handling, argument parsing, DuckDB database operations, and NLTK model configuration. This module eliminates code duplication across pipeline stages.

#### Architecture

The module is organized into five functional groups:

1. **Argument Parsing** (lines 21-69)
   - `BaseArgumentParser`: Reusable CLI argument handler

2. **Logging** (lines 74-142)
   - `LoggerFactory`: Singleton logger factory
   - `get_log_file_path()`: Generates timestamped log paths

3. **File Utilities** (lines 145-175)
   - `list_files_by_extension()`: Glob-based file discovery
   - `read_yaml()`: YAML file parsing
   - `skip_lines()`: Table extraction from raw files
   - `string_to_string_io()`: String to StringIO conversion

4. **DuckDB Utilities** (lines 248-426)
   - `create_duckdb_connection()`: Database connection factory
   - `get_max_date()`: Date column aggregation
   - `get_table_data()`: SELECT with optional WHERE clause
   - `load_data_into_table()`: INSERT/CREATE mode data loading
   - `load_data_with_check()`: Incremental loading with date filtering
   - `check_table_exists()`: Table existence check

5. **NLTK Utilities** (lines 428-440)
   - `configure_nltk()`: Model download and setup

#### Key Functions and Signatures

**Logging Functions:**

```python
# Line 75-92
def get_log_file_path(job_name: Optional[str] = None) -> str:
    """Generates a log file path with a timestamp and optional job name."""

# Line 101-123
def create_logger(log_file: str, log_level: str) -> logging.Logger:
    """Creates a logger with specified log file and level."""
```

**File Utilities:**

```python
# Line 145-158
def list_files_by_extension(dir_name: str, file_extension: str) -> List[str]:
    """Lists all files with the specified extension in the given directory."""

# Line 179-231
def skip_lines(
    file_path: str,
    start_identifier: str = None,
    end_identifier: str = None,
    start_index: int = None,
    end_index: int = None
) -> str:
    """Skips lines in a file based on the occurrence of newline characters."""
    # Implementation details:
    # - Reads file, strips whitespace from each line
    # - Uses start_identifier (e.g., "TranDate,CHQNO,...") to find table start
    # - Uses end_identifier (e.g., "\n") to find table end
    # - Returns lines between start and end indices

# Line 233-244
def string_to_string_io(text: str) -> StringIO:
    """Converts a string to a StringIO object."""
```

**DuckDB Utilities:**

```python
# Line 248-263
def create_duckdb_connection(database_path: str) -> DuckDBPyConnection:
    """Creates a DuckDB connection to the specified database."""

# Line 265-288
def get_max_date(cursor: DuckDBPyConnection, schema_name: str, table_name: str, dt_col: str) -> str:
    """Retrieves the maximum date from a specified column."""

# Line 290-316
def get_table_data(cursor: DuckDBPyConnection, schema_name: str, table_name: str,
                   cols: List[str] = None, where_cond: str = None) -> config.PANDAS_DF:
    """Retrieves data from a DuckDB table with optional filtering."""

# Line 318-381
def load_data_into_table(
    cursor: DuckDBPyConnection,
    schema_name: str,
    table_name: str,
    source_format: str = None,  # "pandas", "csv", "sql"
    data_dir: str = None,
    file_delimiter: str = None,
    header: bool = None,
    source_df: config.PANDAS_DF = None,
    query: str = None,
    source_cols: List[str] = None,
    mode: str = "create"  # "create", "append", "overwrite"
) -> None:
    """Loads data from various sources into DuckDB table."""

# Line 383-396
def load_data_with_check(cursor, schema_name, table_name, query, dt_col=None, mode=None):
    """Smart loading with incremental mode detection."""

# Line 399-425
def check_table_exists(cursor: DuckDBPyConnection, schema_name: str, table_name: str) -> bool:
    """Check if table exists in schema."""
```

**NLTK Utilities:**

```python
# Line 428-439
def configure_nltk(nltk_models: List[str] = None) -> None:
    """Downloads and configures NLTK models."""
```

#### Input/Output Specifications

| Function | Input | Output |
|----------|-------|--------|
| `get_log_file_path` | `job_name` (optional) | str (log file path) |
| `create_logger` | log_file, log_level | Logger object |
| `list_files_by_extension` | dir_name, file_extension | List[str] (file paths) |
| `read_yaml` | file_path | dict |
| `skip_lines` | file_path, identifiers/indices | List[str] (lines) |
| `string_to_string_io` | str | StringIO |
| `create_duckdb_connection` | database_path | DuckDBPyConnection |
| `get_max_date` | cursor, schema, table, dt_col | str (max date) or None |
| `get_table_data` | cursor, schema, table, cols, where_cond | pd.DataFrame |
| `load_data_into_table` | cursor, schema, table, source_config | None (side effect) |
| `configure_nltk` | nltk_models List | None (side effect) |

#### Techniques Used

- **Singleton Pattern**: `LoggerFactory` maintains class-level `_log` instance
- **Regex-based Table Extraction**: Parses raw bank statement files to isolate data table
- **DuckDB SQL Generation**: Dynamic query building for flexible data loading
- **Batch Processing**: File discovery via glob patterns

#### Tools/Libraries Used

- `argparse`: CLI argument parsing
- `logging`: Application logging
- `duckdb`: Embedded analytical database
- `nltk`: Natural language toolkit
- `glob`: File pattern matching
- `yaml`: Configuration file parsing

---

### 2.3 process_input_data.py

**File Location:** `pipeline/process_input_data.py`

#### Purpose

Ingests raw bank statement CSV files, transforms them into structured transaction records, and loads them into DuckDB. This module handles the initial data extraction and curation, converting messy bank statements into clean, queryable data.

#### Architecture

**Main Components:**

1. **TransactionParser Class** (lines 13-83)
   - Parses transaction "particulars" field to extract transaction mode
   - Uses rule-based parsing with separator patterns

2. **Data Loading Functions** (lines 87-220)
   - `get_raw_data_to_df()`: Reads and processes raw CSV files
   - `load_raw_data()`: Loads raw data into DuckDB
   - `load_curated_data()`: Transforms and loads curated data

3. **Main Entry Point** (lines 224-316)
   - Orchestrates the full ingestion pipeline
   - Handles incremental vs. historical load modes

#### TransactionParser Class Details

```python
# Line 13-83
class TransactionParser:
    """Parse transaction particulars to extract mode, entity, and entity ID."""

    # Define parsing rules: (trans_modes, separator, part_index)

    sep1_idx1 = ['ACH', 'ATM', 'BRN', 'NACH']
    sep2_idx1 = ['AUTOBPAY', 'POS', 'MBBPAY']
    sep2_idx2 = ['CSHBCK', 'ICONNREF', 'MOB', 'NBSM', 'INB', 'NEFT']
    sep2_idx3 = ['IMPS', 'RTGS', 'UPI']
    sep2_idx4 = ['CLG', 'ECS', 'IFT']

    PARSING_RULES = [
        (sep1_idx1, '-', 0),      # ACH-123-456 → mode=ach
        (sep2_idx1, '/', 0),      # AUTOBPAY/123/456 → mode=autobpay
        (sep2_idx2, '/', 0),
        (sep2_idx3, '/', 0),
        (sep2_idx4, '/', 0),
    ]

    MIN_SEPARATORS = {'-': 2, '/': 2}
```

**Parsing Logic:**
- Iterates through `PARSING_RULES` to find matching pattern
- Checks separator count (`min_sep_count`)
- Checks if transaction starts with known mode prefix
- Extracts transaction mode from first segment
- Returns `pd.Series` with `trans_mode` field

#### Data Processing Pipeline

```python
# Line 87-146: get_raw_data_to_df()
def get_raw_data_to_df(
    raw_dir: str,
    columns: List[str] = None,
    dt_col: str = None,
    start_index: int = None,
    end_index: int = None,
    start_identifier: str = None,
    end_identifier: str = None,
    na_values: Set[str] = None,
):
```

**Processing Steps:**
1. List all CSV files in `raw_dir` (config.py:34 - `RAW_CSV_DIR`)
2. For each CSV file:
   - Extract table using `skip_lines()` with start/end identifiers
   - Convert to StringIO via `string_to_string_io()`
   - Parse CSV with `pd.read_csv()` using specified columns and date format
   - Convert date column to datetime
3. Concatenate all DataFrames
4. Return combined DataFrame

**Date Parsing:** Uses `dayfirst=True` for Indian date format (DD-MM-YYYY)

```python
# Line 147-174: load_raw_data()
def load_raw_data(
    cursor: DuckDBPyConnection,
    raw_df: pd.DataFrame,
    schema_name: str = None,
    table_name: str = None,
    columns: List[str] = None,
    dt_col: str = None,
    mode: str = "append",
    ingest_mode: str = None
):
```

**Key Logic:**
- Registers pandas DataFrame as DuckDB view (`raw_df`)
- Constructs SELECT query with date casting: `col::DATE AS dt_col`
- Uses `load_data_with_check()` for smart incremental loading
- Hash column for deduplication: `md5(concat(trans_dt::DATE::STRING, particulars::STRING))`

```python
# Line 177-220: load_curated_data()
def load_curated_data(cursor: DuckDBPyConnection,
                    schema_name:str,
                    raw_table:str,
                    curated_table: str,
                    dt_col:str,
                    mode:str = "append",
                    ingest_mode:str = None
):
```

**Curated Data Transformation:**
1. Reads from `expenses_raw` table
2. Applies `TransactionParser.extract()` to extract transaction mode
3. Merges debit/credit columns:
   ```sql
   CASE WHEN debit IS NULL THEN credit ELSE debit END AS trans_amt
   CASE WHEN debit IS NULL THEN 'credit' ELSE 'debit' END AS trans_type
   ```
4. Loads into `expenses_curated` with curated columns

#### Input/Output Specifications

| Function | Input | Output |
|----------|-------|--------|
| `TransactionParser.extract()` | particulars (str) | pd.Series (trans_mode) |
| `get_raw_data_to_df()` | raw_dir, columns, dt_col, identifiers | pd.DataFrame |
| `load_raw_data()` | cursor, raw_df, schema, table | DuckDB table: `expenses_raw` |
| `load_curated_data()` | cursor, schema, raw_table, curated_table | DuckDB table: `expenses_curated` |

#### Ingestion Modes

The module supports two ingestion modes controlled by CLI argument `--ingest_mode`:

1. **Incremental Mode** (`incremental`/`incr`)
   - Loads only new records after max date in target table
   - Uses `mode="append"`

2. **Historical Mode** (`historical`/`hist`)
   - Full reload of all data
   - Uses `mode="overwrite"`

#### Techniques Used

- **Rule-based Parsing**: Separator-based transaction mode extraction
- **Dynamic Column Casting**: SQL-level date type conversion
- **Incremental Loading**: Date-based filtering for efficiency
- **DataFrame Concatenation**: Multiple file merging

#### Tools/Libraries Used

- `pandas`: DataFrame manipulation, CSV parsing, date conversion
- `duckdb`: SQL database operations
- `re`: Regular expression for line processing
- `os`: File path operations
- `config`: Central configuration (imported)

---

### 2.4 tokenization.py

**File Location:** `pipeline/tokenization.py`

#### Purpose

Extracts meaningful tokens from transaction particulars, normalizes bank names, and filters junk words. This module transforms raw text into clean, normalized token lists suitable for downstream processing like canonicalization and embedding generation.

#### Architecture

**Main Components:**

1. **Data Loading** (lines 18-55)
   - `get_source_data()`: Fetches data from DuckDB with incremental support

2. **Token Processing** (lines 57-171)
   - `get_tokens()`: Regex-based token extraction
   - `filter_tokens_fuzzy()`: Fuzzy filtering using RapidFuzz
   - `normalize_tokens_fuzzy()`: Bank name normalization
   - `generate_tokens_with_tags()`: Full tokenization pipeline

3. **Data Loading** (lines 173-195)
   - `load_tokenized_data()`: Saves tokenized data to DuckDB

4. **Main Entry Point** (lines 200-281)
   - Orchestrates tokenization pipeline

#### Token Extraction Functions

```python
# Line 57-65
def get_tokens(row, pattern, corpus_col):
    """Extract tokens from corpus column using regex pattern."""
    # Returns: list of words with length >= 3
    return [word for word in re.findall(pattern, str(row[corpus_col]).lower()) if len(word) >= 3]
```

**Pattern Used:** `r'[A-Za-z]+'` - extracts alphabetic tokens

#### Fuzzy Filtering

```python
# Line 67-77
def filter_tokens_fuzzy(word: str, word_list: list[str], threshold: int = 80) -> bool:
    """Return True if `word` is similar to any word in `word_list` above threshold."""
    # Uses RapidFuzz for efficient fuzzy matching
    match = process.extractOne(word, word_list, scorer=fuzz.ratio)
    if match is None:
        return False
    _, score, _ = match
    return score >= threshold
```

**Purpose:** Filters out words similar to junk/filter words (e.g., "paytm" → filtered if in FILTER_WORDS)

#### Bank Name Normalization

```python
# Line 79-140
def normalize_tokens_fuzzy(tokens: list[str], bank_names: list[str], threshold: int = 85) -> list[str]:
    """Normalize extracted tokens to canonical bank names."""
```

**Normalization Strategy:**

1. **Prefix Matching** (lines 92-101)
   - Checks if token is prefix of bank name
   - Example: "stateba" → "statebankofindia" (requires len >= 5)

2. **Substring Matching** (lines 102-107)
   - Checks if bank name is contained in token
   - Example: "hdfc" in "hdfcbankltd" → "hdfc"

3. **Fuzzy Matching** (lines 112-138)
   - For tokens >= 5 characters
   - Uses RapidFuzz with length ratio adjustment
   - Threshold: 85% similarity
   - Minimum length ratio: 0.4 for short tokens (5-8 chars), 0.6 for longer (9+ chars)

**Example:**
```
Input: ["hdfcbankltd", "rtgs", "statebankofindia"]
Bank Names: ["hdfc", "statebankofindia", "yesbank"]
Output: ["hdfc", "rtgs", "statebankofindia"]
```

#### Full Tokenization Pipeline

```python
# Line 144-171
def generate_tokens_with_tags(df, pattern, corpus_col, filter_words=None, bank_threshold=90, filter_threshold=85):
    """Complete tokenization pipeline with normalization and filtering."""
```

**Pipeline Steps:**

1. **Regex Tokenization** (line 153)
   - `text_series.str.lower().apply(lambda x: re.findall(pattern, x))`

2. **Bank Name Normalization** (lines 156-158)
   - Apply `normalize_tokens_fuzzy()` with bank names

3. **Fuzzy Filtering** (lines 161-163)
   - Remove tokens matching filter words (Fuzzy, threshold=85)

4. **NLTK Tokenization** (line 166)
   - Re-tokenize with NLTK for proper word boundaries

5. **Deduplication** (line 169)
   - Preserve order using `dict.fromkeys()`

```python
df["tokenized"] = tokenized_lists.apply(lambda t: list(dict.fromkeys(t)))
```

#### Data Loading

```python
# Line 173-195
def load_tokenized_data(cursor: DuckDBPyConnection,
                    tokenized_df:str,
                    schema_name:str,
                    table_name: str,
                    dt_col:str,
                    ingest_mode:str = None,
                    mode: str = "append"
):
```

- Registers DataFrame as `tokenized_df` in DuckDB
- Uses `load_data_with_check()` for incremental support

#### Input/Output Specifications

| Function | Input | Output |
|----------|-------|--------|
| `get_source_data()` | cursor, schema, src_table, tgt_table, ingest_mode, dt_col | pd.DataFrame |
| `get_tokens()` | row, pattern, corpus_col | List[str] |
| `filter_tokens_fuzzy()` | word, word_list, threshold | bool |
| `normalize_tokens_fuzzy()` | tokens, bank_names, threshold | List[str] |
| `generate_tokens_with_tags()` | df, pattern, corpus_col, filter_words | pd.DataFrame (with tokenized column) |
| `load_tokenized_data()` | cursor, tokenized_df, schema, table, dt_col | DuckDB table: `expenses_tokenized` |

#### Techniques Used

- **Multi-stage Normalization**: Rule-based → Prefix/Substring → Fuzzy
- **Length-adaptive Thresholds**: Different fuzzy thresholds based on token length
- **Ordered Deduplication**: Preserves first occurrence order
- **RapidFuzz**: Fast fuzzy string matching (faster than fuzzywuzzy)

#### Tools/Libraries Used

- `re`: Regular expression token extraction
- `nltk`: Word tokenization
- `rapidfuzz`: Fast fuzzy matching (fuzz.ratio, process.extractOne)
- `pandas`: DataFrame operations
- `duckdb`: Database operations
- `sentence_transformers`: Loaded but not actively used in this module
- `sklearn`: cosine_similarity imported but not used

---

### 2.5 canonicalizer.py

**File Location:** `pipeline/canonicalizer.py`

#### Purpose

Learns supervised token-to-canonical mappings from labeled transaction data and applies semantic normalization to new transactions. Uses sentence embeddings for fallback matching when exact mappings don't exist. Maintains versioned artifacts for reproducibility.

#### Architecture

**Main Components:**

1. **Data Loading** (lines 27-49)
   - `load_training_data()`: Loads canonicalization labels from `acc_labelled` table
   - `load_new_data()`: Loads unlabeled data from `expenses_tokenized`

2. **Token Extraction** (lines 55-62)
   - `extract_tokens()`: Converts tokenized field to clean lowercase tokens

3. **Mapping Building** (lines 68-85)
   - `build_token_mapping()`: Learns token→canonical mappings from training data

4. **Embedding Functions** (lines 91-94)
   - `build_source_embeddings()`: Encodes canonical source names

5. **Normalization Functions** (lines 100-123)
   - `normalize_token()`: Single token normalization
   - `normalize_transaction()`: Full transaction normalization

6. **Artifact Management** (lines 129-165)
   - `get_next_version()`: Determines next version folder
   - `save_versioned_artifacts()`: Saves mapping and embeddings

7. **Pipeline Functions** (lines 171-248)
   - `train_token_canonicalizer()`: Training pipeline
   - `normalize_new_data()`: Inference pipeline
   - `run_pipeline()`: Main orchestrator

#### Training Pipeline

```python
# Line 171-196
def train_token_canonicalizer(incremental=True):
    """Trains the token canonicalizer from labeled data."""
```

**Training Steps:**

1. Load account-labeled data from `acc_labelled` table
2. Extract tokens from `tokenized` column
3. If incremental mode and artifacts exist, load previous mappings
4. Build token→canonical mapping:
   ```python
   for each (token, trans_src) pair in training data:
       count[token][trans_src]++
   for each token:
       canonical = most_common(trans_src)
   ```
5. Build source embeddings for canonical names
6. Save versioned artifacts

#### Normalization Pipeline

```python
# Line 198-236
def normalize_new_data(output_to="duckdb"):
    """Applies canonicalization to new unlabeled data."""
```

**Normalization Steps:**

1. Load new data from the selected statement's tokenized table
2. Load latest artifacts (mapping + embeddings)
3. Retrieve lexical and semantic source candidates, then fuse their rankings with reciprocal-rank fusion
4. Prefer a unique whole-phrase source match over a shared token match
5. If candidate rankings conflict or semantic scores are close, rerank the candidates with the configured cross-encoder; if it is unavailable or inconclusive, leave the source unmatched
6. Save to the selected statement's normalized DuckDB table or CSV

The cross-encoder (`CROSS_ENCODER_MODEL`) is loaded only when an ambiguous transaction needs reranking. Sentence Transformers may download that model the first time; if loading fails, ambiguous matches are left unresolved rather than guessed.

#### Token Normalization Details

```python
# Line 100-113
def normalize_token(token, model, token_to_canonical, src_names, src_embeddings,
                    similarity_threshold=config.SIMILARITY_THRESHOLD):
    """Normalize a single token."""
    if not token:
        return None
    token = str(token).lower().strip()

    # Step 1: Exact match in mapping
    if token in token_to_canonical:
        return token_to_canonical[token]

    # Step 2: Semantic similarity fallback
    vec = model.encode([token], convert_to_numpy=True, normalize_embeddings=True)
    sims = cosine_similarity(vec, src_embeddings)[0]
    best_idx = np.argmax(sims)
    best_sim = sims[best_idx]
    return src_names[best_idx] if best_sim >= similarity_threshold else None
```

#### Artifact Versioning

```python
# Line 129-165
def get_next_version(base_dir=config.ARTIFACT_DIR):
    """Determine next artifact version based on existing folders."""
    # Scans artifacts/ directory for v1, v2, etc.
    # Returns next version string (e.g., "v3")

def save_versioned_artifacts(version, token_to_canonical, src_names, src_embeddings):
    """Save artifacts under artifacts/<version>/"""
    # Creates: artifacts/v{N}/
    #   - token_to_canonical_supervised.json
    #   - src_names.npy
    #   - src_embeddings.npy
    # Creates symlinks: latest_*
```

**Versioning Strategy:**
- Version folders: `artifacts/v1/`, `artifacts/v2/`, etc.
- Latest symlinks: `artifacts/latest_token_to_canonical.json`, etc.
- Enables rollbacks and reproducibility

#### Input/Output Specifications

| Function | Input | Output |
|----------|-------|--------|
| `load_training_data()` | limit (optional) | pd.DataFrame (tokenized, trans_src) |
| `load_new_data()` | limit (optional) | pd.DataFrame (all columns) |
| `extract_tokens()` | tokenized field | List[str] |
| `build_token_mapping()` | train_df, existing_map | dict (token→canonical) |
| `build_source_embeddings()` | model, trans_src_list | (np.array, list) |
| `normalize_token()` | token, model, mapping, embeddings | str or None |
| `normalize_transaction()` | tokens, model, mapping, embeddings | str or None |
| `train_token_canonicalizer()` | incremental flag | Artifacts (JSON + NumPy) |
| `normalize_new_data()` | output_to | DuckDB: `expenses_normalized` or CSV |

#### Configuration

```python
# config.py lines 96-102
CN_TRAIN = True
CN_NORMALIZE = True
CN_INCREMENTAL = False
MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.9
ARTIFACT_DIR = PATHS.ARTIFACT_DIR  # "artifacts"
```

#### Techniques Used

- **Supervised Learning**: Token→canonical from labeled examples
- **Incremental Updates**: Load previous mappings, merge with new data
- **Semantic Similarity**: SentenceTransformer embeddings for fallback
- **Artifact versioning**: Artifacts stored in versioned directories with latest pointers; local generated files are excluded from Git
- **Most Common Selection**: Canonical = mode of training labels

#### Tools/Libraries Used

- `sentence_transformers`: all-MiniLM-L6-v2 model for embeddings
- `sklearn.metrics.pairwise`: cosine_similarity computation
- `numpy`: Array operations
- `pandas`: DataFrame manipulation
- `json`: Artifact serialization
- `duckdb`: Database operations

---

### 2.6 vector_embedding.py

**File Location:** `pipeline/vector_embedding.py`

#### Purpose

Generates dense vector embeddings for both transaction-level and token-level data, storing them in ChromaDB for efficient similarity search. This is the final stage that enables semantic queries on the transaction corpus.

#### Architecture

**Main Components:**

1. **Data Loading** (lines 11-23)
   - `load_tokenized_data()`: Fetches tokenized transactions from DuckDB

2. **Embedding Generation** (lines 26-132)
   - `create_transaction_embeddings()`: Transaction-level embeddings
   - `create_token_embeddings()`: Token-level embeddings

3. **Main Pipeline** (lines 135-171)
   - Orchestrates embedding creation and storage

#### Transaction-Level Embeddings

```python
# Line 26-75
def create_transaction_embeddings(model, df: pd.DataFrame, corpus_col: str,
                                  txn_collection, batch_size: int = 64):
    """Encode transaction-level embeddings in batches and store them in ChromaDB."""
```

**Process:**

1. Iterate through DataFrame rows
2. Extract tokens from `corpus_col` (e.g., "tokenized")
3. Skip empty token lists
4. Check for duplicate transaction IDs
5. Join tokens to form text: `"token1 token2 token3"`
6. Collect IDs, texts, metadata (trans_mode, trans_type, trans_amt)
7. Batch encode with SentenceTransformer (batch_size=64)
8. Add to ChromaDB collection

**Metadata Structure:**
```python
{
    "trans_mode": row.get("trans_mode") or "Unknown",
    "trans_type": row.get("trans_type"),
    "trans_amt": row.get("trans_amt")
}
```

#### Token-Level Embeddings

```python
# Line 78-132
def create_token_embeddings(model, df: pd.DataFrame, corpus_col: str,
                            token_collection, batch_size: int = 128):
    """Encode token-level embeddings in batches and store them in ChromaDB."""
```

**Process:**

1. Flatten tokens from all transactions
2. Create unique token IDs: `{trans_id}_{token}` (e.g., "20240101_001_hdfc")
3. Check for duplicates, log warnings
4. Collect texts (individual tokens), IDs, metadata (trans_id)
5. Batch encode with SentenceTransformer (batch_size=128)
6. Add to ChromaDB collection

**Token ID Format:** `{trans_id}_{token}` ensures uniqueness across transactions

#### ChromaDB Setup

```python
# Line 140-146
# Setup DuckDB connection
con = create_duckdb_connection(config.DATABASE_PATH)

# Setup ChromaDB client
chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

# Get or create collections
txn_collection = chroma_client.get_or_create_collection("expenses_transactions")
token_collection = chroma_client.get_or_create_collection("expenses_transaction_tokens")
```

**ChromaDB Configuration:**
- Persistent client with path from `config.CHROMA_DB_PATH`
- Two collections:
  - `expenses_transactions`: Transaction-level embeddings (384-dim)
  - `expenses_transaction_tokens`: Token-level embeddings (384-dim)

#### Batch Processing

```python
# Transaction embeddings (batch_size=64)
for i in tqdm(range(0, len(texts), batch_size), desc="Transaction Batches"):
    batch_texts = texts[i:i + batch_size]
    batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
    embeddings.extend(batch_embeddings)

# Token embeddings (batch_size=128)
for i in tqdm(range(0, len(texts), batch_size), desc="Token Batches"):
    batch_texts = texts[i:i + batch_size]
    batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
    embeddings.extend(batch_embeddings)
```

**Batch Encoding Benefits:**
- Memory efficiency for large datasets
- GPU utilization optimization
- Progress tracking via tqdm

#### Input/Output Specifications

| Function | Input | Output |
|----------|-------|--------|
| `load_tokenized_data()` | None | pd.DataFrame from `expenses_tokenized` |
| `create_transaction_embeddings()` | model, df, corpus_col, txn_collection | ChromaDB: `expenses_transactions` |
| `create_token_embeddings()` | model, df, corpus_col, token_collection | ChromaDB: `expenses_transaction_tokens` |

#### Data Stored in ChromaDB

| Collection | ID | Document | Embedding | Metadata |
|------------|-----|----------|-----------|----------|
| `expenses_transactions` | trans_id | "token1 token2..." | 384-dim vector | trans_mode, trans_type, trans_amt |
| `expenses_transaction_tokens` | `{trans_id}_{token}` | token string | 384-dim vector | trans_id |

#### Techniques Used

- **Batch Encoding**: Efficient GPU utilization
- **Persistent Storage**: ChromaDB with filesystem backend
- **Duplicate Prevention**: ID uniqueness checks
- **Progress Tracking**: tqdm for batch progress
- **Error Prevention**: Assert statements for data integrity

#### Tools/Libraries Used

- `sentence_transformers`: all-MiniLM-L6-v2 model
- `chromadb`: Vector database for embeddings
- `tqdm`: Progress bar
- `pandas`: DataFrame operations
- `numpy`: Array operations

---

## 3. Data Flow Diagram

### Detailed Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         STAGE 1: RAW DATA INGESTION                                    │
│                                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  Input: CSV Files (Axis Bank Statements)                                                        │   │
│  │  Location: landing_zone/statements/acc/axisbank/*.csv                                               │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  process_input_data.py:get_raw_data_to_df()                                                    │   │
│  │  ├─ list_files_by_extension(raw_dir, "csv")                                                   │   │
│  │  ├─ For each CSV:                                                                              │   │
│  │  │   └─ skip_lines() → extracts table between START_ID and END_ID                          │   │
│  │  │   └─ string_to_string_io() → converts to StringIO                                        │   │
│  │  │   └─ pd.read_csv() → parses with RAW_COLUMNS, dayfirst=True                              │   │
│  │  │   └─ Convert dt_col to datetime                                                           │   │
│  │  └─ pd.concat() → combines all DataFrames                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  process_input_data.py:load_raw_data()                                                        │   │
│  │  ├─ Register raw_df with DuckDB                                                               │   │
│  │  ├─ Construct SELECT with date casting: trans_dt::DATE                                        │   │
│  │  ├─ Hash column: md5(concat(trans_dt::DATE::STRING, particulars::STRING))                  │   │
│  │  └─ load_data_with_check() → incremental aware loading                                       │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT: DuckDB Table - expenses_raw                                                          │   │
│  │  Schema: trans_id (md5), trans_dt, cheque_no, particulars, debit, credit, balance, sol       │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         STAGE 2: DATA CURATION                                         │
│                                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  Input: expenses_raw table                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  process_input_data.py:load_curated_data()                                                    │   │
│  │  ├─ SELECT * FROM expenses_raw                                                                │   │
│  │  ├─ Apply TransactionParser.extract() to particulars                                         │   │
│  │  │   └─ Pattern matching: separator + mode prefix                                           │   │
│  │  │   └─ Returns: trans_mode                                                                  │   │
│  │  ├─ Merge debit/credit:                                                                       │   │
│  │  │   └─ trans_amt = COALESCE(debit, credit)                                                  │   │
│  │  │   └─ trans_type = CASE WHEN debit IS NULL THEN 'credit' ELSE 'debit'                   │   │
│  │  └─ load_data_with_check() → creates expenses_curated                                        │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT: DuckDB Table - expenses_curated                                                      │   │
│  │  Schema: trans_id, trans_dt, particulars, trans_type, trans_mode, trans_amt, trans_entity   │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         STAGE 3: TOKENIZATION                                          │
│                                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  Input: expenses_curated table                                                                │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  tokenization.py:get_source_data()                                                           │   │
│  │  ├─ If historical mode: SELECT * FROM curated                                                │   │
│  │  └─ If incremental mode: SELECT * WHERE trans_dt > max_dt                                    │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  tokenization.py:generate_tokens_with_tags()                                                 │   │
│  │  Step 1: Regex Tokenization                                                                   │   │
│  │  │   └─ re.findall(r'[A-Za-z]+', text.lower())                                              │   │
│  │  │   └─ Filter: word length >= 3                                                             │   │
│  │  │                                                                                            │   │
│  │  Step 2: Bank Name Normalization                                                              │   │
│  │  │   └─ normalize_tokens_fuzzy(tokens, BANK_NAMES, threshold=90)                            │   │
│  │  │   ├─ Strategy 1: Prefix match (e.g., "stateba" → "statebankofindia")                  │   │
│  │  │   ├─ Strategy 2: Substring match (e.g., "hdfc" in "hdfcbankltd")                       │   │
│  │  │   └─ Strategy 3: Fuzzy match (threshold=85, len_ratio adaptive)                        │   │
│  │  │                                                                                            │   │
│  │  Step 3: Fuzzy Filtering                                                                      │   │
│  │  │   └─ filter_tokens_fuzzy(token, FILTER_WORDS, threshold=85)                              │   │
│  │  │   └─ Removes: payment modes, junk words                                                   │   │
│  │  │                                                                                            │   │
│  │  Step 4: NLTK Tokenization                                                                   │   │
│  │  │   └─ nltk.word_tokenize(" ".join(tokens))                                                │   │
│  │  │                                                                                            │   │
│  │  Step 5: Deduplication                                                                       │   │
│  │  │   └─ list(dict.fromkeys(tokens)) → preserves order                                       │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  tokenization.py:load_tokenized_data()                                                       │   │
│  │  └─ Register tokenized_df, load_data_with_check()                                            │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT: DuckDB Table - expenses_tokenized                                                    │   │
│  │  Schema: + tokenized (List[str]) column                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         STAGE 4: CANONICALIZATION                                     │
│                                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  Input: expenses_tokenized table + acc_labelled (labeled)                                 │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                         │
│                          ┌───────────────────┴───────────────────┐                                   │
│                          ▼                                       ▼                                   │
│  ┌────────────────────────────────────────────────┐  ┌───────────────────────────────────────────┐  │
│  │  TRAIN MODE (train=True)                       │  │  NORMALIZE MODE (normalize=True)         │  │
│  │                                                │  │                                           │  │
│  │  load_training_data()                          │  │  load_new_data()                          │  │
│  │  └─ SELECT tokenized, trans_src                │  │  └─ SELECT * FROM expenses_tokenized      │  │
│  │     FROM acc_labelled                       │  │                                           │  │
│  │                                                │  │  normalize_transaction()                 │  │
│  │  build_token_mapping()                        │  │  ├─ For each token:                      │  │
│  │  ├─ Count(token, trans_src)                   │  │  │   └─ Exact match → mapping[token]      │  │
│  │  ├─ Canonical = mode(trans_src)                │  │  │   └─ Else → semantic similarity        │  │
│  │  └─ If incremental: merge with existing       │  │  │       └─ cosine_similarity            │  │
│  │                                                │  │  │       └─ threshold check (0.9)        │  │
│  │  build_source_embeddings()                   │  │  └─ Most common normalized value         │  │
│  │  └─ model.encode(unique trans_src)             │  │                                           │  │
│  │                                                │  │  Output: normalized_trans_src           │  │
│  │  save_versioned_artifacts()                    │  │                                           │  │
│  │  └─ artifacts/v{N}/                           │  │  Save to DuckDB / CSV                    │  │
│  │      ├─ token_to_canonical_supervised.json   │  │                                           │  │
│  │      ├─ src_names.npy                         │  │                                           │  │
│  │      └─ src_embeddings.npy                    │  │                                           │  │
│  └────────────────────────────────────────────────┘  └───────────────────────────────────────────┘  │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT: DuckDB Table - expenses_normalized + Artifacts                                       │   │
│  │  Schema: + normalized_trans_src column                                                       │   │
│  │  Artifacts: artifacts/v{N}/ (token mapping, embeddings)                                      │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         STAGE 5: VECTOR EMBEDDING                                      │
│                                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  Input: expenses_tokenized table                                                              │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  vector_embedding.py:load_tokenized_data()                                                   │   │
│  │  └─ SELECT * FROM expenses_tokenized WHERE trans_id IS NOT NULL                             │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────┐  ┌─────────────────────────────────────────┐   │
│  │  create_transaction_embeddings()                │  │  create_token_embeddings()             │   │
│  │                                                 │  │                                         │   │
│  │  For each row:                                  │  │  Flatten tokens from all transactions:  │   │
│  │  ├─ tokens = row['tokenized']                  │  │  ├─ token_id = f"{trans_id}_{token}"   │   │
│  │  ├─ text = " ".join(tokens)                    │  │  ├─ texts = [token for all]            │   │
│  │  ├─ id = trans_id                               │  │  ├─ IDs = [token_id for all]          │   │
│  │  ├─ metadata = {trans_mode, trans_type,       │  │  ├─ metadata = {trans_id: ...}        │   │
│  │  │                    trans_amt}               │  │                                         │   │
│  │  └─ Collect texts, ids, metadatas              │  │  └─ Collect                              │   │
│  │                                                 │  │                                         │   │
│  │  Batch encode (batch_size=64):                 │  │  Batch encode (batch_size=128):        │   │
│  │  └─ model.encode(batch_texts)                   │  │  └─ model.encode(batch_texts)          │   │
│  │                                                 │  │                                         │   │
│  │  Add to ChromaDB:                               │  │  Add to ChromaDB:                      │   │
│  │  └─ txn_collection.add()                        │  │  └─ token_collection.add()            │   │
│  └─────────────────────────────────────────────────┘  └─────────────────────────────────────────┘   │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT: ChromaDB Vector Store                                                                  │   │
│  │                                                                                                 │   │
│  │  Collection 1: expenses_transactions                                                           │   │
│  │  ├─ IDs: trans_id (e.g., "a1b2c3d4...")                                                        │   │
│  │  ├─ Documents: "hdfc bank transfer imps"                                                      │   │
│  │  ├─ Embeddings: 384-dim vectors (all-MiniLM-L6-v2)                                            │   │
│  │  └─ Metadata: {trans_mode: "imps", trans_type: "debit", trans_amt: 5000.0}                   │   │
│  │                                                                                                 │   │
│  │  Collection 2: expenses_transaction_tokens                                                     │   │
│  │  ├─ IDs: "{trans_id}_{token}" (e.g., "a1b2c3d4_hdfc")                                         │   │
│  │  ├─ Documents: "hdfc", "bank", "transfer"                                                    │   │
│  │  ├─ Embeddings: 384-dim vectors                                                               │   │
│  │  └─ Metadata: {trans_id: "a1b2c3d4..."}                                                        │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Purpose of Each Step

### Stage 1: Process Input Data (Ingestion)

**Purpose:** Convert raw bank statement CSV files into structured data in DuckDB.

**Why It Exists:**
- Raw bank statements are in varied, messy formats with headers, footers, and metadata
- Each bank has different CSV structures and date formats
- Need a standardized internal representation for downstream processing
- Enables SQL-based querying and transformations

**Key Transformations:**
- Table extraction from raw file (skip_lines)
- Date parsing (dayfirst=True for DD-MM-YYYY format)
- Column standardization (RAW_COLUMNS)
- Deduplication via hash column (MD5 of date + particulars)
- Incremental loading support

### Stage 2: Curation

**Purpose:** Transform raw transaction data into analytics-ready curated records.

**Why It Exists:**
- Raw data has separate debit/credit columns; need unified amount
- Transaction type (debit/credit) needs explicit column
- Transaction mode (ACH, UPI, NEFT, etc.) needs extraction from particulars
- Clean schema enables consistent downstream processing

**Key Transformations:**
- Debit/Credit consolidation into `trans_amt`
- Transaction type classification
- Transaction mode parsing via separator patterns
- Schema projection (CURATED_COLUMNS)

### Stage 3: Tokenization

**Purpose:** Extract meaningful tokens from transaction particulars for semantic processing.

**Why It Exists:**
- Raw text is too noisy for direct embedding
- Need to normalize variations (e.g., "HDFC Bank" → "hdfc")
- Need to filter irrelevant words (payment modes, junk)
- Enables entity extraction and similarity matching

**Key Transformations:**
- Regex token extraction (alphabetic tokens ≥ 3 chars)
- Bank name normalization (fuzzy matching to canonical names)
- Junk word filtering (payment modes, common words)
- NLTK tokenization for proper word boundaries
- Deduplication while preserving order

### Stage 4: Canonicalization

**Purpose:** Learn supervised mappings from labeled data and apply semantic normalization.

**Why It Exists:**
- Same entity can appear with different tokenizations (e.g., "HDFCBANK", "HDFC BANK LTD", "HDFC")
- Rule-based normalization insufficient for variations
- Need to learn from human-labeled examples
- Enables consistent entity resolution

**Key Transformations:**
- Token→canonical mapping from labeled data (training)
- Semantic similarity fallback using embeddings (inference)
- Most common canonical selection for transactions
- Versioned artifacts for reproducibility

### Stage 5: Vector Embedding

**Purpose:** Generate dense vector representations for semantic similarity search.

**Why It Exists:**
- Enable similarity queries (find similar transactions)
- Support clustering and anomaly detection
- Provide efficient retrieval (ChromaDB)
- Both transaction-level and token-level embeddings for flexibility

**Key Transformations:**
- Text joining from tokens
- SentenceTransformer encoding (all-MiniLM-L6-v2)
- Batch processing for efficiency
- ChromaDB storage with metadata

---

## 5. Technology Stack Summary

### Programming Language

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.x | Primary language for all modules |

### Data Processing & Storage

| Library | Version | Purpose | Usage |
|---------|---------|---------|-------|
| **pandas** | Latest | DataFrame manipulation | CSV parsing, transformations, DataFrame operations |
| **duckdb** | Latest | Embedded analytical DB | Schema storage, SQL queries, data persistence |
| **chromadb** | Latest | Vector database | Embedding storage, similarity search |

### Natural Language Processing

| Library | Version | Purpose | Usage |
|---------|---------|---------|-------|
| **nltk** | Latest | NLP toolkit | word_tokenize, POS tagging (configured but not actively used) |
| **sentence-transformers** | Latest | Text embeddings | all-MiniLM-L6-v2 model for token/transaction encoding |
| **rapidfuzz** | Latest | Fuzzy string matching | Token filtering, bank name normalization |

### Machine Learning

| Library | Version | Purpose | Usage |
|---------|---------|---------|-------|
| **sklearn** | Latest | Similarity computation | cosine_similarity for semantic fallback |
| **numpy** | Latest | Array operations | Embedding storage and manipulation |

### Configuration & Utilities

| Library | Version | Purpose | Usage |
|---------|---------|---------|-------|
| **python-dotenv** | Latest | Environment variables | Loading config from .env file |
| **yaml** | Latest | Config file parsing | read_yaml() utility |
| **tqdm** | Latest | Progress bars | Batch processing progress |
| **argparse** | Built-in | CLI arguments | BaseArgumentParser for CLI |
| **logging** | Built-in | Application logging | LoggerFactory |

### Data Ingestion

| Format | Source | Destination |
|--------|--------|-------------|
| CSV | `landing_zone/statements/acc/axis/` | DuckDB (`acc_raw`) |
| CSV (extracted from xlsx/pdf) | `landing_zone/statements/cc/{axis,icici}/csv/` | DuckDB (`cc_raw`) |
| Parquet | N/A | Internal processing |
| SQL | DuckDB queries | DuckDB tables |

### Database Schema

```
cstone_db (DuckDB)
├── acc_raw / cc_raw                (Raw columns from CSV)
├── acc_curated / cc_curated        (+ trans_type, trans_mode, trans_amt)
├── acc_tokenized / cc_tokenized    (+ tokenized)
├── acc_normalized / cc_normalized  (+ normalized_trans_src)
├── acc_categorized / cc_categorized (predicted category/subcategory)
├── acc_labelled                 (Account labels; also used to train shared canonicalization artifacts)
└── cc_labelled                  (Credit-card category labels)

chroma_db (Vector Store)
├── acc_transactions / cc_transactions                  (transaction-level embeddings)
└── acc_transaction_tokens / cc_transaction_tokens      (token-level embeddings)
```

### File System Structure

```
expenses_tracker/
├── pipeline/                   # Importable pipeline package
│   ├── config.py               # Environment and model configuration
│   ├── utils.py                # Shared database, logging, and file utilities
│   ├── process_input_data.py   # Ingestion and curation
│   ├── tokenization.py          # Transaction tokenization
│   ├── canonicalizer.py         # Source normalization
│   ├── predict_category.py      # Category/subcategory prediction
│   ├── vector_embedding.py     # ChromaDB embedding generation
│   └── cli.py                  # Unified command dispatcher
├── tests/                      # Automated regression tests
├── dashboard/                  # Grafana dashboard and Compose deployment
├── data/                       # SQL query reference
├── docs/                       # Project and dashboard documentation
├── .env.example                # Safe configuration template
└── main.py                     # Root CLI shim

Personal statement CSVs and DuckDB files belong under the external data directory
configured by `delta_dir`. Generated artifacts and logs are local outputs; there is
no `scripts/` directory in the current project layout.
```

### Model Configuration

```python
# SentenceTransformer Model
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim embeddings, ~80MB

# Similarity Thresholds
SIMILARITY_THRESHOLD = 0.9       # Canonicalization fallback

# Batch Sizes
Transaction Embeddings: batch_size = 64
Token Embeddings: batch_size = 128

# NLTK Models (configured but not actively used)
- words
- punkt
- maxent_ne_chunker
- averaged_perceptron_tagger
```

### Key Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `JOB_NAME` | "expenses_etl" | ETL job identifier |
| `LOG_LEVEL` | "INFO" | Logging level |
| `TOTAL_NUM_DAYS` | 730 (2 years) | Data retention window |
| `CN_TRAIN` | True | Enable canonicalizer training |
| `CN_NORMALIZE` | True | Enable normalization |
| `CN_INCREMENTAL` | False | Incremental canonicalizer updates |
| `FILTER_WORDS` | 30+ items | Junk + payment modes to filter |
| `BANK_NAMES` | 50+ items | Bank name patterns for normalization |

---

## Appendix: File-to-Line Reference

| File | Key Functions/Classes | Lines |
|------|---------------------|-------|
| `config.py` | PATHS, TABLES, PAYMENT_MODES, BANK_NAMES, MODEL_NAME | 32-111 |
| `utils.py` | BaseArgumentParser, LoggerFactory, skip_lines, DuckDB utilities | 21-440 |
| `process_input_data.py` | TransactionParser, get_raw_data_to_df, load_raw_data, load_curated_data | 13-316 |
| `tokenization.py` | get_source_data, normalize_tokens_fuzzy, generate_tokens_with_tags | 18-281 |
| `canonicalizer.py` | load_training_data, build_token_mapping, normalize_token, save_versioned_artifacts | 27-260 |
| `vector_embedding.py` | load_tokenized_data, create_transaction_embeddings, create_token_embeddings | 11-171 |

---

*Documentation generated for expenses_tracker project. Last updated: May 2026.*