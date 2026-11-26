import pandas as pd
import os
from . import config
from duckdb import DuckDBPyConnection
from typing import List, Set
from .utils import *

logger = get_logger()

class TransactionParser:
    """Parse transaction particulars to extract mode, entity, and entity ID."""
    
    # Define parsing rules: (trans_modes, separator, part_index)

    sep1_idx1 = ['ACH', 'ATM', 'BRN', 'NACH']
    sep2_idx1 = ['AUTOBPAY', 'POS', 'MBBPAY']
    sep2_idx2 = ['CSHBCK', 'ICONNREF', 'MOB', 'NBSM', 'INB', 'NEFT']
    sep2_idx3 = ['IMPS', 'RTGS', 'UPI']
    sep2_idx4 = ['CLG', 'ECS', 'IFT']

    PARSING_RULES = [
        (sep1_idx1, '-', 0),
        (sep2_idx1, '/', 0),
        (sep2_idx2, '/', 0),
        (sep2_idx3, '/', 0),
        (sep2_idx4, '/', 0),
    ]

    
    MIN_SEPARATORS = {'-': 2, '/': 2}
    
    # @staticmethod
    # def _clean_entity(raw: str) -> str | None:
    #     """Extract clean entity name from raw string."""
    #     if not raw or not isinstance(raw, str):
    #         return None
        
    #     raw = raw.strip()
    #     # Keep only alphabetic characters and spaces
    #     name = re.sub(r"[^A-Za-z ]+", ' ', raw).strip()
    #     name = re.sub(r"\s+", ' ', name).lower()
        
    #     # Return None if name is too short (< 4 characters)
    #     if len(name) < 4:
    #         return None
        
    #     return name
    
    @classmethod
    def extract(cls, particulars: str) -> pd.Series:
        """Extract transaction mode and entity from particulars string."""
        if not isinstance(particulars, str):
            return cls._empty_result()
        
        p = particulars.strip()
        p_upper = p.upper()
        
        for trans_modes, separator, part_index in cls.PARSING_RULES:
            min_sep_count = cls.MIN_SEPARATORS.get(separator, 1)
            
            if p.count(separator) >= min_sep_count and any(p_upper.startswith(mode) for mode in trans_modes):
                parts = p.split(separator)
                
                if abs(part_index) <= len(parts):
                    # ent = cls._clean_entity(parts[part_index])
                    trans_mode_val = parts[0].strip().lower() if parts else None
                    return pd.Series({
                        'trans_mode': trans_mode_val,
                        # 'trans_entity': ent
                    })
        
        return cls._empty_result()
    
    @staticmethod
    def _empty_result() -> pd.Series:
        """Return empty result Series."""
        return pd.Series({
            'trans_mode': None, 
            # 'trans_entity': None
        })


# --- Function to process CSV files ---
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
    
    """Read statement CSVs into the canonical raw-column layout."""

    try:
        logger.info("Loading raw statement data from %s", raw_dir)
        if not columns or not dt_col:
            raise ValueError("columns and dt_col are required to parse statement CSVs")
        
        # Get list of raw CSV files
        raw_csv_list = sorted(list_files_by_extension(raw_dir, config.RAW_FILE_FORMAT))
        if not raw_csv_list:
            raise FileNotFoundError(f"No .{config.RAW_FILE_FORMAT} statement files found in {raw_dir}")

        dfs = []
        for csv_file_path in raw_csv_list:
            csv_file_name = os.path.basename(csv_file_path)
            logger.info("Processing statement file: %s", csv_file_name)
            try:
                csv_lines = skip_lines(
                    csv_file_path,
                    start_index=start_index,
                    end_index=end_index,
                    start_identifier=start_identifier,
                    end_identifier=end_identifier,
                )
                csv_text = string_to_string_io(csv_lines)
                if not csv_text.getvalue().strip():
                    logger.warning("No transaction rows found in %s", csv_file_name)
                    continue

                df = pd.read_csv(
                    csv_text,
                    names=columns,
                    skipinitialspace=True,
                    na_values=na_values,
                    dtype={col: "string" for col in ("cheque_no", "particulars", "sol") if col in columns},
                    on_bad_lines="error",
                )
                missing_columns = set(columns) - set(df.columns)
                if missing_columns:
                    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

                df[dt_col] = pd.to_datetime(df[dt_col], format="mixed", dayfirst=True, errors="raise")
                dfs.append(df)
                logger.info("Parsed %s transaction rows from %s", len(df), csv_file_name)
            except Exception as exc:
                raise ValueError(f"Failed to parse statement file {csv_file_name}: {exc}") from exc

        if not dfs:
            raise ValueError(f"No transaction rows were found in statement files under {raw_dir}")

        df = pd.concat(dfs, axis=0, ignore_index=True)
        logger.info("Loaded %s transaction rows from %s files", len(df), len(dfs))
        return df
    except Exception:
        logger.error("Error processing statement CSV files", exc_info=True)
        raise
        
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
    
    try:
        logger.info("Loading Raw Data....")
        
        # Check if table exists
        select_cols = ",".join([f"{col}::DATE AS {dt_col}" if col.lower() == dt_col else col for col in columns])
        get_raw_data_query = f"""SELECT {config.HASH_COLUMN}, {select_cols} FROM raw_df"""
        
        if ingest_mode == config.HISTORICAL:
            mode = "overwrite"
            dt_col = None
        
        # Load data into raw table from CSV file
        load_data_with_check(cursor, schema_name, table_name, get_raw_data_query, dt_col, mode=mode)
        logger.info("Raw Data Loading Complete!!!!!")
    except Exception:
        logger.error(f"Error occurred while loading raw data.", exc_info=True)
        raise

# --- Function to load clean data ---   
def load_curated_data(cursor: DuckDBPyConnection, 
                    schema_name:str, 
                    raw_table:str, 
                    curated_table: str, 
                    dt_col:str,
                    mode:str = "append",
                    ingest_mode:str = None
                    ):
    
    try:
        logger.info("Loading Curated Data....")

        # --- Read data from raw
        curated_df = cursor.execute(f"SELECT * FROM {schema_name}.{raw_table}").df()

        def extract_trans_type_entity(particulars: str) -> pd.Series:
            return TransactionParser.extract(particulars)

        # Read curated table into DataFrame
        trans_info = curated_df['particulars'].apply(extract_trans_type_entity)
        curated_df = pd.concat([curated_df, trans_info], axis=1)
            
        cursor.register('curated_df', curated_df)

        select_cols = ','.join([c for c in config.CURATED_COLUMNS if c in curated_df.columns])
        get_curated_data_query = f"""
        SELECT 
            {select_cols},
            CASE WHEN debit IS NULL THEN credit ELSE debit END AS trans_amt,
            CASE WHEN debit IS NULL THEN 'credit' ELSE 'debit' END AS trans_type
        FROM curated_df
        """
        
        if ingest_mode == config.HISTORICAL:
            mode = "create"
            dt_col = None
        
        # Load data into curated table from raw table
        load_data_with_check(cursor, schema_name, curated_table, get_curated_data_query, dt_col, mode=mode)
    
        logger.info("Curated Data Loading Complete!!!!!")
    except Exception:
        logger.error(f"Error occurred while loading curated data.", exc_info=True)
        raise


# --- Main function ---
def main(ingest_mode, bank="axis"):
    con = None
    try:
        logger.info("=============== Starting Data Processing =============")

        # Create DuckDB connection
        con = create_duckdb_connection(config.DATABASE_PATH)
        # Use connection directly for registering pandas DataFrames and executing SQL

        t = config.tables("account")
        raw_csv_dir = config.landing_dir("account", bank)

        logger.info("-------------- Starting Raw Data Processing --------------")

        # Process each raw CSV file
        df = get_raw_data_to_df(
            raw_csv_dir,
            columns=config.RAW_COLUMNS,
            dt_col=config.DT_COL,
            start_index=config.RAW_FILE_START_IDX,
            end_index=config.RAW_FILE_END_IDX,
            start_identifier=config.RAW_FILE_START_IDENTIFIER,
            end_identifier=config.RAW_FILE_END_IDENTIFIER,
            na_values=config.DEFAULT_MISSING,
        )

        # Register the pandas DataFrame with DuckDB so SQL queries can reference it as raw_df
        # Note: duckdb's register takes the name and the dataframe
        con.register('raw_df', df)

        # Pass the connection object to loader (functions use cursor.execute; DuckDB connection works)
        load_raw_data(con, df, config.SCHEMA_NAME, t.raw, config.RAW_COLUMNS, config.DT_COL, ingest_mode=ingest_mode)

        # Process curated data
        logger.info("--------------  Starting Curated Data Processing --------------")
        load_curated_data(
            con,
            config.SCHEMA_NAME,
            t.raw,
            t.curated,
            config.DT_COL,
            ingest_mode=ingest_mode,
        )

        logger.info("=============== Data Processing Complete =============")

    except Exception:
        logger.error("Error in main function", exc_info=True)
        raise

    finally:
        # Close the DuckDB connection if it exists
        if con is not None:
            con.close()



if __name__ == "__main__":
  # Initialize logger with script name
  script_name = os.path.basename(__file__).replace('.py', '')
  logger = LoggerFactory.get_logger(get_log_file_path(script_name), config.LOG_LEVEL)
  set_logger(logger)  # Share logger with utils.py

  # Parse arguments
  parser = BaseArgumentParser()
  # Argument for ingest_mode with choices and metavar
  parser.add_argument(
      '-i', '--ingest_mode', 
      dest='ingest_mode', 
      metavar='<Ingestion Mode>', 
      required=True, 
      choices=['incr', 'incremental', 'hist', 'historical'],  # Limiting to valid choices
      help=(
          "Ingestion mode: Choose 'incr' or 'incremental' for Incremental mode, "
          "'hist' or 'historical' for Historical mode.\n"
          "Example: --ingest_mode incr"
      )
  )
  args = parser.parse_args()
  
  ingest_mode = args.ingest_mode.lower()
  if ingest_mode in ['incr', 'incremental']:
    logger.info(f"Ingestion mode is incremental.")
    ingest_mode = config.INCREMENTAL
  elif ingest_mode in ['hist', 'historical']:
    logger.info(f"Ingestion mode is historical.")
    ingest_mode = config.HISTORICAL
  else:
    logger.info('Invalid ingest_mode provided.')
    parser.print_help()
    exit(1)
  

  # Executing main method
  main(ingest_mode=ingest_mode)
