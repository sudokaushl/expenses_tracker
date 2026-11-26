import re
from functools import lru_cache
import pandas as pd
import nltk
from . import config
from .utils import *
import os
from rapidfuzz import fuzz, process

logger = get_logger()


@lru_cache(maxsize=1)
def _english_words() -> frozenset:
    try:
        from nltk.corpus import words
        return frozenset(w.lower() for w in words.words())
    except LookupError:
        return frozenset()

# --- Function to get source data from DuckDB table ---
def get_source_data(cursor, schema_name: str, src_table: str, tgt_table: str, ingest_mode: str = None, dt_col: str = None) -> config.PANDAS_DF:
    """
    Gets data from the source table in DuckDB.

    Args:
        cursor: DuckDB connection object.
        schema_name: Name of the schema.
        src_table: Name of the source table.
        tgt_table: Name of the target table for comparison.
        ingest_mode: Whether to load all data or only data after a specific date ("hist" for full load, "incr" for incremental load).
        dt_col: Date column to use for incremental load.

    Returns:
        Pandas DataFrame containing the source data.
    """
    try:
        where_cond = None
        if ingest_mode == config.HISTORICAL:
            # Fetch all data for full load
            df = get_table_data(cursor, schema_name, src_table, where_cond=where_cond)
        elif ingest_mode == config.INCREMENTAL:
            if not check_table_exists(cursor, schema_name, tgt_table):
                df = get_table_data(cursor, schema_name, src_table)
            else:
                source_columns = {
                    row[1]
                    for row in cursor.execute(
                        f"PRAGMA table_info('{schema_name}.{src_table}')"
                    ).fetchall()
                }
                target_columns = {
                    row[1]
                    for row in cursor.execute(
                        f"PRAGMA table_info('{schema_name}.{tgt_table}')"
                    ).fetchall()
                }
                if "trans_id" in source_columns and "trans_id" in target_columns:
                    query = f"""
                        SELECT source.*
                        FROM {schema_name}.{src_table} AS source
                        WHERE source.trans_id IS NOT NULL
                          AND NOT EXISTS (
                              SELECT 1 FROM {schema_name}.{tgt_table} AS target
                              WHERE target.trans_id = source.trans_id
                          )
                    """
                    df = cursor.execute(query).df()
                else:
                    max_dt = get_max_date(cursor, schema_name, tgt_table, dt_col)
                    where_cond = f"{dt_col}::DATE > DATE '{max_dt}'" if max_dt is not None else None
                    df = get_table_data(cursor, schema_name, src_table, where_cond=where_cond)
        else:
            logger.error(f"Unrecognized ingest_mode: {ingest_mode}")
            raise ValueError("Invalid ingest_mode provided.")
            
        return df if df is not None else pd.DataFrame()  # Return empty DataFrame if None
    
    except Exception as e:
        logger.error(f"Error getting source data from table: {schema_name}.{src_table}", exc_info=True)
        raise

def get_tokens(row, pattern, corpus_col):
  # trans_entity_val = row.get('trans_entity', None)

  # if trans_entity_val and isinstance(trans_entity_val, str) and trans_entity_val.strip():
  #   # Use trans_entity as the token(s) (flat list)
  #   return trans_entity_val.strip().lower().split()
  
  # Otherwise, extract tokens from corpus_col
  return [word for word in re.findall(pattern, str(row[corpus_col]).lower()) if len(word) >= 3]

def filter_tokens_fuzzy(word: str, word_list: list[str], threshold: int = 80) -> bool:
    """Return True if `word` is similar to any word in `word_list` above threshold."""
    if not word_list:
        return False

    if word in word_list:
        return True

    # Typos rarely change the first letter; this keeps "mother" from matching the junk word "other"
    candidates = [w for w in word_list if w[:1] == word[:1]]
    if not candidates:
        return False

    match = process.extractOne(word, candidates, scorer=fuzz.ratio)
    if match is None:
        return False

    _, score, _ = match
    return score >= threshold

def normalize_tokens_fuzzy(tokens: list[str], bank_names: list[str], threshold: int = 85) -> list[str]:
    normalized_tokens = []
    english = _english_words()
    
    # Sort bank names by length (longest first) to prioritize longer matches
    bank_names_sorted = sorted(bank_names, key=len, reverse=True)
    
    for tok in tokens:
        tok_clean = tok.strip().lower()
        
        # Skip empty tokens
        if not tok_clean:
            continue
        
        # Strategy 1: Check for prefix/substring match (bidirectional)
        matched = False
        for bank_name in bank_names_sorted:
            if len(bank_name) >= 4:
                # Truncated names only (e.g. "stateba"); real words like "mother" must not expand to "motherdairy"
                if bank_name.startswith(tok_clean) and len(tok_clean) >= 5 and tok_clean not in english:
                    normalized_tokens.append(bank_name)
                    logger.debug(f"Prefix match: '{tok}' → '{bank_name}'")
                    matched = True
                    break
                # Check if bank name is contained in token (e.g., "hdfc" in "hdfcbankltd")
                elif bank_name in tok_clean:
                    normalized_tokens.append(bank_name)
                    logger.debug(f"Substring match: '{tok}' → '{bank_name}'")
                    matched = True
                    break
        
        if matched:
            continue
        
        # Strategy 2: Fuzzy matching with adjusted length ratio
        if len(tok_clean) >= 5:
            match = process.extractOne(
                tok_clean, bank_names, scorer=fuzz.ratio, processor=str.lower
            )
            
            if match:
                best_match, score, _ = match
                
                # Calculate length ratio - more lenient for shorter tokens
                len_ratio = min(len(best_match), len(tok_clean)) / max(len(best_match), len(tok_clean))
                
                # Adjusted thresholds:
                # - If token is short (5-8 chars), allow lower length ratio (0.4)
                # - If token is longer (9+ chars), require higher length ratio (0.6)
                min_len_ratio = 0.4 if len(tok_clean) <= 8 else 0.6
                
                if score >= threshold and len_ratio >= min_len_ratio:
                    normalized_tokens.append(best_match.lower())
                    logger.debug(f"Fuzzy match: '{tok}' → '{best_match}' (score={score:.2f}, len_ratio={len_ratio:.2f})")
                else:
                    normalized_tokens.append(tok_clean)
            else:
                normalized_tokens.append(tok_clean)
        else:
            # Token too short for fuzzy matching
            normalized_tokens.append(tok_clean)
    
    return normalized_tokens



def generate_tokens_with_tags(df, pattern, corpus_col, filter_words=None, bank_threshold=90, filter_threshold=85):

    filter_words = set(filter_words or [])
    logger.info(f"Generating tokens for column: {corpus_col}")

    # Fill NaNs and convert to string
    text_series = df[corpus_col].fillna("").astype(str)

    # Step 1: Regex tokenization
    token_lists = text_series.str.lower().apply(lambda x: re.findall(pattern, x))

    # Step 2: Normalize bank names
    token_lists_normalized = token_lists.apply(
        lambda tokens: normalize_tokens_fuzzy(tokens, config.BANK_NAMES, threshold=bank_threshold)
    )

    # Step 3: Fuzzy filter unwanted words
    token_lists_filtered = token_lists_normalized.apply(
        lambda tokens: [t.strip() for t in tokens if len(t) >= 3 and not filter_tokens_fuzzy(t, filter_words, threshold=filter_threshold)]
    )

    # Step 4: Final NLTK tokenization
    tokenized_lists = token_lists_filtered.apply(lambda t: nltk.word_tokenize(" ".join(t)))

    # Step 5: Deduplicate
    df["tokenized"] = tokenized_lists.apply(lambda t: list(dict.fromkeys(t)))

    return df
  
def load_tokenized_data(cursor: DuckDBPyConnection, 
                    tokenized_df:str, 
                    schema_name:str, 
                    table_name: str, 
                    dt_col:str,
                    ingest_mode:str = None,
                    mode: str = "append"
                    ):
  try:
    logger.info("Loading Tokenized Data....")
    select_cols = ",".join([f"{col}::DATE AS {dt_col}" if col.lower() == dt_col else col for col in tokenized_df.columns])
    select_query = f"SELECT {select_cols} FROM tokenized_df"
    
    if ingest_mode == config.HISTORICAL:
      mode = "create"
      dt_col = None
    
    load_data_with_check(cursor, schema_name, table_name, query=select_query, dt_col=dt_col, mode=mode)
    
    logger.info("Tokenized Data Loading Complete!!!!!")
  except Exception as e:
    logger.error(f"Error occurred while loading tokenized data.", exc_info=True)
    raise



# --- Main function ---
def main(ingest_mode, statement="account"):
        """Execute tokenization and persist the tokenized stage."""
        con = None
        try:
                nltk.download("punkt")
                nltk.download("punkt_tab")
                logger.info("=============== Starting Tokenization Process =============")
                configure_nltk(config.NLTK_MODELS)

                con = create_duckdb_connection(config.DATABASE_PATH)
                t = config.tables(statement)
                src_table = t.curated
                tgt_table = t.tokens
                logger.info("Getting source data...")
                df_src = get_source_data(
                        con,
                        config.SCHEMA_NAME,
                        src_table,
                        tgt_table,
                        dt_col=config.DT_COL,
                        ingest_mode=ingest_mode,
                )

                logger.info("Generating tokens...")
                df_tokenized = generate_tokens_with_tags(
                        df=df_src,
                        corpus_col=config.CORPUS_COL,
                        pattern=r"[A-Za-z]+",
                        filter_words=config.FILTER_WORDS,
                )
                con.register("tokenized_df", df_tokenized)
                load_tokenized_data(
                        con,
                        df_tokenized,
                        config.SCHEMA_NAME,
                        tgt_table,
                        config.DT_COL,
                        ingest_mode=ingest_mode,
                )
                logger.info("=============== Tokenization Process Ended =============")
        except Exception:
                logger.error("Error in tokenization pipeline", exc_info=True)
                raise
        finally:
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
