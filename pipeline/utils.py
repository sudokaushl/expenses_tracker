import yaml
import re
import os
import nltk
import logging
import json
import duckdb
from . import config
import inspect
from duckdb import DuckDBPyConnection
from typing import List, Optional, Any
from io import StringIO
from glob import glob
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path

# --- Argument Utilities

import argparse

class BaseArgumentParser:
    """
    A base class for argument parsing.
    Provides basic argument parsing functionality and can be extended for specific needs.
    """
    def __init__(self, description: str = None):
        """
        Initializes the argument parser with an optional description.
        
        Args:
            description (str): An optional description for the command-line tool.
        """
        self.parser = argparse.ArgumentParser(description=description)
        
    def add_argument(self, *args, **kwargs):
        """
        Add an argument to the parser.

        Args:
            *args: Positional arguments for the argument.
            **kwargs: Keyword arguments for the argument, like type, help, etc.
        """
        self.parser.add_argument(*args, **kwargs)
    
    def parse_args(self):
        """
        Parses the command-line arguments.

        Returns:
            Namespace: The parsed arguments as an object with attributes.
        """
        return self.parser.parse_args()

    def print_help(self):
        """
        Prints the help message for the argument parser.
        """
        self.parser.print_help()

    def parse_and_print(self):
        """
        Parses the arguments and prints them.

        Returns:
            Namespace: The parsed arguments.
        """
        args = self.parse_args()
        print(args)
        return args



# --- Logging Utilities ---

def get_log_file_path(job_name: Optional[str] = None) -> str:
    """
    Generates a log file path with a timestamp and optional job name.

    Args:
        job_name: Optional job name to include in the log file name.

    Returns:
        The log file path as a string.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if job_name:
        log_file = f"{job_name}_{current_time}.log"
    else:
        log_file = f"log_file_{current_time}.log"

    parent_dir = Path(__file__).resolve().parents[1]
    log_dir = parent_dir / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)

    return str(log_dir / log_file)

class LoggerFactory:
    """
    A class to manage logging configuration.
    """
    _log = None

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            entry = {
                "timestamp": datetime.fromtimestamp(record.created).astimezone().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "function": record.funcName,
                "message": record.getMessage(),
            }
            if record.exc_info:
                entry["exception"] = self.formatException(record.exc_info)
            return json.dumps(entry, ensure_ascii=False)

    @staticmethod
    def create_logger(log_file: str, log_level: str) -> logging.Logger:
        """
        Creates a logger with specified log file and level.

        Args:
            log_file: Path to the log file.
            log_level: Logging level (INFO, ERROR, DEBUG).

        Returns:
            A logging.Logger object.
        """
        log_format = "%(asctime)s %(funcName)s %(levelname)s - %(message)s"
        LoggerFactory._log = logging.getLogger()
        logging.basicConfig(level=logging.INFO, format=log_format, datefmt=r"%Y-%m-%d")
        log_file = os.path.abspath(log_file)
        if not any(
            getattr(handler, "baseFilename", None) == log_file
            for handler in LoggerFactory._log.handlers
        ):
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(LoggerFactory.JsonFormatter())
            LoggerFactory._log.addHandler(file_handler)

        # Set the logging level based on the user selection
        if log_level == "INFO":
            LoggerFactory._log.setLevel(logging.INFO)
        elif log_level == "ERROR":
            LoggerFactory._log.setLevel(logging.ERROR)
        elif log_level == "DEBUG":
            LoggerFactory._log.setLevel(logging.DEBUG)
        return LoggerFactory._log

    @staticmethod
    def get_logger(log_file: str, log_level: str) -> logging.Logger:
        """
        Retrieves a logger with specified log file and level.

        Args:
            log_file: Path to the log file.
            log_level: Logging level (INFO, ERROR, DEBUG).

        Returns:
            A logging.Logger object.
        """
       
        logger = LoggerFactory.create_logger(log_file, log_level)
        return logger

# Shared logger (set by calling script)
shared_logger = None

def set_logger(logger_obj):
    """Set the shared logger from the calling script."""
    global shared_logger
    shared_logger = logger_obj

def get_logger():
    """Get the shared logger or create a fallback one."""
    global shared_logger
    if shared_logger is not None:
        return shared_logger
    # Fallback if no logger set (creates utils log)
    if shared_logger is None:
        shared_logger = LoggerFactory.get_logger(get_log_file_path("utils"), config.LOG_LEVEL)
    return shared_logger

# --- File Utilities ---
def list_files_by_extension(dir_name: str, file_extension: str) -> List[str]:
    """
    Lists all files with the specified extension in the given directory.

    Args:
        dir_name: Path to the directory.
        file_extension: File extension to search for.

    Returns:
        A list of file paths.
    """
    get_logger().info(f"Checking file extension: *.{file_extension} in directory {dir_name}")
    file_extension_pattern = f"*.{file_extension}"
    return glob(os.path.join(dir_name, file_extension_pattern))

def read_yaml(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path: Path to the YAML file.

    Returns:
        A dictionary containing the YAML file contents.
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        get_logger().error(f"Error reading YAML file: {file_path}", exc_info=True)
        raise

# --- Data Processing Utilities ---

def skip_lines(
    file_path: str, 
    start_identifier: str = None, 
    end_identifier: str = None, 
    start_index: int = None, 
    end_index: int = None
    ) -> List[str]:
    """
    Skips lines in a file based on the occurrence of newline characters.

    Args:
        file_path: Path to the file.
        start_index: Occurrence of the newline after which data needs to be kept.
        end_index: Occurrence of the newline before which data needs to be kept.
        start_identifier: Line identifying the start of the table
        end_identifier: Line identifying the end of the table

    Returns:
        A string containing the skipped lines.
    """
    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
            lines = file.readlines()

        get_logger().info("Read %s source lines from %s", len(lines), file_path)
        first_index = None
        last_index = None

        if start_index is not None or end_index is not None:
            if start_index is None or end_index is None:
                raise ValueError("start_index and end_index must be supplied together")
            first_index, last_index = start_index, end_index
        elif start_identifier is not None:
            first_index = next(
                (idx for idx, line in enumerate(lines) if start_identifier in line),
                None,
            )
            if first_index is None:
                raise ValueError(f"Start marker not found in {file_path}")

            if end_identifier is not None:
                for idx in range(first_index + 1, len(lines)):
                    line = lines[idx]
                    is_end = (
                        not line.strip()
                        if end_identifier in ("\n", "\r\n")
                        else line.strip() == end_identifier.strip()
                    )
                    if is_end:
                        last_index = idx
                        break
        else:
            return lines

        if first_index < -1 or first_index >= len(lines):
            raise ValueError(f"Invalid start index {first_index} for {file_path}")
        if last_index is not None and (last_index < first_index + 1 or last_index > len(lines)):
            raise ValueError(f"Invalid end index {last_index} for {file_path}")

        extracted = lines[first_index + 1:last_index]
        get_logger().info("Extracted %s data lines from %s", len(extracted), file_path)
        return extracted
    except Exception as e:
        get_logger().error(f"Error skipping lines in file: {file_path}", exc_info=True)
        raise

def string_to_string_io(text) -> StringIO:
    """
    Converts a string to a StringIO object.

    Args:
        text: The string to convert.

    Returns:
        A StringIO object containing the string.
    """
    get_logger().info("Converting extracted CSV text to StringIO")
    if isinstance(text, str):
        content = text
    else:
        content = "".join(text)
    return StringIO(content)

# --- DuckDB Utilities ---

def create_duckdb_connection(database_path: str) -> DuckDBPyConnection:
    """
    Creates a DuckDB connection to the specified database.

    Args:
        database_path: Path to the DuckDB database file.

    Returns:
        A DuckDBPyConnection object.
    """
    try:
        get_logger().info(f"Creating DuckDB connection to database: {database_path}")
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        return duckdb.connect(database_path)
    except Exception as e:
        get_logger().error(f"Error creating DuckDB connection: {database_path}", exc_info=True)
        raise
    
def get_max_date(cursor: DuckDBPyConnection, schema_name: str, table_name: str, dt_col: str) -> str:
    """
    Retrieves the maximum date from a specified column in a DuckDB table.

    Args:
        cursor: DuckDB connection object.
        schema_name: Name of the schema.
        table_name: Name of the table.
        dt_col: Name of the date column.

    Returns:
        The maximum date as a string, or None if no valid date is found.
    """
    try:
        get_logger().info(f"Fetching maximum date from table: {schema_name}.{table_name}")
        max_dt_query = f"SELECT MAX({dt_col}) FROM {schema_name}.{table_name}"
        result = cursor.execute(max_dt_query).fetchone()
        
        # If result is None, return None (meaning no dates are available)
        return result[0] if result[0] is not None else None
        
    except Exception as e:
        get_logger().error(f"Error getting maximum date from table: {schema_name}.{table_name}", exc_info=True)
        raise

def get_table_data(cursor: DuckDBPyConnection, schema_name: str, table_name: str, cols: List[str] = None, where_cond: str = None) -> config.PANDAS_DF:
    """
    Retrieves data from a DuckDB table with optional column selection and filtering.

    Args:
        cursor: DuckDB connection object.
        schema_name: Name of the schema.
        table_name: Name of the table.
        cols: List of columns to select. If None, selects all columns.
        where_cond: WHERE clause for filtering data.

    Returns:
        A Pandas DataFrame containing the table data.
    """
    try:
        get_logger().info(f"Fetching data from table: {schema_name}.{table_name}")
        select_query = f"SELECT {',' .join(cols) if cols else '*'} FROM {schema_name}.{table_name}"
        
        if where_cond:
            select_query += f" WHERE {where_cond}"
        
        get_logger().info(f"SQL QUERY FOR FETCHING DATA: {select_query}")
        return cursor.execute(select_query).df()
    
    except Exception as e:
        get_logger().error(f"Error getting data from table: {schema_name}.{table_name}", exc_info=True)
        raise

def load_data_into_table(
    cursor: DuckDBPyConnection, 
    schema_name: str, 
    table_name: str, 
    source_format: str = None, 
    data_dir: str = None,
    file_delimiter: str = None,
    header: bool = None,
    source_df: config.PANDAS_DF = None, 
    query: str = None,
    source_cols: List[str] = None, 
    mode: str = "create"
) -> None:
    """
    Loads data from a Pandas DataFrame, CSV, or SQL query into a DuckDB table.

    Args:
        cursor: DuckDB connection object.
        schema_name: Name of the schema.
        table_name: Name of the table.
        mode: Load mode ("create", "append", "overwrite").
    """
    try:
        get_logger().info(f"Loading data into table: {schema_name}.{table_name}")

        # Map source format to corresponding data source string
        format_map = {
            "pandas": "source_df",
            "csv": f"read_csv('{data_dir}/*.csv', delim='{file_delimiter}', header={str(header).lower()})",
            "sql": f"({query}) q"
        }

        src_str = format_map.get(source_format.lower())
        if not src_str:
            raise ValueError(f"Unsupported source format: {source_format}")

        src_cols = ','.join(source_cols) if source_cols else "*"
 
        create_table_query = f"CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} AS SELECT {src_cols} FROM {src_str}"
        insert_data_query = f"INSERT INTO {schema_name}.{table_name} SELECT {src_cols} FROM {src_str}"
        drop_table_query = f"DROP TABLE IF EXISTS {schema_name}.{table_name}"
        truncate_table_query = f"TRUNCATE {schema_name}.{table_name}"
        
        query_map = {
            "create": f"{drop_table_query}; {create_table_query};",
            "append": f"{insert_data_query}",
            "overwrite" : f"{truncate_table_query}; {insert_data_query}"
        }
        get_logger().info(f"Load mode is : {mode.lower()}")
        query_str = query_map.get(mode.lower())
        if not query_str:
            raise ValueError(f"Unsupported load mode: {mode}")
        
        get_logger().info(f"SQL QUERY FOR LOADING DATA : {query_str}")

        # Execute the query
        cursor.execute(query_str)
        cursor.commit()
        
        get_logger().info(f"Data loaded into table: {schema_name}.{table_name}")

    except Exception as e:
        get_logger().error(f"Error loading data into table: {schema_name}.{table_name}")
        raise

def load_data_with_check(cursor, schema_name, table_name, query, dt_col=None, mode=None):
    get_logger().info("Loading data based on date column: %s", dt_col)
    table_exists = check_table_exists(cursor, schema_name, table_name)
    if not table_exists:
        mode = "create"
    elif mode in (None, "append"):
        # Prefer transaction identity over a date watermark: a strict date > max
        # misses late-arriving rows whose date matches the current maximum.
        columns = {
            row[1]
            for row in cursor.execute(
                f"PRAGMA table_info('{schema_name}.{table_name}')"
            ).fetchall()
        }
        if "trans_id" in columns:
            query = f"""
                SELECT incoming.*
                FROM ({query}) AS incoming
                WHERE incoming.trans_id IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1
                      FROM {schema_name}.{table_name} AS existing
                      WHERE existing.trans_id = incoming.trans_id
                  )
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY incoming.trans_id ORDER BY incoming.trans_id
                ) = 1
            """
        elif dt_col is not None:
            max_dt = get_max_date(cursor, schema_name, table_name, dt_col)
            if max_dt is not None:
                query = f"SELECT * FROM ({query}) AS incoming WHERE incoming.{dt_col}::DATE > DATE '{max_dt}'"
        mode = "append"
    load_data_into_table(cursor, schema_name, table_name, source_format="sql", query=query, mode=mode)


def check_table_exists(cursor: DuckDBPyConnection, schema_name: str, table_name: str) -> bool:
    """
    Check if table exists in schema or not. Returns boolean.

    Args:
        cursor (DuckDBPyConnection): DuckDB connection object.
        schema_name: Name of the schema.
        table_name: Name of the table.

    Returns:
        bool: Returns True of table exists, False otherwise.
    """
    
    try:
        get_logger().info(f"Checking if table exists...")
        result = cursor.execute(
            """SELECT COUNT(*)
               FROM information_schema.tables
               WHERE table_schema = ? AND table_name = ?""",
            [schema_name, table_name],
        ).fetchone()

        if result[0] > 0:
            get_logger().info(f"Table exists: {schema_name}.{table_name}")
            return True
        else:
            get_logger().info(f"Table does not exist: {schema_name}.{table_name}")
            return False
    except Exception as e:
        get_logger().error(f"Error checking table: {schema_name}.{table_name}", exc_info=True)
        raise
    
# --- NLTK Utilities ---
def configure_nltk(nltk_models: List[str] = None) -> None:
    """
    Downloads and configures NLTK models.

    Args:
        nltk_models: List of NLTK models to download.
    """
    try:
        for model in nltk_models:
            nltk.download(model)
    except Exception as e:
        get_logger().error("Error configuring NLTK models", exc_info=True)
        raise