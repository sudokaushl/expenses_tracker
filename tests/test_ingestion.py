"""Regression tests for statement CSV ingestion and incremental persistence."""

import logging
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from pipeline import config
from pipeline.process_input_data import get_raw_data_to_df
from pipeline.utils import load_data_with_check, set_logger, skip_lines, string_to_string_io


class CsvIngestionTests(unittest.TestCase):
    def setUp(self):
        set_logger(logging.getLogger("expenses_tracker_test"))

    def test_extract_and_parse_preserves_description_spaces_and_quotes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "statement.csv"
            path.write_text(
                "Statement period,Feb 2024\n"
                f"{config.RAW_FILE_START_IDENTIFIER}\n"
                '01-02-2024,,"UPI/abc/Starbucks Coffee",100.00,,500.00,123\n'
                "\n"
                "End of statement\n",
                encoding="utf-8",
            )

            rows = skip_lines(
                str(path),
                start_identifier=config.RAW_FILE_START_IDENTIFIER,
                end_identifier="\n",
            )
            csv_text = string_to_string_io(rows)
            self.assertIn('"UPI/abc/Starbucks Coffee"', csv_text.getvalue())

            frame = get_raw_data_to_df(
                temp_dir,
                columns=config.RAW_COLUMNS,
                dt_col=config.DT_COL,
                start_identifier=config.RAW_FILE_START_IDENTIFIER,
                end_identifier="\n",
                na_values={""},
            )

            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.loc[0, "particulars"], "UPI/abc/Starbucks Coffee")
            self.assertEqual(frame.loc[0, "trans_dt"], pd.Timestamp("2024-02-01"))

    def test_string_io_joins_lines_without_inserting_spaces(self):
        self.assertEqual(string_to_string_io(["a,b\n", '"c d",e\n']).getvalue(), 'a,b\n"c d",e\n')

    def test_empty_directory_fails_instead_of_returning_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                get_raw_data_to_df(
                    temp_dir,
                    columns=config.RAW_COLUMNS,
                    dt_col=config.DT_COL,
                    start_identifier=config.RAW_FILE_START_IDENTIFIER,
                    end_identifier="\n",
                )

    def test_file_with_no_transaction_rows_fails_safely(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "statement.csv"
            path.write_text(
                f"{config.RAW_FILE_START_IDENTIFIER}\n\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "No transaction rows"):
                get_raw_data_to_df(
                    temp_dir,
                    columns=config.RAW_COLUMNS,
                    dt_col=config.DT_COL,
                    start_identifier=config.RAW_FILE_START_IDENTIFIER,
                    end_identifier="\n",
                )


class IncrementalLoadTests(unittest.TestCase):
    def test_incremental_load_includes_late_same_day_records_and_is_idempotent(self):
        connection = duckdb.connect(":memory:")
        connection.execute("CREATE SCHEMA expenses")
        connection.execute(
            "CREATE TABLE expenses.target (trans_id VARCHAR, trans_dt DATE)"
        )
        connection.execute(
            "INSERT INTO expenses.target VALUES ('existing', DATE '2024-02-01')"
        )
        connection.execute(
            "CREATE TEMP TABLE incoming (trans_id VARCHAR, trans_dt DATE)"
        )
        connection.execute(
            "INSERT INTO incoming VALUES "
            "('existing', DATE '2024-02-01'), "
            "('late-same-day', DATE '2024-02-01'), "
            "('late-same-day', DATE '2024-02-01')"
        )

        query = "SELECT trans_id, trans_dt FROM incoming"
        load_data_with_check(
            connection, "expenses", "target", query, dt_col="trans_dt", mode="append"
        )
        load_data_with_check(
            connection, "expenses", "target", query, dt_col="trans_dt", mode="append"
        )

        result = connection.execute(
            "SELECT trans_id FROM expenses.target ORDER BY trans_id"
        ).fetchall()
        self.assertEqual(result, [("existing",), ("late-same-day",)])
        connection.close()

    def test_transaction_hash_changes_when_amount_or_balance_changes(self):
        connection = duckdb.connect(":memory:")
        frame = pd.DataFrame(
            [
                [pd.Timestamp("2024-02-01"), "", "UPI/abc/shop", 100.0, None, 500.0, "1"],
                [pd.Timestamp("2024-02-01"), "", "UPI/abc/shop", 200.0, None, 400.0, "1"],
            ],
            columns=config.RAW_COLUMNS,
        )
        connection.register("raw_df", frame)

        ids = connection.execute(
            f"SELECT {config.HASH_COLUMN} FROM raw_df"
        ).fetchall()
        self.assertNotEqual(ids[0][0], ids[1][0])
        connection.close()


if __name__ == "__main__":
    unittest.main()
