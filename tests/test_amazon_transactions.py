"""Tests for in-memory Amazon Pay capture and CSV export."""

import csv
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.amazon_transactions import API_PATH, COLUMNS, HistoryCapture, main, scroll_history, write_csv


def transaction(timestamp, amount="10.00"):
    return {"transactionDetails": {"date": timestamp, "amount": amount, "status": "SUCCESS",
                                   "type": "DEBIT", "authenticator": "Paid, café",
                                   "authenticatorTwo": "Amazon Pay"},
            "transactionDetailPageURL": "https://example.test/?a=1&b=2"}


class FakeResponse:
    def __init__(self, data, *, operation="getTransactionHistory", ok=True, status=200):
        self.url = "https://www.amazon.in" + API_PATH
        self.request = SimpleNamespace(post_data=json.dumps({"operationName": operation}))
        self.ok = ok
        self.status = status
        self.data = data

    async def json(self):
        return self.data


def history(transactions, has_more):
    return {"data": {"GetTransactionHistory": {
        "transactions": transactions, "hasMoreTransactions": has_more,
    }}}


class FakePage:
    def __init__(self, capture, responses=()):
        self.capture = capture
        self.responses = list(responses)
        self.mouse = self
        self.scrolls = 0

    async def evaluate(self, expression):
        pass

    async def wheel(self, x, y):
        pass

    async def wait_for_timeout(self, milliseconds):
        self.scrolls += 1
        if self.responses:
            self.capture.on_response(self.responses.pop(0))


class AmazonHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_url_pages_stay_in_memory_and_keep_duplicates(self):
        capture = HistoryCapture()
        first = transaction("2026-01-01T10:00:00Z")
        capture.on_response(FakeResponse(history([first], True)))
        capture.on_response(FakeResponse(history([first, transaction("2026-01-02T10:00:00Z")], False)))
        await capture.drain()
        self.assertEqual((capture.pages, capture.transactions, capture.has_more), (2, 3, False))
        self.assertEqual(len(capture.rows), 3)
        self.assertEqual(capture.rows[0], capture.rows[1])

    async def test_scrolls_until_final_response_without_height_change(self):
        capture = HistoryCapture()
        page = FakePage(capture, [FakeResponse(history([transaction("2026-01-01T00:00:00Z")], True)),
                                  FakeResponse(history([], False))])
        await scroll_history(page, capture)
        self.assertEqual((page.scrolls, capture.pages, capture.has_more), (2, 2, False))

    async def test_stalled_scroll_is_reported_as_incomplete(self):
        capture = HistoryCapture()
        capture.on_response(FakeResponse(history([], True)))
        with self.assertRaisesRegex(RuntimeError, "history may be incomplete"):
            await scroll_history(FakePage(capture), capture)

    async def test_other_operations_ignored_and_graphql_errors_reported(self):
        capture = HistoryCapture()
        capture.on_response(FakeResponse({}, operation="unrelated", ok=False))
        capture.on_response(FakeResponse({"errors": [{}]}))
        with self.assertRaisesRegex(RuntimeError, "could not be captured"):
            await scroll_history(FakePage(capture), capture)
        self.assertEqual(capture.errors, ["Amazon returned GraphQL errors"])

    async def test_inclusive_dates_use_india_time_across_utc_midnight(self):
        capture = HistoryCapture(date(2026, 1, 2), date(2026, 1, 3))
        dates = ["2026-01-01T18:29:59Z", "2026-01-01T18:30:00Z",
                 "2026-01-03T18:29:59Z", "2026-01-03T18:30:00Z"]
        capture.on_response(FakeResponse(history([transaction(value) for value in dates], False)))
        await capture.drain()
        self.assertEqual([row["date"] for row in capture.rows], dates[1:3])
        self.assertEqual(capture.transactions, 4)

    async def test_one_sided_filters_and_bad_filtered_date(self):
        dates = ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"]
        for start, end, expected in [(date(2026, 1, 2), None, dates[1:]),
                                     (None, date(2026, 1, 1), dates[:1])]:
            capture = HistoryCapture(start, end)
            capture.on_response(FakeResponse(history([transaction(value) for value in dates], False)))
            await capture.drain()
            self.assertEqual([row["date"] for row in capture.rows], expected)
        capture = HistoryCapture(date(2026, 1, 1))
        capture.on_response(FakeResponse(history([transaction("bad-date")], False)))
        with self.assertRaisesRegex(RuntimeError, "could not be captured"):
            await scroll_history(FakePage(capture), capture)
        self.assertEqual(capture.rows, [])


class AmazonCsvTests(unittest.TestCase):
    def test_csv_columns_values_and_overwrite_protection(self):
        from scripts.amazon_transactions import flatten_transaction

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new" / "transactions.csv"
            row = flatten_transaction(transaction("2026-01-01T00:00:00Z"))
            write_csv([row, row], output)
            with output.open(encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, list(COLUMNS))
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["authenticator"], "Paid, café")
            self.assertEqual(rows[0]["amount"], "10.00")
            self.assertEqual(rows[0]["transactionDetailPageURL"], "https://example.test/?a=1&b=2")
            with self.assertRaises(FileExistsError):
                write_csv([], output)
            write_csv([], output, overwrite=True)
            with output.open(encoding="utf-8", newline="") as file:
                self.assertEqual(list(csv.DictReader(file)), [])

    def test_invalid_cli_dates_do_not_start_browser(self):
        with patch("scripts.amazon_transactions.download_transactions") as download:
            for args in (["--start-date", "2026-02-30"],
                         ["--start-date", "20260101"],
                         ["--start-date", "2026-01-03", "--end-date", "2026-01-01"]):
                with self.assertRaises(SystemExit) as exc:
                    main(args)
                self.assertEqual(exc.exception.code, 2)
            download.assert_not_called()

    def test_no_csv_written_if_download_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "transactions.csv"
            output.write_text("original", encoding="utf-8")
            with patch("scripts.amazon_transactions.download_transactions", side_effect=RuntimeError("incomplete")):
                with self.assertRaises(SystemExit) as exc:
                    main(["--output", str(output), "--overwrite"])
            self.assertEqual(exc.exception.code, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "original")

    def test_command_writes_only_csv_from_downloaded_rows(self):
        from scripts.amazon_transactions import flatten_transaction

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "transactions.csv"
            capture = HistoryCapture()
            capture.pages = 2
            capture.rows = [flatten_transaction(transaction("2026-01-02T00:00:00Z"))]
            with patch("scripts.amazon_transactions.download_transactions", return_value=capture) as download:
                main(["--start-date", "2026-01-01", "--end-date", "2026-01-31", "--output", str(output)])
            download.assert_awaited_once_with(date(2026, 1, 1), date(2026, 1, 31))
            with output.open(encoding="utf-8", newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 1)
            self.assertEqual(list(Path(directory).glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()