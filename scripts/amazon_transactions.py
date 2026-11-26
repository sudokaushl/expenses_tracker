"""Download Amazon Pay transaction history directly into a CSV."""

import argparse
import asyncio
import csv
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright

URL = "https://www.amazon.in/pay/history"
API_PATH = "/apay/audi/authenticateAndExecuteV2"
OUTPUT = Path("amazon_pay_responses/transactions.csv")
MAX_IDLE_SCROLLS = 5
SCROLL_DELAY_MS = 2000
INDIA_TIME = timezone(timedelta(hours=5, minutes=30))
DETAIL_COLUMNS = ("date", "amount", "status", "type", "authenticator", "authenticatorTwo")
COLUMNS = (*DETAIL_COLUMNS, "transactionDetailPageURL")


def flatten_transaction(transaction, start_date=None, end_date=None):
    if not isinstance(transaction, dict) or not isinstance(transaction.get("transactionDetails"), dict):
        raise ValueError("Invalid transactionDetails in transaction history")
    details = transaction["transactionDetails"]
    if start_date is not None or end_date is not None:
        timestamp = details.get("date")
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("timestamp has no timezone")
            transaction_date = parsed.astimezone(INDIA_TIME).date()
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Invalid or missing transaction date; cannot apply date filter") from exc
        if (start_date is not None and transaction_date < start_date) or (
            end_date is not None and transaction_date > end_date
        ):
            return None
    row = {column: details.get(column) for column in DETAIL_COLUMNS}
    row["transactionDetailPageURL"] = transaction.get("transactionDetailPageURL")
    return row


def write_csv(rows, output: Path, *, overwrite=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class HistoryCapture:
    def __init__(self, start_date=None, end_date=None):
        self.start_date = start_date
        self.end_date = end_date
        self.pages = 0
        self.transactions = 0
        self.rows = []
        self.has_more = None
        self.errors = []
        self._lock = asyncio.Lock()
        self._pending = set()

    def on_response(self, response):
        url = urlparse(response.url)
        if url.hostname != "www.amazon.in" or url.path != API_PATH:
            return
        task = asyncio.create_task(self._capture(response))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _capture(self, response):
        # Serialize handlers so page counts and completion reflect response order.
        async with self._lock:
            try:
                try:
                    request_data = json.loads(response.request.post_data or "{}")
                except (TypeError, ValueError):
                    request_data = {}
                operation = request_data.get("operationName") if isinstance(request_data, dict) else None
                if operation and operation != "getTransactionHistory":
                    return
                if not response.ok:
                    raise RuntimeError(f"HTTP {response.status}")
                data = await response.json()
                if not isinstance(data, dict):
                    return
                history = (data.get("data") or {}).get("GetTransactionHistory")
                if history is None:
                    if data.get("errors") and operation == "getTransactionHistory":
                        raise RuntimeError("Amazon returned GraphQL errors")
                    return
                if not isinstance(history, dict) or not isinstance(history.get("transactions"), list):
                    raise RuntimeError("Unexpected transaction history response")
                if not isinstance(history.get("hasMoreTransactions"), bool):
                    raise RuntimeError("Missing hasMoreTransactions in response")

                rows = [flatten_transaction(transaction, self.start_date, self.end_date)
                        for transaction in history["transactions"]]
                self.rows.extend(row for row in rows if row is not None)
                self.pages += 1
                self.transactions += len(history["transactions"])
                self.has_more = history["hasMoreTransactions"]
                print(f"Captured page {self.pages}: {len(history['transactions'])} transactions; "
                      f"more pages: {self.has_more}")
            except Exception as exc:
                self.errors.append(str(exc))
                print(f"Failed to capture an Amazon Pay response: {exc}")

    async def drain(self):
        while self._pending:
            await asyncio.gather(*tuple(self._pending))


async def scroll_history(page, capture):
    idle_scrolls = 0
    while True:
        await capture.drain()
        if capture.errors:
            raise RuntimeError("A response could not be captured; history may be incomplete.")
        if capture.has_more is False:
            return
        if idle_scrolls >= MAX_IDLE_SCROLLS:
            raise RuntimeError("No new transaction pages after repeated scrolling; "
                               "history may be incomplete. Check the browser and retry.")

        previous_pages = capture.pages
        # Move away from the bottom before returning to trigger infinite-scroll listeners.
        await page.evaluate("window.scrollTo(0, Math.max(0, document.body.scrollHeight - window.innerHeight - 500))")
        await page.mouse.wheel(0, 1000)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(SCROLL_DELAY_MS)
        await capture.drain()
        idle_scrolls = idle_scrolls + 1 if capture.pages == previous_pages else 0
        print(f"Captured {capture.pages} pages, {capture.transactions} transactions")


async def download_transactions(start_date=None, end_date=None):
    capture = HistoryCapture(start_date, end_date)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            page.on("response", capture.on_response)
            await page.goto(URL)

            print("Log in to Amazon if required.")
            await asyncio.to_thread(input, "Once transaction history is visible, press Enter...")
            await scroll_history(page, capture)
        finally:
            await capture.drain()
            await browser.close()
    if capture.errors:
        raise RuntimeError("A response could not be captured; history may be incomplete.")
    return capture


def parse_date(value):
    try:
        parsed = date.fromisoformat(value)
        if value != parsed.isoformat():
            raise ValueError
        return parsed
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be in YYYY-MM-DD format") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=parse_date, help="Inclusive start date (YYYY-MM-DD, India time)")
    parser.add_argument("--end-date", type=parse_date, help="Inclusive end date (YYYY-MM-DD, India time)")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Destination CSV path")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output CSV")
    args = parser.parse_args(argv)
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("--start-date must not be after --end-date")
    if args.output.exists() and not args.overwrite:
        parser.error("output already exists; choose another --output or use --overwrite")
    try:
        capture = asyncio.run(download_transactions(args.start_date, args.end_date))
        write_csv(capture.rows, args.output, overwrite=args.overwrite)
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"Download failed: {exc}\n")
    print(f"Exported {len(capture.rows)} transactions from {capture.pages} pages to {args.output}")


if __name__ == "__main__":
    main()