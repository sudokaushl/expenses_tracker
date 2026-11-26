"""Tests for machine-readable pipeline log records."""

import json
import logging
import unittest

from pipeline.utils import LoggerFactory


class StructuredLoggingTests(unittest.TestCase):
    def test_file_formatter_emits_parseable_context(self):
        record = logging.LogRecord(
            name="expenses_tracker", level=logging.WARNING, pathname="test.py",
            lineno=12, msg="missing transaction source: %s", args=("UPI",), exc_info=None,
        )

        entry = json.loads(LoggerFactory.JsonFormatter().format(record))

        self.assertEqual(entry["level"], "WARNING")
        self.assertEqual(entry["logger"], "expenses_tracker")
        self.assertEqual(entry["message"], "missing transaction source: UPI")
        self.assertIn("timestamp", entry)
        self.assertIn("function", entry)


if __name__ == "__main__":
    unittest.main()
