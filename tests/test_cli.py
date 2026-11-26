"""Tests for the public command-line interface."""

import unittest

from pipeline.cli import _build_parser


class CommandLineTests(unittest.TestCase):
    def test_ingest_and_tokenize_accept_supported_modes(self):
        parser = _build_parser()

        ingest = parser.parse_args(["ingest", "--mode", "historical"])
        tokenize = parser.parse_args(["tokenize", "--mode", "incremental"])

        self.assertEqual((ingest.command, ingest.mode), ("ingest", "historical"))
        self.assertEqual((tokenize.command, tokenize.mode), ("tokenize", "incremental"))

    def test_statement_flag_defaults_to_account_and_selects_tables(self):
        from pipeline import config

        parser = _build_parser()
        self.assertEqual(parser.parse_args(["tokenize"]).statement, "account")
        for command in ("tokenize", "canonicalize", "predict", "embed"):
            self.assertEqual(parser.parse_args([command, "--statement", "creditcard"]).statement, "creditcard")
        ingest = parser.parse_args(["ingest", "--statement", "creditcard", "--bank", "icici"])
        self.assertEqual((ingest.statement, ingest.bank), ("creditcard", "icici"))
        self.assertEqual(config.tables("account").tokens, "acc_tokenized")
        self.assertEqual(config.tables("creditcard").categorized, "cc_categorized")

    def test_predict_options_are_forwarded_as_cli_flags(self):
        args = _build_parser().parse_args(["predict", "--llm", "--no-write", "--skip-eval"])

        self.assertTrue(args.llm)
        self.assertTrue(args.no_write)
        self.assertTrue(args.skip_eval)


if __name__ == "__main__":
    unittest.main()
