"""Unified command-line interface for the Expenses Tracker pipeline."""

from __future__ import annotations

import argparse
import os

from . import config
from .utils import LoggerFactory, get_log_file_path, set_logger


def _add_statement(command: argparse.ArgumentParser, with_bank: bool = False) -> None:
    command.add_argument(
        "--statement",
        choices=config.STATEMENTS,
        default="account",
        help="Statement type; selects the acc_* or cc_* tables.",
    )
    if with_bank:
        command.add_argument("--bank", choices=("axis", "icici", "all"), default=None, help="Defaults to axis for account, all for creditcard.")


def _resolve_bank(parser: argparse.ArgumentParser, statement: str, bank: str | None) -> str:
    bank = bank or ("all" if statement == "creditcard" else "axis")
    if bank != "all" and bank not in config.BANKS[statement]:
        parser.error(f"--bank {bank} is not supported for --statement {statement}")
    if statement == "account" and bank == "all":
        bank = "axis"
    return bank


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for name, description in (
        ("ingest", "Import statement CSVs and build curated transactions."),
        ("tokenize", "Tokenize curated transaction particulars."),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument(
            "--mode",
            choices=("incremental", "historical"),
            default="incremental",
            help="Incrementally append new transactions or rebuild the stage.",
        )
        _add_statement(command, with_bank=name == "ingest")

    canonicalize = commands.add_parser("canonicalize", help="Train mappings and normalize transaction sources.")
    canonicalize.add_argument("--no-train", action="store_true", help="Skip token-mapping training.")
    canonicalize.add_argument("--no-normalize", action="store_true", help="Skip source normalization.")
    canonicalize.add_argument("--incremental", action="store_true", help="Update existing mappings incrementally.")
    _add_statement(canonicalize)

    predict = commands.add_parser("predict", help="Predict category and subcategory labels.")
    predict.add_argument("--llm", action="store_true", help="Allow the configured local LLM to resolve low-confidence rows.")
    predict.add_argument("--no-write", action="store_true", help="Evaluate and report without writing results.")
    predict.add_argument("--skip-eval", action="store_true", help="Skip cross-validation evaluation.")
    _add_statement(predict)

    extract = commands.add_parser("extract", help="Extract raw credit card statements into CSV (offline).")
    extract.add_argument("--statement", choices=("creditcard",), required=True)
    extract.add_argument("--bank", choices=("axis", "icici"), required=True)

    embed = commands.add_parser("embed", help="Generate ChromaDB transaction and token embeddings.")
    _add_statement(embed)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    name = os.path.basename(__file__).removesuffix(".py")
    set_logger(LoggerFactory.get_logger(get_log_file_path(name), config.LOG_LEVEL))

    ingest_mode = config.HISTORICAL if getattr(args, "mode", None) == "historical" else config.INCREMENTAL

    if args.command == "ingest":
        bank = _resolve_bank(parser, args.statement, args.bank)
        if args.statement == "creditcard":
            from .process_cc_data import main as ingest_cc

            ingest_cc(ingest_mode, bank)
        else:
            from .process_input_data import main as ingest

            ingest(ingest_mode, bank)
    elif args.command == "tokenize":
        from .tokenization import main as tokenize

        tokenize(ingest_mode, args.statement)
    elif args.command == "canonicalize":
        from .canonicalizer import run_pipeline

        run_pipeline(
            train=not args.no_train,
            normalize=not args.no_normalize,
            incremental=args.incremental,
            statement=args.statement,
        )
    elif args.command == "predict":
        from .predict_category import main as predict

        predict(use_llm=args.llm, write=not args.no_write, skip_eval=args.skip_eval, statement=args.statement)
    elif args.command == "extract":
        from .statement_parsers import extract_statements

        extract_statements(args.statement, args.bank)
    elif args.command == "embed":
        from .vector_embedding import main as embed

        embed(args.statement)


if __name__ == "__main__":
    main()
