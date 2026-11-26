"""Focused tests for category prediction preprocessing and inference."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

import duckdb
import numpy as np
import pandas as pd

from pipeline import config
from pipeline.predict_category import Featurizer, evaluate, fit_models, load_data, main, predict, prep


class CategoryPredictionTests(unittest.TestCase):
    def setUp(self):
        self.frame = prep(pd.DataFrame(
            {
                "txt": ["coffee cafe", "restaurant meal", "bus ticket", "train travel"],
                "trans_mode": ["upi", "pos", "card", "upi"],
                "trans_type": ["debit"] * 4,
                "normalized_trans_src": ["cafe", "restaurant", "bus", "rail"],
                "trans_amt": [150.0, 400.0, 50.0, 120.0],
                "category": ["Food", "Food", "Travel", "Travel"],
                "subcategory": ["Cafe", "Dining", "Bus", "Rail"],
            }
        ))

    def test_prep_normalizes_source_and_transaction_text(self):
        frame = self.frame.assign(
            normalized_trans_src=["[AI] Cafe & Co", None, "City Bus", "Rail"]
        )
        prepared = prep(frame)

        self.assertEqual(prepared.loc[0, "src"], "cafe_co")
        self.assertEqual(prepared.loc[1, "src"], "none")
        self.assertEqual(prepared.loc[0, "txt"], "coffee cafe")

    def test_model_predicts_labels_and_bounded_confidences(self):
        models = fit_models(self.frame)
        candidates = self.frame.drop(columns=["category", "subcategory"]).copy()
        candidates.loc[0, "trans_mode"] = "new-mode"

        predictions, top_candidates, classes = predict(models, candidates)

        self.assertEqual(len(predictions), len(candidates))
        self.assertEqual(set(predictions.columns), {
            "category", "subcategory", "category_conf", "subcategory_conf"
        })
        self.assertTrue(predictions["category_conf"].between(0, 1).all())
        self.assertTrue(predictions["subcategory_conf"].between(0, 1).all())
        self.assertEqual(top_candidates.shape[0], len(candidates))
        self.assertGreater(len(classes), 0)

    def test_featurizer_handles_unseen_categorical_values(self):
        featurizer = Featurizer()
        featurizer.fit_transform(self.frame)
        unseen = self.frame.iloc[[0]].copy()
        unseen.loc[:, "trans_mode"] = "unknown-mode"

        features = featurizer.transform(unseen)

        self.assertEqual(features.shape[0], 1)

    def test_evaluation_reports_macro_metrics_and_confidence_calibration(self):
        labelled = pd.concat([self.frame] * 3, ignore_index=True)
        output = StringIO()

        with redirect_stdout(output):
            evaluate(labelled, n_splits=2)

        self.assertIn("macro precision/recall/F1", output.getvalue())
        self.assertIn("Brier score", output.getvalue())

    def test_load_data_uses_statement_specific_label_table(self):
        normalized = pd.DataFrame(columns=[
            "trans_id", "trans_dt", "particulars", "trans_amt", "trans_type",
            "trans_mode", "normalized_trans_src", "txt",
        ])
        labels = pd.DataFrame(columns=["trans_id", "labelled_trans_src", "category", "subcategory"])

        for statement, expected_table, expected_normalized in (
            ("account", "expenses.acc_labelled", "expenses.acc_normalized"),
            ("creditcard", "expenses.cc_labelled", "expenses.cc_normalized"),
        ):
            with self.subTest(statement=statement):
                con = Mock()
                con.query.side_effect = [
                    Mock(to_df=Mock(return_value=normalized.copy())),
                    Mock(to_df=Mock(return_value=labels.copy())),
                ]

                load_data(con, statement)

                normalized_query, labels_query = [call.args[0] for call in con.query.call_args_list]
                self.assertIn(expected_normalized, normalized_query)
                self.assertIn(expected_table, labels_query)

    def test_load_data_matches_comma_formatted_label_amount(self):
        con = duckdb.connect(":memory:")
        con.execute("CREATE SCHEMA expenses")
        con.execute("""
            CREATE TABLE expenses.acc_normalized (
                trans_id VARCHAR, trans_dt DATE, particulars VARCHAR, trans_amt DOUBLE,
                trans_type VARCHAR, trans_mode VARCHAR, normalized_trans_src VARCHAR,
                tokenized VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE expenses.acc_labelled (
                trans_id VARCHAR, trans_dt DATE, trans_amt VARCHAR, particulars VARCHAR,
                normalized_trans_src VARCHAR, category VARCHAR, subcategory VARCHAR
            )
        """)
        con.execute("""
            INSERT INTO expenses.acc_normalized VALUES
            ('pipeline-id', DATE '2026-01-02', 'MOTHER HOOD NOIDA', 1000.0,
             'debit', 'pos', 'Motherhood Hospital', '[mother, hood, noida]')
        """)
        con.execute("""
            INSERT INTO expenses.acc_labelled VALUES
            ('manual-id', DATE '2026-01-02', '1,000.00', 'mother hood noida',
             'Motherhood Hospital', 'Health', 'Clinic')
        """)

        result = load_data(con, "account")

        self.assertEqual(result.loc[0, "category"], "Health")
        self.assertEqual(result.loc[0, "subcategory"], "Clinic")
        con.close()

    def test_manual_ground_truth_is_preserved_in_categorized_table_for_each_statement(self):
        for statement in config.STATEMENTS:
            with self.subTest(statement=statement):
                con = duckdb.connect(":memory:")
                con.execute("CREATE SCHEMA expenses")
                normalized_table = f"expenses.{config.tables(statement).normalized}"
                labelled_table = f"expenses.{config.labelled_table(statement)}"
                categorized_table = f"expenses.{config.tables(statement).categorized}"
                con.execute(f"""
                    CREATE TABLE {normalized_table} (
                        trans_id VARCHAR, trans_dt DATE, particulars VARCHAR, trans_amt DOUBLE,
                        trans_type VARCHAR, trans_mode VARCHAR, normalized_trans_src VARCHAR,
                        tokenized VARCHAR
                    )
                """)
                con.execute(f"""
                    CREATE TABLE {labelled_table} (
                        trans_id VARCHAR, trans_dt DATE, trans_amt VARCHAR, particulars VARCHAR,
                        normalized_trans_src VARCHAR, category VARCHAR, subcategory VARCHAR
                    )
                """)

                normalized_rows = []
                labelled_rows = []
                for index in range(10):
                    particulars = f"MANUAL MERCHANT {index}"
                    normalized_rows.append((
                        f"{statement}-pipeline-{index}", "2026-01-02", particulars, 1000.0,
                        "debit", "pos", "Model Merchant", "[manual, merchant]",
                    ))
                    labelled_rows.append((
                        f"{statement}-manual-{index}", "2026-01-02", "1,000.00",
                        particulars.lower(), "Ground Truth Merchant", "Manual Category", "Manual Subcategory",
                    ))
                con.executemany(f"INSERT INTO {normalized_table} VALUES (?, ?, ?, ?, ?, ?, ?, ?)", normalized_rows)
                con.executemany(f"INSERT INTO {labelled_table} VALUES (?, ?, ?, ?, ?, ?, ?)", labelled_rows)

                def wrong_model_predictions(_models, frame):
                    size = len(frame)
                    predictions = pd.DataFrame({
                        "category": ["Wrong Category"] * size,
                        "subcategory": ["Wrong Subcategory"] * size,
                        "category_conf": [0.2] * size,
                        "subcategory_conf": [0.2] * size,
                    })
                    return predictions, np.zeros((size, 1), dtype=int), np.array(["Wrong Category | Wrong Subcategory"])

                with patch("pipeline.predict_category.fit_models", return_value=None), \
                     patch("pipeline.predict_category.predict", side_effect=wrong_model_predictions), \
                     redirect_stdout(StringIO()):
                    output = main(
                        use_llm=False,
                        write=True,
                        skip_eval=True,
                        con=con,
                        statement=statement,
                    )

                self.assertEqual(len(output), 10)
                self.assertTrue((output["label_source"] == "manual").all())
                self.assertTrue((output["normalized_trans_src"] == "Ground Truth Merchant").all())
                self.assertTrue((output["category"] == "Manual Category").all())
                self.assertTrue((output["subcategory"] == "Manual Subcategory").all())
                categorized = con.execute(
                    f"SELECT normalized_trans_src, category, subcategory FROM {categorized_table}"
                ).df()
                self.assertTrue((categorized["normalized_trans_src"] == "Ground Truth Merchant").all())
                self.assertTrue((categorized["category"] == "Manual Category").all())
                self.assertTrue((categorized["subcategory"] == "Manual Subcategory").all())
                con.close()


if __name__ == "__main__":
    unittest.main()
