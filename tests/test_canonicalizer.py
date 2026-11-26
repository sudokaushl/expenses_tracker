"""Tests for ambiguity-aware source canonicalization."""

import unittest

import numpy as np

from pipeline.canonicalizer import normalize_transaction_hybrid


class _SemanticModel:
    def encode(self, texts, **_kwargs):
        return np.array([[0.49, 0.51]], dtype=float)


class _CrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs, **_kwargs):
        self.pairs = pairs
        if callable(self.scores):
            return self.scores(pairs)
        return self.scores


class CanonicalizerTests(unittest.TestCase):
    def test_complete_compact_phrase_beats_shared_token_mapping(self):
        result = normalize_transaction_hybrid(
            ["mother", "hood"],
            model=None,
            token_to_canonical={"mother": "Mother Dairy"},
            src_names=["Mother Dairy", "Motherhood Hospital"],
            src_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )

        self.assertEqual(result, "Motherhood Hospital")

    def test_phrase_match_ignores_trailing_location_token(self):
        result = normalize_transaction_hybrid(
            ["mother", "hood", "noida"],
            model=None,
            token_to_canonical={"mother": "Mother Dairy"},
            src_names=["Mother Dairy", "Motherhood Hospital"],
            src_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )

        self.assertEqual(result, "Motherhood Hospital")

    def test_conflicting_rankings_are_cross_encoded_with_transaction_context(self):
        names = ["Mother Dairy", "Motherhood Hospital"]
        cross_encoder = _CrossEncoder(
            lambda pairs: [2.0 if "Motherhood Hospital" in pair[1] else 0.1 for pair in pairs]
        )

        result = normalize_transaction_hybrid(
            ["mother", "clinic"],
            model=_SemanticModel(),
            token_to_canonical={"mother": "Mother Dairy"},
            src_names=names,
            src_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
            context="MOTHER HOOD CLINIC POS PURCHASE",
            cross_encoder=cross_encoder,
        )

        self.assertEqual(result, "Motherhood Hospital")
        self.assertTrue(cross_encoder.pairs)
        self.assertIn("MOTHER HOOD CLINIC POS PURCHASE", cross_encoder.pairs[0][0])

    def test_close_cross_encoder_scores_leave_source_unmatched(self):
        result = normalize_transaction_hybrid(
            ["mother", "clinic"],
            model=_SemanticModel(),
            token_to_canonical={"mother": "Mother Dairy"},
            src_names=["Mother Dairy", "Motherhood Hospital"],
            src_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
            cross_encoder=_CrossEncoder([0.3, 0.1]),
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()