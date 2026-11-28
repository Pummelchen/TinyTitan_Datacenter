#!/usr/bin/env python3
"""Tests for the long-prompt ANE-vs-GPU correctness comparison.

The parts worth pinning are the ones that decide what the record means: the text
comparison itself, and the rule that an arm which fell back to the GPU is an
error rather than a comparison against the GPU path.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
_spec = importlib.util.spec_from_file_location(
    "ane_prefill_correctness",
    pathlib.Path(__file__).resolve().parent / "ane_prefill_correctness.py")
correctness = importlib.util.module_from_spec(_spec)
sys.modules["ane_prefill_correctness"] = correctness
_spec.loader.exec_module(correctness)


def arm(response: str, used_ane: bool = True, causal_warning: bool = False) -> dict:
    return {"arm": "x", "exit": 0, "response": response, "used_ane": used_ane,
            "causal_warning": causal_warning, "decode_tokens": 8,
            "prefill_seconds": 1.0, "finish": "length"}


class CommonPrefixTests(unittest.TestCase):
    def test_identical_strings_agree_to_the_end(self):
        self.assertEqual(correctness.common_prefix("abcdef", "abcdef"), 6)

    def test_the_prefix_stops_at_the_first_disagreement(self):
        self.assertEqual(correctness.common_prefix("abcdef", "abcXef"), 3)
        self.assertEqual(correctness.common_prefix("", "abc"), 0)

    def test_divergence_shows_both_sides(self):
        out = correctness.divergence("a" * 40 + "X" + "tail", "a" * 40 + "Y" + "tail", 40)
        self.assertEqual(out["at_character"], 40)
        self.assertIn("X", out["off"])
        self.assertIn("Y", out["arm"])


class SummarizeTests(unittest.TestCase):
    def test_a_folded_arm_that_tracks_the_gpu_is_not_an_error(self):
        row = correctness.summarize("m", {
            "off": arm("the same text", used_ane=False),
            "on": arm("the same text"),
        }, max_new=8)
        self.assertNotIn("error", row)
        self.assertTrue(row["comparisons"]["on"]["identical"])

    def test_an_arm_that_fell_back_is_an_error_not_a_comparison(self):
        # A fallback arm is a GPU-vs-GPU comparison wearing an ANE label.
        fallen = arm("the same text", used_ane=False)
        fallen["fallback_reason"] = "ane-prefill fallback: chunk outside coverage"
        row = correctness.summarize("m", {
            "off": arm("the same text", used_ane=False),
            "on": fallen,
        }, max_new=8)
        self.assertIn("fell back", row["error"])

    def test_a_causal_control_must_announce_itself_and_diverge(self):
        # The control is only a control if it ran the causal-only mask and came
        # out different; otherwise the check proves nothing about the fold.
        silent = correctness.summarize("m", {
            "off": arm("the same text", used_ane=False),
            "on": arm("the same text"),
            "causal": arm("the same text", causal_warning=False),
        }, max_new=8)
        self.assertIn("did not announce", silent["error"])

        agreeing = correctness.summarize("m", {
            "off": arm("the same text", used_ane=False),
            "on": arm("the same text"),
            "causal": arm("the same text", causal_warning=True),
        }, max_new=8)
        self.assertIn("warning", agreeing)
        self.assertNotIn("error", agreeing)

        diverging = correctness.summarize("m", {
            "off": arm("the same text", used_ane=False),
            "on": arm("the same text"),
            "causal": arm("the sa", causal_warning=True),
        }, max_new=8)
        self.assertNotIn("error", diverging)
        self.assertNotIn("warning", diverging)
        self.assertLess(diverging["comparisons"]["causal"]["common_prefix_characters"],
                        diverging["comparisons"]["on"]["common_prefix_characters"])


if __name__ == "__main__":
    unittest.main()
