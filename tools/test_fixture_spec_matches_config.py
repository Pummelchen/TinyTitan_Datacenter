#!/usr/bin/env python3
"""The fixture's spec must agree with the fixture's config, field by field.

Why this exists, in the order it happened. `DC-093` claimed the MoE fixture never exercises partial
RoPE; the fixture had carried `partial_rotary_factor` all along, and the claim came from a check that
looked the field up under `text_config` — where the **real** checkpoint nests its geometry — while the
fixture keeps those keys at the **top level**. A blind spot in a shell one-liner became a documented
"fact" about the repository, and the retraction cost a round.

The lesson is not "be careful": it is that this comparison deserves to be a tool with tests, like the
link gate and the table gate, instead of a lookup somebody types. And the first thing the tool has to
get right is the thing that fooled me — **both nestings are legal**, because a real checkpoint nests
and the fixtures do not.

What it asserts: for every field in `MAPPING`, the spec's camelCase value equals the config's
snake_case value, reading the config at the top level or under `text_config`, whichever is present.
A field missing from either side is a failure, not a skip: the point is to catch a **silent default**,
which is precisely the failure a missing field produces.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

#: `config.json` field -> `spec.json` field. Every pair here has been read off the real install's
#: `install.json` and the checkpoint's `config.json` and seen to agree, so this is a record of a
#: verified correspondence rather than a guess at one.
MAPPING = {
    "hidden_size": "hiddenSize",
    "head_dim": "headDim",
    "num_attention_heads": "numAttentionHeads",
    "num_key_value_heads": "numKeyValueHeads",
    "vocab_size": "vocabSize",
    "rms_norm_eps": "rmsNormEps",
    "num_experts": "numExperts",
    "num_experts_per_tok": "numExpertsPerToken",
    "moe_intermediate_size": "moeIntermediateSize",
    "shared_expert_intermediate_size": "sharedExpertIntermediateSize",
    "full_attention_interval": "fullAttentionInterval",
    "attn_output_gate": "attnOutputGate",
    "partial_rotary_factor": "partialRotaryFactor",
    "rope_theta": "ropeTheta",
    "linear_num_key_heads": "linearKeyHeads",
    "linear_num_value_heads": "linearValueHeads",
    "linear_value_head_dim": "linearValueHeadDim",
    "linear_conv_kernel_dim": "linearConvKernelDim",
}

FIXTURE = Path("tests/DatacenterEngineTests/Fixtures/tiny-qwen36")


def geometry_layers(config: dict) -> list[dict]:
    """Every place a config may keep the field, in the order the importer would find it.

    This function exists because **three** nestings are in play and my checks have now been wrong
    about two of them:

    - a real checkpoint (`Qwen/Qwen3.6-35B-A3B`) keeps text geometry under `text_config`, while the
      fixtures keep it at the top level;
    - the RoPE fields live under `rope_parameters` in **both** — which is what the importers read
      (`partialRotaryFactor: rope?.partial_rotary_factor`), and which the first version of this tool
      did not look in, so it reported `rope_theta` and `partial_rotary_factor` as absent from a config
      that has both.

    That second mistake was caught by this tool before it reached the tracker, which is the whole
    argument for writing it. The first one became `DC-093` and cost a round.
    """
    layers: list[dict] = [config]
    for source in [config, config.get("text_config")]:
        if isinstance(source, dict):
            if source is not config and source not in layers:
                layers.append(source)
            rope = source.get("rope_parameters")
            if isinstance(rope, dict) and rope not in layers:
                layers.append(rope)
    return layers


def find(config: dict, key: str):
    """The field's value from the first layer that has it, or a sentinel if none does."""
    for layer in geometry_layers(config):
        if key in layer:
            return True, layer[key]
    return False, None


def mismatches(config: dict, spec_config: dict, mapping: dict[str, str]) -> list[str]:
    """One message per field that is missing or different. Empty means they agree."""
    problems: list[str] = []
    for source, target in sorted(mapping.items()):
        present, want = find(config, source)
        if not present:
            problems.append(f"{source}: absent from the config")
            continue
        if target not in spec_config:
            problems.append(f"{target}: absent from the spec (a silent default would follow)")
            continue
        got = spec_config[target]
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            if abs(float(want) - float(got)) > 1e-12 * max(1.0, abs(float(want))):
                problems.append(f"{target}: spec has {got}, config has {want}")
        elif want != got:
            problems.append(f"{target}: spec has {got}, config has {want}")
    return problems


def check(fixture: Path = FIXTURE) -> list[str]:
    config = json.loads((fixture / "config.json").read_text(encoding="utf-8"))
    spec = json.loads((fixture / "spec.json").read_text(encoding="utf-8"))
    return mismatches(config, spec["config"], MAPPING)


class FixtureSpecMatchesConfigTests(unittest.TestCase):
    def test_the_qwen36_fixture_agrees_with_its_config(self):
        self.assertEqual(check(), [])

    def test_a_differing_value_is_reported(self):
        """The check has to be able to fail, or it is decoration."""
        problems = mismatches({"hidden_size": 2048}, {"hiddenSize": 4096}, {"hidden_size": "hiddenSize"})
        self.assertEqual(len(problems), 1)
        self.assertIn("4096", problems[0])

    def test_a_field_the_spec_lacks_is_reported(self):
        """A silent default is the failure this exists to catch, so absence is a failure."""
        # The config HAS the field and the spec does not -- which is the silent-default case. Putting
        # the field in neither is a different failure (absent from the config), and the first version
        # of this test conflated the two.
        problems = mismatches(
            {"rope_theta": 1e7, "partial_rotary_factor": 0.25},
            {"ropeTheta": 1e7},
            {"rope_theta": "ropeTheta", "partial_rotary_factor": "partialRotaryFactor"},
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("silent default", problems[0])

    def test_fields_under_rope_parameters_are_found(self):
        """The nesting that fooled this tool itself, and the real checkpoint uses it too."""
        config = {"rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25}}
        spec = {"ropeTheta": 1e7, "partialRotaryFactor": 0.25}
        self.assertEqual(
            mismatches(config, spec, {"rope_theta": "ropeTheta", "partial_rotary_factor": "partialRotaryFactor"}),
            [],
        )

    def test_a_nested_config_is_read(self):
        """The nesting that fooled the one-liner: `text_config` is where a real checkpoint keeps it."""
        nested = {"text_config": {"hidden_size": 2048, "partial_rotary_factor": 0.25}}
        spec = {"hiddenSize": 2048, "partialRotaryFactor": 0.25}
        self.assertEqual(mismatches(nested, spec, {"hidden_size": "hiddenSize", "partial_rotary_factor": "partialRotaryFactor"}), [])

    def test_a_bool_field_is_compared_by_value(self):
        self.assertEqual(mismatches({"attn_output_gate": True}, {"attnOutputGate": True}, {"attn_output_gate": "attnOutputGate"}), [])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        found = check()
        for problem in found:
            print(problem)
        print(f"{len(found)} field(s) disagree between the qwen36 fixture's config and its spec.")
        raise SystemExit(1 if found else 0)
    unittest.main()
