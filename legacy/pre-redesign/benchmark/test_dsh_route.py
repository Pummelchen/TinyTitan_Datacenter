"""The DeepSeek Harness route writer: one generated block, no hand-written YAML.

`tools/dsh_route.sh` turns the installs under `models/` into the `llm-pi-ai`
route the harness reads, so its model picker follows the catalog instead of a
list someone typed. These tests pin what matters: the ids and effort ladders it
derives, the three switches that are easy to get wrong by hand, the settings-file
surgery (replace the section, keep everything else, back it up), and its refusal
to describe nothing.

They run against `tools/testdata/catalog-example.json` through
`TINYTITAN_CATALOG_JSON`, so they need no model, no built server and no network.

Run from this directory, like the other benchmark tests:

    cd benchmark && python3 -m unittest test_dsh_route -v
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/dsh_route.sh"
CATALOG = ROOT / "tools/testdata/catalog-example.json"

try:
    import yaml
except ImportError:  # pragma: no cover - the parse check is a bonus
    yaml = None


def catalog_models() -> list[dict]:
    return json.loads(CATALOG.read_text())["models"]


def run_route(*args: str, catalog: pathlib.Path | None = CATALOG,
              expect: int = 0) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    if catalog is None:
        environment.pop("TINYTITAN_CATALOG_JSON", None)
    else:
        environment["TINYTITAN_CATALOG_JSON"] = str(catalog)
    environment.pop("TINYTITAN_MODELS_DIR", None)
    run = subprocess.run(["bash", str(SCRIPT), *args], text=True,
                         capture_output=True, check=False, env=environment)
    if expect is not None:
        assert run.returncode == expect, run.stderr
    return run


def declared_ids(block: str) -> list[str]:
    return re.findall(r"^        - id: (.+)$", block, re.MULTILINE)


def parsed_levels(model_block: str) -> dict[str, str | None]:
    """The reasoningEfforts map of one model, keyed as the harness reads it.

    Only the keys matter here: a level's *value* is its wire spelling, which the
    block writes as the level's own name (or `on` for a binary family). PyYAML
    is not used for this because it follows YAML 1.1, where a bare `off` key
    parses as the boolean false, while the harness's parser (YAML 1.2) reads the
    string `off`.
    """
    section = re.search(r"reasoningEfforts:\n((?:            .*\n)+)", model_block)
    assert section is not None, model_block
    keys = [line.strip().split(":")[0].strip('"') for line in section.group(1).splitlines()]
    return {key: None for key in keys}


class RouteBlockTests(unittest.TestCase):
    def test_declares_every_model_in_the_catalog(self) -> None:
        run = run_route()
        ids = declared_ids(run.stdout)
        self.assertEqual(ids, [model["id"] for model in catalog_models()])

    def test_effort_ladders_follow_each_template(self) -> None:
        run = run_route("--models", "qwen38")
        blocks = run.stdout.split("        - id: ")[1:]
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            self.assertEqual(sorted(parsed_levels(block)), ["low", "medium", "off", "xhigh"])
            self.assertIn("            xhigh: xhigh", block)
            self.assertNotIn("on:", block)

        # A binary-thinking family has one thinking mode: `off`, plus that mode
        # offered as `medium` with the wire value `on`, because pi-ai's level
        # vocabulary has no `on` of its own.
        binary = run_route("--models", "qwen3.6").stdout
        self.assertIn("            off:\n            medium: on", binary)
        self.assertNotIn("            on:", binary)

    def test_the_three_switches_and_the_usage_contract_are_set(self) -> None:
        block = run_route("--models", "qwen38").stdout
        self.assertIn("      baseURL: http://127.0.0.1:8080/v1", block)
        self.assertIn("        authorization: Bearer tinytitan-local", block)
        self.assertIn("      streamIdleTimeoutMs: 3600000", block)
        self.assertIn("            thinkingFormat: chat-template", block)
        self.assertIn("              enable_thinking: { $var: thinking.enabled }", block)
        self.assertIn("              reasoning_effort: { $var: thinking.effort }", block)
        self.assertIn("            maxTokensField: max_tokens", block)
        self.assertIn("            supportsUsageInStreaming: true", block)

    def test_options_reach_the_block(self) -> None:
        block = run_route("--models", "qwen38", "--port", "8096", "--reasoning", "off",
                          "--context", "131072", "--max-tokens", "4096").stdout
        self.assertIn("      baseURL: http://127.0.0.1:8096/v1", block)
        self.assertIn("      reasoning: off", block)
        self.assertIn("          contextWindow: 131072", block)
        self.assertIn("          maxTokens: 4096", block)

    def test_filters_match_ids_keys_and_families(self) -> None:
        by_key = declared_ids(run_route("--models", "qwen38").stdout)
        self.assertEqual(by_key, ["qwen3.8-flash-next_4-Bit", "qwen3.8-flash-next_8-Bit"])
        by_id = declared_ids(run_route("--models", "qwen3.5-4b_4-Bit").stdout)
        self.assertEqual(by_id, ["qwen3.5-4b_4-Bit"])
        by_family = declared_ids(run_route("--models", "qwen3_5_dense").stdout)
        self.assertEqual(len(by_family), 4)


class SettingsSurgeryTests(unittest.TestCase):
    def settings(self, body: str) -> pathlib.Path:
        directory = pathlib.Path(tempfile.mkdtemp(prefix="dsh-route-test-"))
        path = directory / "settings.yaml"
        path.write_text(body)
        return path

    def test_write_replaces_the_section_and_keeps_everything_else(self) -> None:
        path = self.settings(
            "ui-theme:\n  preference: dark\nagent-presets:\n  default: qwen38\n"
            "llm-pi-ai:\n  providers:\n    stale-route:\n      displayName: old\n")
        run = run_route("--models", "qwen38", "--write", "--settings", str(path))
        self.assertIn("replaced", run.stdout)
        text = path.read_text()
        self.assertIn("ui-theme:", text)
        self.assertIn("  preference: dark", text)
        self.assertIn("  default: qwen38", text)
        self.assertNotIn("stale-route", text)
        self.assertEqual(declared_ids(text), ["qwen3.8-flash-next_4-Bit",
                                             "qwen3.8-flash-next_8-Bit"])
        backups = list(path.parent.glob("settings.yaml.bak-*"))
        self.assertEqual(len(backups), 1)
        self.assertIn("stale-route", backups[0].read_text())

        if yaml is not None:
            parsed = yaml.safe_load(text)
            route = parsed["llm-pi-ai"]["providers"]["tinytitan"]
            self.assertEqual(route["api"], "openai-completions")
            self.assertEqual(route["baseURL"], "http://127.0.0.1:8080/v1")
            self.assertEqual(route["reasoning"], "medium")
            self.assertEqual(len(route["models"]), 2)

    def test_a_rewrite_replaces_the_previous_generated_header(self) -> None:
        # The plugin refreshes the route at every harness boot, and the header
        # sits above `llm-pi-ai:` — so a naive section replacement orphans it
        # and each refresh adds three more stale comment lines.
        path = self.settings("ui-theme:\n  preference: dark\n")
        run_route("--models", "qwen38", "--write", "--settings", str(path))
        first = path.read_text()
        run_route("--models", "qwen38", "--write", "--settings", str(path))
        second = path.read_text()
        self.assertEqual(second.count("# Generated by tools/dsh_route.sh"), 1)
        self.assertEqual(second.count("settings.yaml is hot-reloaded."), 1)
        self.assertEqual(first, second)

    def test_write_appends_when_there_is_no_section(self) -> None:
        path = self.settings("ui-theme:\n  preference: dark\n")
        run = run_route("--models", "qwen38", "--write", "--settings", str(path))
        self.assertIn("appended", run.stdout)
        text = path.read_text()
        self.assertTrue(text.startswith("ui-theme:\n  preference: dark\n"))
        self.assertIn("llm-pi-ai:", text)


class RefusalTests(unittest.TestCase):
    def test_bad_arguments_exit_two(self) -> None:
        run_route("--reasoning", "bogus", expect=2)
        run_route("--port", "eighty", expect=2)
        run_route("--models", "no-such-install", expect=2)

    def test_no_catalog_and_no_server_is_an_error(self) -> None:
        run_route("--from-server", "--port", "59999", expect=2)

    def test_write_without_a_settings_file_is_an_error(self) -> None:
        missing = pathlib.Path(tempfile.mkdtemp()) / "absent.yaml"
        run = run_route("--models", "qwen38", "--write", "--settings", str(missing), expect=2)
        self.assertIn("no DSH settings file", run.stderr)


if __name__ == "__main__":
    unittest.main()
