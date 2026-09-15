#!/usr/bin/env python3
"""The disk watchdog's *decisions*, tested without signalling anything.

The watchdog stops real processes, so a test that let it act would be a test that kills the
machine's work. These patch the two edges instead — the free-space reading and the stop itself —
and assert what the watchdog decided, which is the part that can be wrong.

The behaviour worth pinning is the debounce. Its first version acted on a single reading of
4.67 GB that was 17.55 GB six seconds later, because APFS "purgeable" space appears and
disappears as the system reclaims caches; it killed a read-only verification that had done
nothing wrong. A floor worth enforcing is worth confirming first.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import disk_watchdog


class WatchdogTests(unittest.TestCase):
    def run_watchdog(self, available: float, consecutive: int, marker: Path):
        """Run one check with the reading faked and the kill replaced by a recorder."""
        stopped: list[list] = []

        def record(processes, grace):
            stopped.append(list(processes))
            return [f"would stop {len(processes)}"]

        with mock.patch.object(disk_watchdog, "free_gb", return_value=available), mock.patch.object(
            disk_watchdog, "stop_them", side_effect=record
        ):
            code = disk_watchdog.main(
                [
                    "--once",
                    "--threshold-gb", "5",
                    "--consecutive", str(consecutive),
                    "--marker", str(marker),
                ]
            )
        return code, stopped

    def test_a_single_low_reading_does_not_stop_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "DISK_STOP"
            code, stopped = self.run_watchdog(4.6, 3, marker)
            self.assertEqual(code, 0)
            self.assertFalse(marker.exists(), "one transient reading must not trip the floor")
            self.assertEqual(stopped, [])

    def test_confirmed_low_readings_stop_the_jobs_and_write_the_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "DISK_STOP"
            code, stopped = self.run_watchdog(4.6, 1, marker)
            self.assertEqual(code, 0)
            self.assertEqual(len(stopped), 1, "the confirmed reading must stop the jobs")
            self.assertTrue(marker.exists(), "the marker must exist so new jobs refuse to start")
            self.assertIn("below 5.0 GB", marker.read_text())

    def test_ample_space_writes_no_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "DISK_STOP"
            code, stopped = self.run_watchdog(40.0, 1, marker)
            self.assertEqual(code, 0)
            self.assertFalse(marker.exists())
            self.assertEqual(stopped, [])

    def test_it_never_matches_itself_or_the_harness(self):
        """A command line containing this file must be excluded, or the watchdog kills itself."""
        listing = (
            "  101 /usr/bin/python3 tools/disk_watchdog.py --threshold-gb 5\n"
            "  102 /usr/bin/python3 -m unittest discover -s tools\n"
            "  103 .build/release/datacenter-trace /some/snapshot out 1,2,3\n"
            "  104 /usr/bin/python3 tools/quantize.py build /snap /out --spec s.json\n"
        )
        with mock.patch.object(
            disk_watchdog.subprocess, "run", return_value=mock.Mock(stdout=listing)
        ):
            found = disk_watchdog.heavy_processes({999})
        pids = [pid for pid, _ in found]
        self.assertNotIn(101, pids, "the watchdog must never stop itself")
        self.assertNotIn(102, pids, "the agent harness is a Python process too, and not heavy")
        self.assertEqual(pids, [103, 104], "the engine and the install builder are heavy")


if __name__ == "__main__":
    unittest.main()
