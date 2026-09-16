"""Tests for the memory watchdog.

Two things are worth asserting and one is worth asserting carefully.

* The detector reads **resident** memory of processes whose command line names a heavy job, and it must not
  mistake the watchers themselves for work.
* `stop_them` really does stop a process — tested against a child this test owns, so the instrument is
  exercised rather than described.
* The `--once` form is a no-op when nothing is over the limit: no marker, no killing, exit 0.

What is deliberately *not* tested is the three-reading rule by sleeping through it: the rule lives in one
comparison in `main`, and a test that waits fifteen seconds to observe it would cost more than it checks. The
reading it protects is documented in `D54` and in `disk_watchdog.py`, which learned it the expensive way.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import memory_watchdog  # noqa: E402


class ResidentTests(unittest.TestCase):
    def test_finds_a_process_by_its_command_line(self):
        child = subprocess.Popen(["sleep", "30"])
        try:
            time.sleep(0.2)
            found = memory_watchdog.resident_gb(patterns=("sleep 30",))
            self.assertIn(child.pid, found)
            command, size = found[child.pid]
            self.assertIn("sleep 30", command)
            self.assertGreaterEqual(size, 0.0)
        finally:
            child.kill()
            child.wait()

    def test_ignores_the_watchers_themselves(self):
        # A line naming either watcher is not work, and a real heavy job still is. `ps` is faked rather
        # than fiddled with, so the assertion is about this module's rule and nothing else.
        listing = (
            "  111  2048 python3 tools/memory_watchdog.py --once\n"
            "  222  4096 python3 tools/disk_watchdog.py --interval 5\n"
            "  333  8192 python3 tools/quantize.py --install .build/m1-install\n"
        )
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=listing, stderr="")
        with unittest.mock.patch.object(memory_watchdog.subprocess, "run", return_value=fake):
            found = memory_watchdog.resident_gb()
        self.assertNotIn(111, found, "the memory watchdog is not a heavy job")
        self.assertNotIn(222, found, "the disk watchdog is not a heavy job")
        self.assertIn(333, found, "a real heavy job is still found")
        self.assertAlmostEqual(found[333][1], 8192 / 1048576, places=6)

    def test_offenders_respects_the_limit(self):
        child = subprocess.Popen(["sleep", "30"])
        try:
            time.sleep(0.2)
            over = memory_watchdog.offenders(limit_gb=0.0, patterns=("sleep 30",))
            self.assertIn(child.pid, over)
            under = memory_watchdog.offenders(limit_gb=1000.0, patterns=("sleep 30",))
            self.assertNotIn(child.pid, under)
        finally:
            child.kill()
            child.wait()


class StopTests(unittest.TestCase):
    def test_stop_them_kills_a_child(self):
        child = subprocess.Popen(["sleep", "60"])
        time.sleep(0.2)
        done = memory_watchdog.stop_them({child.pid: ("sleep 60", 0.01)}, grace=0.2)
        child.wait(timeout=10)
        self.assertIsNotNone(child.poll())
        self.assertTrue(any("SIGTERM" in line for line in done))
        self.assertTrue(any("SIGKILL" in line for line in done))

    def test_stop_them_tolerates_a_process_that_already_exited(self):
        child = subprocess.Popen(["sleep", "60"])
        child.kill()
        child.wait()
        done = memory_watchdog.stop_them({child.pid: ("sleep 60", 0.01)}, grace=0.05)
        self.assertEqual(done, [])


class OnceTests(unittest.TestCase):
    def test_once_writes_no_marker_when_nothing_is_over(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "MEMORY_STOP"
            code = memory_watchdog.main(
                ["--once", "--limit-gb", "1000", "--marker", str(marker)]
            )
            self.assertEqual(code, 0)
            self.assertFalse(marker.exists())

    def test_write_marker_records_the_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "deep" / "MEMORY_STOP"
            memory_watchdog.write_marker(marker, "MEMORY STOP: 2 heavy job(s) over 4.0 GB real")
            text = marker.read_text()
            self.assertIn("MEMORY STOP", text)
            self.assertIn("over 4.0 GB real", text)
            self.assertRegex(text, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ")


if __name__ == "__main__":
    unittest.main()
