#!/usr/bin/env python3
"""The disk floor, tested where it is cheap to test.

This guard exists because of a specific failure: on 2026-09-16 this node ran a 35 B engine and a
20 GB install build concurrently, exhausted disk, then swap, and the hardware watchdog panicked
the machine twice. The guard's job is to make that a refusal rather than an incident, so what is
worth testing is exactly the refusal — including the case where there is room but the watchdog has
already stopped a run, because resuming by itself is the failure mode a disk check alone misses.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from check_disk_headroom import free_gb, require_headroom


class DiskHeadroomTests(unittest.TestCase):
    def test_it_allows_a_job_when_there_is_room(self):
        require_headroom(0.001, purpose="a test")  # must not raise

    def test_it_refuses_a_job_below_the_floor(self):
        with self.assertRaises(SystemExit) as raised:
            require_headroom(1e6, purpose="a heavy test")
        message = str(raised.exception)
        self.assertIn("refusing to start a heavy test", message)
        self.assertIn("GB free", message)
        # The message must say *why* the floor exists, or the next person raises it casually.
        self.assertIn("panic", message.lower())

    def test_a_stop_marker_refuses_even_with_room(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "DISK_STOP"
            marker.write_text("stopped a run\n")
            with self.assertRaises(SystemExit) as raised:
                require_headroom(0.001, purpose="a resumed test", marker=marker)
            message = str(raised.exception)
            self.assertIn("disk watchdog stopped a run", message)
            # And it must not clear itself: the operator clears it, deliberately.
            self.assertIn(f"rm {marker}", message)
            self.assertTrue(marker.exists(), "the guard must not clear the marker it refuses on")

    def test_free_space_is_read_from_the_filesystem(self):
        available = free_gb()
        self.assertGreater(available, 0.0)
        self.assertLess(available, 100_000.0)


if __name__ == "__main__":
    unittest.main()
