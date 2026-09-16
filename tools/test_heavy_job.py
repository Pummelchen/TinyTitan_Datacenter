"""Tests for `tools/heavy_job.py`.

The parsers are tested against **recorded** output from this machine, trailing periods and all: the first
version of these tests would have invented a tidier `vm_stat` than the tool prints, which is the mistake
that let a derivation in `check_baselines.py` pass its tests while crashing on real metrics.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_disk_headroom import HeadroomError  # noqa: E402
from heavy_job import (  # noqa: E402
    SYSTEM_RESERVE_GB,
    HeavyClaim,
    lock_path,
    claim_heavy,
    describe,
    parse_swapusage,
    parse_vm_stat,
    process_alive,
    require_memory_headroom,
    usable_gb,
)

# Copied from `vm_stat` and `sysctl -n vm.swapusage` on this node.
REAL_VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    58450.
Pages active:                                 169870.
Pages inactive:                               161248.
Pages speculative:                              7852.
Pages throttled:                                   0.
Pages wired down:                              75551.
Pages purgeable:                                1334.
"""
REAL_SWAPUSAGE = "total = 2048.00M  used = 1486.88M  free = 561.12M  (encrypted)"


class ParserTests(unittest.TestCase):
    def test_vm_stat_is_parsed_in_gb_using_its_own_page_size(self) -> None:
        pages = parse_vm_stat(REAL_VM_STAT)
        self.assertAlmostEqual(pages["free"], 58450 * 16384 / 1e9, places=6)
        self.assertAlmostEqual(pages["inactive"], 161248 * 16384 / 1e9, places=6)
        self.assertIn("wired_down", pages, "the label's spaces become underscores so it is addressable")
        self.assertNotIn("active", pages.get("page", ""), "no invented keys")

    def test_a_trailing_period_does_not_stop_the_parse(self) -> None:
        self.assertIn("free", parse_vm_stat(REAL_VM_STAT))

    def test_output_without_a_page_size_is_refused_rather_than_guessed(self) -> None:
        with self.assertRaises(ValueError):
            parse_vm_stat("Pages free: 100.\n")

    def test_swapusage_is_parsed_from_its_real_form(self) -> None:
        swap = parse_swapusage(REAL_SWAPUSAGE)
        self.assertAlmostEqual(swap["total"], 2.048, places=6)
        self.assertAlmostEqual(swap["used"], 1.48688, places=6)
        self.assertAlmostEqual(swap["free"], 0.56112, places=6)

    def test_a_gigabyte_value_is_not_read_as_a_megabyte(self) -> None:
        self.assertAlmostEqual(parse_swapusage("total = 1.00G  used = 0.50G")["total"], 1.0, places=6)


class UsableMemoryTests(unittest.TestCase):
    def test_usable_is_physical_less_the_measured_reserve(self) -> None:
        self.assertAlmostEqual(usable_gb(8.59), 8.59 - SYSTEM_RESERVE_GB, places=6)
        self.assertAlmostEqual(usable_gb(64.0), 60.5, places=6)

    def test_the_reserve_is_the_one_the_project_measured(self) -> None:
        """~4.5 GB usable of 8 GB is in the notes; a silent change to it would be a silent policy change."""
        self.assertAlmostEqual(SYSTEM_RESERVE_GB, 3.5, places=6)
        self.assertAlmostEqual(usable_gb(8.59), 5.09, places=2)


class RefusalTests(unittest.TestCase):
    def test_a_job_that_cannot_fit_is_refused_with_the_arithmetic(self) -> None:
        with self.assertRaises(HeadroomError) as caught:
            require_memory_headroom(1000.0, purpose="an impossible job")
        message = str(caught.exception)
        self.assertIn("an impossible job", message)
        self.assertIn("cannot make this safe", message)

    def test_a_job_that_fits_is_allowed_and_described(self) -> None:
        message = require_memory_headroom(0.1, purpose="a tiny job")
        self.assertIn("physical", message)
        self.assertIn("usable", message)

    def test_no_declared_need_only_reports(self) -> None:
        self.assertIn("physical", require_memory_headroom(None, purpose="an inspection"))

    def test_the_description_names_swap_when_it_is_known(self) -> None:
        described = describe({"physical_gb": 8.59, "reclaimable_gb": 1.0, "swap_used": 1.5, "swap_total": 2.0})
        self.assertIn("swap 1.50 of 2.00 GB used", described)


class LockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.lock = Path(self.directory.name) / "HEAVY_JOB_LOCK"

    def test_a_second_live_claim_is_refused(self) -> None:
        claim = claim_heavy("the first job", lock=self.lock, release_at_exit=False)
        self.addCleanup(claim.release)
        with self.assertRaises(HeadroomError) as caught:
            claim_heavy("the second job", lock=self.lock, release_at_exit=False)
        self.assertIn("the second job", str(caught.exception))
        self.assertIn("one heavy job at a time", str(caught.exception).lower())

    def test_releasing_lets_the_next_one_start(self) -> None:
        first = claim_heavy("the first job", lock=self.lock, release_at_exit=False)
        first.release()
        second = claim_heavy("the second job", lock=self.lock, release_at_exit=False)
        self.addCleanup(second.release)
        self.assertIn("the second job", self.lock.read_text())

    def test_a_claim_whose_process_is_gone_is_taken_over(self) -> None:
        """A lock a crashed job leaves behind must not refuse every future run — that is how a guard
        becomes a formality that the next person deletes."""
        # A pid that cannot be alive: this test's own process is, so use a very large one.
        self.lock.write_text("purpose=the crashed job pid=999999 at=2026-01-01T00:00:00\n")
        self.assertFalse(process_alive(999999), "pid 999999 should not exist")
        claim = claim_heavy("the new job", lock=self.lock, release_at_exit=False)
        self.addCleanup(claim.release)
        self.assertIn("the new job", self.lock.read_text())

    def test_release_leaves_another_holders_claim_alone(self) -> None:
        """A claim removes only its **own** record: releasing must not delete a live job's lock."""
        self.lock.write_text("purpose=someone else pid=1 at=2026-01-01T00:00:00\n")
        HeavyClaim(self.lock, "my job").release()
        self.assertTrue(self.lock.exists(), "a claim held by another process must survive our release")
        self.assertIn("someone else", self.lock.read_text())

    def test_process_alive_sees_this_process(self) -> None:
        import os

        self.assertTrue(process_alive(os.getpid()))


if __name__ == "__main__":
    unittest.main()


class LockLocationTests(unittest.TestCase):
    """The guard stays enabled when its location moves.

    The first version of the wiring put the claim inside `quantize.build_install` — a library function — so
    every test that built a two-tensor fixture took the **production** lock and refused the next one. The
    lock belongs to a deliberate run, and its location has to be redirectable without switching it off.
    """

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.previous = os.environ.get("HEAVY_JOB_LOCK")
        self.addCleanup(self.restore)

    def restore(self) -> None:
        if self.previous is None:
            os.environ.pop("HEAVY_JOB_LOCK", None)
        else:
            os.environ["HEAVY_JOB_LOCK"] = self.previous

    def test_the_default_is_the_repository_marker(self) -> None:
        os.environ.pop("HEAVY_JOB_LOCK", None)
        self.assertEqual(lock_path().name, "HEAVY_JOB_LOCK")
        self.assertIn(".build", str(lock_path()))

    def test_the_location_can_be_moved_by_the_environment(self) -> None:
        target = Path(self.directory.name) / "elsewhere.lock"
        os.environ["HEAVY_JOB_LOCK"] = str(target)
        self.assertEqual(lock_path(), target)

    def test_claiming_honours_the_moved_location_and_still_refuses_a_second_job(self) -> None:
        target = Path(self.directory.name) / "elsewhere.lock"
        os.environ["HEAVY_JOB_LOCK"] = str(target)
        first = claim_heavy("the first job", release_at_exit=False)
        self.addCleanup(first.release)
        self.assertTrue(target.exists(), "the claim must go where the environment says")
        with self.assertRaises(HeadroomError):
            claim_heavy("the second job", release_at_exit=False)
