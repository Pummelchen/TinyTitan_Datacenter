"""The contract's reader, pointed at an install instead of a checkpoint.

M1's gate says the engine reproduces the contract. It was checked the other way round: the engine read an
**install** while the contract read the **checkpoint**, and `D55` showed that the entire divergence — 24.5%
median relative at layer 0, and forty flipped decisions after it — was that difference in weights and not a
difference in arithmetic. A claim of that shape cannot be tested against two different inputs.

This closes the gap. It exposes the same two methods `ordered_qwen36.streamed_text_forward` asks a source for —
`tensor(name)` and `rows(name, start, end)` — and fills them from the install, through the **same dequantiser
the Swift reader mirrors**. So an install-backed contract run and an install-backed engine run differ only if
the arithmetic differs, which is what the gate is actually about.

Two deliberate refusals:

* a tensor of more than two dimensions is one of the routed expert stacks, whose dequantised form is
  gigabytes; the streaming path exists for those and the contract uses it (`stream_experts=True`);
* an absent name is an error rather than a default, the same rule the install itself follows for a role that
  is missing from the quantisation policy.

Rows are read through `Install.row_range`, so fetching one expert costs one expert's rows and not the rows in
front of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from install_reader import Install, Tensor


class InstallSourceError(Exception):
    """A name the install does not hold, or a tensor this source will not materialise."""


class InstallSource:
    """`tensor(name)` and `rows(name, start, end)` over an install, in fp32."""

    def __init__(self, root: Path, uncached: bool = True) -> None:
        self.install = Install(Path(root), uncached=uncached)

    def __enter__(self) -> "InstallSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.install.close()

    def _tensor(self, name: str) -> Tensor:
        tensor = self.install.tensors.get(name)
        if tensor is None:
            raise InstallSourceError(f"{name} is not in this install")
        return tensor

    @staticmethod
    def _width(tensor: Tensor) -> int:
        """The row width: every trailing dimension flattened, because `rows()` is per leading index.

        `shape[-1]` is right for a matrix and wrong for the convolution's `(8192, 1, 4)`, whose row is four
        values and not one.
        """
        shape = tuple(tensor.shape or ())
        return int(np.prod(shape[1:])) if len(shape) > 1 else 1

    @staticmethod
    def _is_stack(tensor: Tensor) -> bool:
        """The routed expert stacks, which are the tensors that must never be materialised whole."""
        return tensor.role.startswith("expert.stack")

    def _rows_per_index(self, tensor: Tensor) -> int:
        """How many of the install's rows make one of the checkpoint's rows.

        An install flattens an expert's trailing dimensions into the row length: `expert.stack_gate_up` is
        **262,144 rows of 2,048**, not 256 rows of two million, so one expert is **1,024 consecutive install
        rows**. `rows()` is asked in the checkpoint's own index space — one expert is one row there — so the
        two have to be mapped, which is the thing this source exists to get right.
        """
        _rows_total, padded = tensor.geometry()
        return max(self._width(tensor) // max(padded, 1), 1)

    def _materialise(self, tensor: Tensor, start: int, end: int, row_block: int) -> np.ndarray:
        """Rows `[start, end)` in the checkpoint's index space, in fp32, flat; the callers shape it."""
        shape = tuple(tensor.shape or ())
        _rows_total, padded = tensor.geometry()
        columns = int(shape[-1]) if len(shape) > 1 else padded
        factor = self._rows_per_index(tensor)
        values: list[float] = []
        for row in self.install.row_range(tensor, start * factor, end * factor, row_block=row_block):
            values.extend(row[:columns])
        return np.asarray(values, dtype=np.float32)

    def tensor(self, name: str) -> np.ndarray:
        """The whole tensor in fp32. Refuses the expert stacks, which are not whole-tensor tensors."""
        tensor = self._tensor(name)
        shape = tuple(tensor.shape or ())
        if self._is_stack(tensor):
            raise InstallSourceError(
                f"{name} is an expert stack of shape {shape}; those are fetched by index with rows(), "
                "which is what `stream_experts=True` uses. Materialising one is gigabytes."
            )
        # A one-dimensional tensor is one install row of its width, which is what `geometry()` reports and
        # what the first version of this assumed was its length in rows of one.
        leading = 1 if len(shape) == 1 else int(shape[0])
        return self._materialise(tensor, 0, leading, row_block=64).reshape(shape)

    def rows(self, name: str, start: int, end: int) -> np.ndarray:
        """Rows `[start, end)` in the checkpoint's index space — for an expert stack, one expert per row."""
        tensor = self._tensor(name)
        shape = tuple(tensor.shape or ())
        if len(shape) < 2:
            raise InstallSourceError(
                f"{name} has shape {shape}; rows() is for a matrix or a stacked tensor, and the contract "
                "asks for a whole tensor with tensor()"
            )
        return self._materialise(tensor, start, end, row_block=8).reshape((end - start,) + shape[1:])

    def expert_provider(self, name: str):
        """The factory `mixer_weights` wants: one expert is one row of the stacked tensor."""
        def fetch(expert: int) -> np.ndarray:
            return self.rows(name, expert, expert + 1)[0]
        return fetch
