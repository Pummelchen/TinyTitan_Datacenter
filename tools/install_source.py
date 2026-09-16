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
        """The **last axis**, which is the install's row content before padding.

        This docstring used to say "every trailing dimension flattened, because `rows()` is per leading
        index", and that was wrong in the way that matters: it made `_width` disagree with what the
        quantiser writes. `int4Layout` in `Install.swift` takes `rows = prod(shape.dropLast())` and
        `columns = shape.last`, and `install.json` records the resulting `nbytes` -- for the tiny
        `expert.stack_gate_up`, 9,472 bytes, which is 256 rows of 32 padded to 64 and not 128 rows of 64.
        The two only coincide when `shape.last == padded_columns`, which is true for the real model (2,048
        either way) and false for the fixture, so the error was invisible on the model it was written for.
        """
        shape = tuple(tensor.shape or ())
        return int(shape[-1]) if len(shape) > 1 else 1

    @staticmethod
    def _is_stack(tensor: Tensor) -> bool:
        """The routed expert stacks, which are the tensors that must never be materialised whole."""
        return tensor.role.startswith("expert.stack")

    def _rows_per_index(self, tensor: Tensor) -> int:
        """How many of the install's rows make one of the checkpoint's rows.

        An install keeps the leading axis as the index and drops the last one into the row: for a stack
        `(experts, rows, columns)` it is `experts * rows` install rows of `columns` (padded), so **one expert
        is the product of everything between the first and last axis** — 32 for the fixture's `(8, 32, 32)`
        and 2,048 for the real model's `(256, 2,048, 512)`. `rows()` is asked in the checkpoint's own index
        space, where one expert is one row, so the two have to be mapped.

        The first version divided the flattened trailing width by the padded row and so returned 16 where the
        truth is 32, which is exactly half, and it returned the right answer for the real model by
        coincidence: there `shape.last` and `padded_columns` are both 512, so the two formulas agree.
        """
        shape = tuple(tensor.shape or ())
        if len(shape) <= 2:
            return 1
        return max(int(np.prod(shape[1:-1])), 1)

    def _materialise(self, tensor: Tensor, start: int, end: int, row_block: int) -> np.ndarray:
        """Rows `[start, end)` in the checkpoint's index space, in fp32, flat; the callers shape it."""
        shape = tuple(tensor.shape or ())
        _rows_total, padded = tensor.geometry()
        # The row's content is the last axis, which is what the quantiser padded up to `padded`: a padded
        # row holds `shape[-1]` real values followed by padding. A one-dimensional tensor is one value per
        # row, so there `padded` is the row.
        columns = padded if len(shape) <= 1 else self._width(tensor)
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
