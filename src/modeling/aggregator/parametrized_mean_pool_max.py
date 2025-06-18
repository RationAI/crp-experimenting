from typing import Any

import torch
from rationai.masks import HeatmapAssembler
from rationai.mlkit.metrics.aggregators import MeanPoolMaxAggregator
from torch import Tensor


class ParametrizedMeanPoolMaxAggregator:
    """Aggregator to compute the max of predictions after average pooling, parameterized by kernel size.

    TODO: It is a temporary solution. Uses the `MeanPoolMaxAggregator` class
    from the `rationai` library. This should be replaced with a better implementation
    in `rationai` library in the future.

    Problems with `MeanPoolMaxAggregator`:
    1. It is a Pytorch `Metric` class, which caches output after first `compute` call.
    2. It is not possible to set the `kernel_size` during `compute` call.
    """

    def __init__(
        self,
        extent_tile: int,
        stride: int,
    ) -> None:
        """Initialize the aggregator.

        Arguments:
            extent_tile (int): Size of the tile.
            stride (int): Stride of the tiles.
        """
        super().__init__()

        self.aggregator = MeanPoolMaxAggregator(  # Aggregator to be copied
            kernel_size=1,  # To be set before computation
            extent_tile=extent_tile,
            stride=stride,
        )

        self.extent_tile = extent_tile
        self.stride = stride

        self._init_computed_state()

    def _init_computed_state(self) -> None:
        """Initialize the computed state variables."""
        self._computed = False  # First call to compute
        self.extent_x: int | None = None
        self.extent_y: int | None = None
        self.assembler: HeatmapAssembler | None = None

    def update(self, preds: Tensor, targets: Tensor, **kwargs: Any) -> None:
        self.aggregator.update(
            preds=preds,
            targets=targets,
            **kwargs,
        )

    def _set_extents(self) -> None:
        self.extent_x = max(x + self.extent_tile for x in self.aggregator.xs)
        self.extent_y = max(y + self.extent_tile for y in self.aggregator.ys)

    def _set_first_compute(self) -> None:
        self._computed = True
        self._set_extents()

        # Initialize the assembler and update
        self.assembler = HeatmapAssembler(
            self.extent_x,
            self.extent_y,
            self.extent_tile,
            self.extent_tile,
            self.stride,
            self.stride,
            device=self.aggregator.preds[0].device if self.aggregator.preds else "cpu",
        )

        self.assembler.update(
            torch.cat(self.aggregator.preds),
            torch.stack(self.aggregator.xs),
            torch.stack(self.aggregator.ys),
        )

    def compute(self, kernel_size: int) -> tuple[Tensor, Tensor]:
        """Compute the aggregated value.

        Arguments:
            kernel_size (int): Size of the pooling kernel.

        Returns:
            tuple[Tensor, Tensor]: Aggregated value and target value.
        """
        if not self._computed:
            self._set_first_compute()

        pool = torch.nn.AvgPool2d(kernel_size, stride=1)

        return (
            pool(self.assembler.compute().unsqueeze(0).unsqueeze(0)).max(),
            torch.stack(self.aggregator.targets).max(),
        )

    def reset(self) -> None:
        """Reset the aggregator."""
        self.aggregator.reset()
        self._init_computed_state()
