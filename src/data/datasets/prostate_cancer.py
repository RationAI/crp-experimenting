import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from openslide import OpenSlideUnsupportedFormatError
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from src.typing_ import Metadata, PredictSample, Sample


logger = logging.getLogger(__name__)


class ProstateCancer(MetaTiledSlides[Sample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[Sample]]:
        return (
            _ProstateCancerSlideTiles(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                include_label=True,
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )


class ProstateCancerPredict(MetaTiledSlides[PredictSample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[PredictSample]]:
        return (
            _ProstateCancerSlideTiles(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                include_label=False,
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )


class _ProstateCancerSlideTiles(Dataset[Sample | PredictSample]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        include_label: bool,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__()

        # Filter negative tiles in positive slides
        if include_label and int(slide_metadata.cancer) == 1:
            tiles = tiles[tiles["cancer"] == 1]

        self.slide_tiles = OpenSlideTilesDataset(
            slide_path=slide_metadata.path,
            level=slide_metadata.level,
            tile_extent_x=slide_metadata.tile_extent_x,
            tile_extent_y=slide_metadata.tile_extent_y,
            tiles=tiles,
        )
        self.transforms = transforms
        self.include_label = include_label
        self.to_tensor = ToTensorV2()

        if not self._check_slide_exists(slide_metadata.path):
            # Clear tiles if slide does not exist
            # This is necessary to ensure that the program does not crash when
            # the dataset is used in a DataLoader
            self.slide_tiles.tiles = pd.DataFrame()

    def _check_slide_exists(self, slide_path: str) -> bool:
        """Check if the slide exists. If not, log a warning and return False."""
        slide_path = Path(slide_path)
        if not slide_path.exists():
            logger.warning(f"Path {slide_path} does not exist.")
            return False

        # Check whether the slide can be opened
        if len(self.slide_tiles) > 0:
            try:
                self.slide_tiles[0]
            except OpenSlideUnsupportedFormatError:
                logger.warning(f"Slide {slide_path} is not supported.")
                return False

        return True

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> Sample | PredictSample:
        image = self.slide_tiles[idx]
        metadata = Metadata(
            slide=Path(self.slide_tiles.slide_path).stem,
            x=self.slide_tiles.tiles.iloc[idx]["x"],
            y=self.slide_tiles.tiles.iloc[idx]["y"],
        )

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        image = self.to_tensor(image=image)["image"]

        if self.include_label:
            label = torch.tensor([self.slide_tiles.tiles.iloc[idx]["cancer"]]).float()
            return image, label, metadata

        return image, metadata
