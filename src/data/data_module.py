from collections.abc import Iterable

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.datasets import ProstateCancer
from src.typing_ import Input


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        **datasets: ProstateCancer,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = datasets

    # def setup(self, stage: str) -> None:
    # match stage:
    #     case "fit":
    #         self.train = instantiate(self.datasets["train"])
    #         self.val = instantiate(self.datasets["val"])
    #     case "validate":
    #         self.val = instantiate(self.datasets["val"])
    #     case "test":
    #         self.test = instantiate(self.datasets["test"])
    #     case "predict":
    #         self.predict = instantiate(self.datasets["predict"])

    def train_dataloader(self) -> Iterable[Input]:
        return DataLoader(
            self.datasets["train"],
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Input]:
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[Input]:
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> list[Iterable[Input]]:
        return [
            DataLoader(
                dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
            for dataset in self.predict.datasets
        ]
