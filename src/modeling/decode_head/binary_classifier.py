from torch import Tensor, nn

from src.modeling.decode_head.decode_head import DecodeHead


class BinaryClassifier(DecodeHead):
    def __init__(self, in_features: int) -> None:
        """Binary classifier head for the prostate cancer model.

        Arguments:
            in_features: int
                Number of input features to the classifier.
        """
        super().__init__(in_features)
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.proj = nn.Linear(in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.global_pool(x)  # (B, C, 1, 1)
        x = x.flatten(start_dim=-3, end_dim=-1)  # (B, C)
        x = self.dropout(x)
        x = self.proj(x)
        return x.sigmoid()
