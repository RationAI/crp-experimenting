from abc import ABC

from torch import nn


class DecodeHead(nn.Module, ABC):
    """Decode head after the backbone.

    Attributes:
        in_features (int): Number of input features to the classifier.
            It is required to initialize the first layer after the backbone.
    """

    def __init__(self, in_features: int) -> None:
        """Initialize the decode head.

        Args:
            in_features (int): Number of input features to the classifier.
                It is required to initialize the first layer after the backbone.
        """
        super().__init__()
        self.in_features = in_features
