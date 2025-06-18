from torch import Tensor, nn

from src.modeling.decode_head.decode_head import DecodeHead


class LinearProbe(DecodeHead):
    def __init__(self, in_features: int) -> None:
        super().__init__(in_features)
        self.proj = nn.Linear(self.in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        # B = batch size, C = in_features, CL = CLASS_COUNT
        # x.shape = (B, C)
        x = self.proj(x)  # (B, 1)
        return x.sigmoid()
