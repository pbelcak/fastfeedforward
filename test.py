from fastfeedforward import FFF

import torch

layer = FFF(768, 768, 768, 2)
input = torch.randn(32, 768)

layer.eval()
output = layer(input)