import torch
import torch.nn as nn
from thop import profile


lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    bias=True,
    bidirectional=True,
)

x = torch.randn(1, 50, 128)
flops, params = profile(lstm, inputs=(x))
# output = model(img, audio, text, None)

print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Params: {params / 1e6:.6f} M")

output, _ = lstm(x)
print(output.shape, _[0].shape)