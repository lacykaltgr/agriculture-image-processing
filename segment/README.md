# UNET SEGMENTATION PYTORCH

## Installation

```bash
pip install segment-torch
```

## Usage

```python
from segment_torch.unet import UNet
from torch import nn

device = "cuda"

config = dict(
    in_channels=3,
    out_channels=1,
    hiddens=[4, 8, 16, 32],
    dropouts=[0, 0.15, 0.15, 0.15],  # hiddens
    maxpools=2,  # hiddens - 1
    kernel_sizes=3,  # 2*hiddens + 3*hiddens + 2
    paddings='same',  # 2*hiddens + 3*hiddens + 2
    strides=1,  # 2*hiddens + 3*hiddens
    dilation=1,
    criterion=nn.BCELoss(),
    output_activation=nn.Sigmoid(),
    activation=nn.ReLU(),
    dimensions=2,
    device=device
)
unet = UNet(**config)
```

**Different ways to define configs**

```python

# 0. None: default values are used
kernel_sizes=None

# 1. Single value or tuple: all layers have the same value
kernel_sizes = 3 
kernel_sizes = (3, 3)

# 2. Lists of values
encooder_kernel_sizes = [3, 3, 3, 3]
decoder_kernel_sizes = [3, 3, 3, 3, 3]
kernel_sizes = [encooder_kernel_sizes, decoder_kernel_sizes]
```


