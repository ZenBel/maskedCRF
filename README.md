# maskedCRF
Implementation of [masked CRF](https://arxiv.org/abs/2103.10682) layer for both tensorflow and pytorch.

The TensorFlow implementation comes from https://github.com/DandyQi/MaskedCRF but the code has been updated for `tensorflow = "2.7.0"` and the `trans` initialization has been changed to match that in the PyTorch implementation.

The PyTorch implementation comes form https://github.com/zhw666888/Pytorch-MCRF but the code has been extended to choose between using a "classic" CRF layer or a masked one.

The notebook shows a toy example of a forward and backward pass on a synthetic dataset to confirm both implementations yield similar results.


Package versions:
- `tensorflow=2.7.0`
- `torch=1.10.0`
