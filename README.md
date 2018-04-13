## Unofficial Reproduce of PSPNet.

Pyramid Scene Parsing Network [paper](https://arxiv.org/abs/1612.01105) and [official github](https://github.com/hszhao/PSPNet).

Here is an unofficial re-implementation of PSPNet (from training to test) in pure Tensorflow library, where the most interesting point is the implementation of ***sync batch norm*** across multiple gpus (see `./model/utils_mg.py`). Tested on TF1.1 and TF1.4.

### Sync Batch Norm
When concerning image segmentation, batch size is usually limited. Small batch size will harm the performance. Using multi-GPU to increase the batch size does not help because the statistics are computed with each GPU (by default in all DL libraries). More discussion can be found [here](https://github.com/tensorflow/tensorflow/issues/7439) and [here](https://github.com/torch/nn/issues/1071).

So this repo resolves this problem in pure python and pure Tensorflow.

_I do not know if this is the first implementation of sync batch norm in Tensorflow, but there is already an implementation in [PyTorch](http://hangzh.com/PyTorch-Encoding/syncbn.html) and [some applications](https://github.com/CSAILVision/semantic-segmentation-pytorch)._

### L2-SP regularization
L2-SP regularization is a variant of L2 regularization and sets the pre-trained model as reference, instead of the origin like L2 does. More details can be found in the [paper](https://arxiv.org/abs/1802.01483) and [code](https://github.com/holyseven/TransferLearningClassification).

### Results
Experimental results will be updated soon.