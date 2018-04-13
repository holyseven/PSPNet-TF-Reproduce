## Unofficial Reproduce of PSPNet.

Pyramid Scene Parsing Network [paper](https://arxiv.org/abs/1612.01105) and [official github](https://github.com/hszhao/PSPNet).

Here is an unofficial re-implementation of PSPNet (from training to test) in pure Tensorflow library, where the most interesting point is the implementation of ***sync batch norm*** across multiple gpus (see `./model/utils_mg.py`). Tested on TF1.1 and TF1.4.

The L2-SP regularization ([paper](https://arxiv.org/abs/1802.01483) and [code](https://github.com/holyseven/TransferLearningClassification)) is added too.

Experimental results will be updated soon.