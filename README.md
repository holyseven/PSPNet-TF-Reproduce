# Training Reproduce of PSPNet.

(Updated 2019/02/26. A major change of code structure. For the version before, checkout v0.9 https://github.com/holyseven/PSPNet-TF-Reproduce/tree/v0.9.)

This is an implementation of PSPNet (from training to test) in pure Tensorflow library (**tested on TF1.12, Python 3**).

- Supported Backbones: ResNet-V1-50, ResNet-V1-101 and other ResNet-V1s can be easily added.
- Supported Databases: [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), [SBD (Augmented Pascal VOC)](https://github.com/holyseven/PSPNet-TF-Reproduce/issues/12#issuecomment-428876239) and [Cityscapes](https://www.cityscapes-dataset.com/).
- Supported Modes: training, validation and inference with multi-scale inputs.
- More things: L2-SP regularization and sync batch normalization implementation.

## L2-SP Regularization
L2-SP regularization is a variant of L2 regularization. Instead of the origin like L2 does, L2-SP sets the pre-trained model as reference, just like `(w - w0)^2`, where `w0` is the pre-trained model. Simple but effective. More details about L2-SP can be found in the [paper](https://arxiv.org/abs/1802.01483) and the [code](https://github.com/holyseven/TransferLearningClassification).

If you find the L2-SP useful for your research (not limited in image segmentation), please consider citing our work:

```
@inproceedings{li2018explicit,
  author    = {Li, Xuhong and Grandvalet, Yves and Davoine, Franck},
  title     = {Explicit Inductive Bias for Transfer Learning with Convolutional Networks},
  booktitle={International Conference on Machine Learning (ICML)},
   pages     = {2830--2839},
  year      = {2018}
}
```

## Sync Batch Norm
When concerning image segmentation, batch size is usually limited. Small batch size will make the gradients instable and harm the performance, especially for batch normalization layers. Multi-GPU settings by default does not help because the statistics in batch normalization layer are computed independently within each GPU. More discussion can be found [here](https://github.com/tensorflow/tensorflow/issues/7439) and [here](https://github.com/torch/nn/issues/1071).

This repo resolves this problem in pure python and pure Tensorflow by simply using a list as input. The main idea is located in [model/utils_mg.py](https://github.com/holyseven/PSPNet-TF-Reproduce/blob/master/model/utils_mg.py)

_I do not know if this is the first implementation of sync batch norm in Tensorflow, but there is already an implementation in [PyTorch](http://hangzh.com/PyTorch-Encoding/syncbn.html) and [some applications](https://github.com/CSAILVision/semantic-segmentation-pytorch)._

**Update:** There is other implementation that uses NCCL to gather statistics across GPUs, see in [tensorpack](https://github.com/tensorpack/tensorpack/blob/master/tensorpack/models/batch_norm.py#L221). However, TF1.1 does not support gradients passing by `nccl_all_reduce`. Plus, ppc64le with tf1.10, cuda9.0 and nccl1.3.5 was not able to run this code. No idea why, and do not want to spend a lot of time on this. Maybe nccl2 can solve this.

## Results

### Numerical Results

- Random scaling for all
- Random rotation for SBD
- SS/MS on validation set
- Welcome to correct and fill in the table

<table>
   <tr>
      <td></td>
      <td>Backbones</td>
      <td>L2</td>
      <td>L2-SP</td>
   </tr>
   <tr>
      <td rowspan="2">Cityscapes (train set: 3K)</td>
      <td>ResNet-50</td>
      <td>76.9/?</td>
      <td>77.9/?</td>
   </tr>
   <tr>
      <td>ResNet-101</td>
      <td>77.9/?</td>
      <td>78.6/?</td>
   </tr>
   <tr>
      <td rowspan="2">Cityscapes (coarse + train set: 20K + 3K)</td>
      <td>ResNet-50</td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>ResNet-101</td>
      <td>80.0/80.9</td>
      <td>80.1/81.2*</td>
   </tr>
   <tr>
      <td rowspan="2">SBD </td>
      <td>ResNet-50</td>
      <td>76.5/?</td>
      <td>76.6/?</td>
   </tr>
   <tr>
      <td>ResNet-101</td>
      <td>77.5/79.2</td>
      <td>78.5/79.9</td>
   </tr>
   <tr>
      <td rowspan="2">ADE20K</td>
      <td>ResNet-50</td>
      <td>41.92/43.09</td>
      <td></td>
   </tr>
   <tr>
      <td>ResNet-101</td>
      <td>42.80/?</td>
      <td></td>
   </tr>
</table>

*This model gets 80.3 without post-processing methods on [Cityscapes test set (1525)](https://www.cityscapes-dataset.com/method-details/?submissionID=1148).

### Qualitative Results on Cityscapes

![](https://github.com/holyseven/PSPNet-TF-Reproduce/blob/master/demo_examples/berlin_000000_000019_leftImg8bit.png)    ![](https://github.com/holyseven/PSPNet-TF-Reproduce/blob/master/demo_examples/demo_color.png)

## Devil Details

### Training and Evaluation

Download the databases with the links: [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), [SBD (Augmented Pascal VOC)](https://github.com/holyseven/PSPNet-TF-Reproduce/issues/12#issuecomment-428876239) and [Cityscapes](https://www.cityscapes-dataset.com/).

Prepare the database for Cityscapes by generating `*labelTrainIds.png` images with  [createTrainIdLabelImgs](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py), and then change the code in [database/reader.py](https://github.com/holyseven/PSPNet-TF-Reproduce/blob/master/database/reader.py#L36) or move undersired images to other directory.

Download pretrained models.
```bash
cd z_pretrained_weights
sh download_resnet_v1_101.sh
```

A script of training resnet-50 on ADE20K, getting around 41.92 mIoU scores (with single-scale test):
```bash
python ./run.py --network 'resnet_v1_50' --visible_gpus '0,1' --reader_method 'queue' --lrn_rate 0.01 --weight_decay_mode 0 --weight_decay_rate 0.0001 --weight_decay_rate2 0.001 --database 'ADE' --subsets_for_training 'train' --batch_size 8 --train_image_size 480 --snapshot 30000 --train_max_iter 90000 --test_image_size 480 --random_rotate 0 --fine_tune_filename './z_pretrained_weights/resnet_v1_50.ckpt'
```

### Test and Infer

Test with multi-scale (set `batch_size` as large as you can to speed up).
```bash
python predict.py --visible_gpus '0' --network 'resnet_v1_101' --database 'ADE' --weights_ckpt './log/ADE/PSP-resnet_v1_101-gpu_num2-batch_size8-lrn_rate0.01-random_scale1-random_rotate1-480-60000-train-1-0.0001-0.001-0-0-1-1/snapshot/model.ckpt-60000' --test_subset 'val' --test_image_size 480 --batch_size 8 --ms 1 --mirror 1
```

Infer one image (with multi-scale).
```bash
python demo_infer.py --database 'Cityscapes' --network 'resnet_v1_101' --weights_ckpt './log/Cityscapes/old/model.ckpt-50000' --test_image_size 864 --batch_size 4 --ms 1
```

### Uncertainties for Training Details:
1. (Cityscapes only) Whether finely labeled data in the first training stage should be involved?
2. (Cityscapes only) Whether the (base) learning rate should be reduced in the second training stage?
3. Whether logits should be resized to original size before computing the loss?
4. Whether new layers should receive larger learning rate?
5. About [weired padding behavior of tf.image.resize_images()](https://github.com/tensorflow/tensorflow/issues/6720). Whether the `align_corners=True` should be set?
6. What is optimal hyperparameter of decay for statistics of batch normalization layers? (0.9, [0.95](https://github.com/hszhao/PSPNet/blob/master/evaluation/prototxt/pspnet101_VOC2012_473.prototxt#L59), [0.9997](https://github.com/tensorflow/models/blob/master/research/deeplab/model.py#L376))
7. may be more but not sure how much these little changes can effect the results ...
8. Welcome to discuss !

## Change Log

### 26 Febuary, 2019

* Code structure: on-the-fly evaluation during training.
* Code structure: wrapping of the model.
* Add `tf.data` support, but with queue-based reader is faster.
* print results using `python utils.py` in `experiment_manager` dir.
* The default environment is Python 3 and TF1.12. OpenCV is needed for predicting and demo_infer.
* The previous version becomes a branch of this repo named as v0.9.


## External links

 Pyramid Scene Parsing Network [paper](https://arxiv.org/abs/1612.01105) and [official github](https://github.com/hszhao/PSPNet).
