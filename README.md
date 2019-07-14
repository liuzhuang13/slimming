# Network Slimming

This repository contains the code for the following paper 

[Learning Efficient Convolutional Networks through Network Slimming](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf) (ICCV 2017).

[Zhuang Liu](https://liuzhuang13.github.io/), [Jianguo Li](https://sites.google.com/site/leeplus/), [Zhiqiang Shen](http://zhiqiangshen.com), [Gao Huang](http://www.cs.cornell.edu/~gaohuang/), [Shoumeng Yan](https://scholar.google.com/citations?user=f0BtDUQAAAAJ&hl=en), [Changshui Zhang](http://bigeye.au.tsinghua.edu.cn/english/Introduction.html).

The code is based on [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

We have now released another [[PyTorch implementation]](https://github.com/Eric-mingjie/network-slimming) which supports ResNet and DenseNet, based on Qiang Wang's Pytorch implementation listed below.

Other Implementations:
[[Pytorch]](https://github.com/foolwood/pytorch-slimming) by Qiang Wang.
[[Chainer]](https://github.com/dsanno/chainer-slimming) by Daiki Sanno.
[[Pytorch hand detection using YOLOv3]](https://github.com/Lam1360/YOLOv3-model-pruning) by Lam1360.
[[Pytorch object detection using YOLOv3]](https://github.com/talebolano/yolov3-network-slimming) by talebolano.


Citation:

	@inproceedings{Liu2017learning,
		title = {Learning Efficient Convolutional Networks through Network Slimming},
		author = {Liu, Zhuang and Li, Jianguo and Shen, Zhiqiang and Huang, Gao and Yan, Shoumeng and Zhang, Changshui},
		booktitle = {ICCV},
		year = {2017}
	}

## Introduction


Network Slimming is a neural network training scheme that can simultaneously reduce the model size, run-time memory, computing operations, while introducing no accuracy loss to and minimum overhead to the training process. The resulting models require no special libraries/hardware for efficient inference.



## Approach
<div align=center>
<img src="https://user-images.githubusercontent.com/8370623/29604272-d56a73f4-879b-11e7-80ea-0702de6bd584.jpg" width="740">
</div>

<div align=center>
Figure 1:
 The channel pruning process.
</div> 


<br>

We associate a scaling factor (reused from batch normalization layers) with each channel in convolutional layers. Sparsity
regularization is imposed on these scaling factors during training to automatically identify unimportant channels. The channels with small
scaling factor values (in orange color) will be pruned (left side). After pruning, we obtain compact models (right side), which are then
fine-tuned to achieve comparable (or even higher) accuracy as normally trained full network.

<br>


<div align=center>
<img src="https://user-images.githubusercontent.com/8370623/29604357-18f3ce18-879c-11e7-9204-8ee86f5e7245.jpg" width="740">
</div>

<div align=center>
Figure 2: Flow-chart of the network slimming procedure. The dotted line is for the multi-pass version of the procedure.
</div> 


## Example Usage
  
This repo holds the example code for VGGNet on CIFAR-10 dataset. 

0. Prepare the directories to save the results

```
mkdir vgg_cifar10/
mkdir vgg_cifar10/pruned
mkdir vgg_cifar10/converted
mkdir vgg_cifar10/fine_tune
```
1. Train vgg network with channel level sparsity, S is the lambda in the paper which controls the significance of sparsity

```
th main.lua -netType vgg -save vgg_cifar10/ -S 0.0001
```
 2. Identify a certain percentage of relatively unimportant channels and set their scaling factors to 0

```
th prune/prune.lua -percent 0.7 -model vgg_cifar10/model_160.t7  -save vgg_cifar10/pruned/model_160_0.7.t7
```
 3. Re-build a real compact network and copy the weights from the model in the last stage

```
th convert/vgg.lua -model vgg_cifar10/pruned/model_160_0.7.t7 -save vgg_cifar10/converted/model_160_0.7.t7
```
 4. Fine-tune the compact network
 
```
th main_fine_tune.lua -retrain vgg_cifar10/converted/model_160_0.7.t7 -save vgg_cifar10/fine_tune/
```
## Note 
The original paper has a bug on the VGG results on ImageNet (Table 2). Please refer to this [[issue]](https://github.com/Eric-mingjie/rethinking-network-pruning/issues/3#issuecomment-443913400). The correct result was presented in Table 4 of [this paper](https://arxiv.org/abs/1810.05270).

## Contact
liuzhuangthu at gmail.com




