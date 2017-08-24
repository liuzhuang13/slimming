# Network Slimming

Example code for the paper [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519) (In ICCV 2017).

The code is based on [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

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


## Usage
This repo holds the example code for VGGNet on CIFAR-10 dataset. 

To run the example, simply type 

  ```shell
  sh example.sh
  ```
  
More detailed instructions are included as comments in the file example.sh.

## Contact
liuzhuangthu at gmail.com




