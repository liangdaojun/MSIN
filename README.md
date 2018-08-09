# Multi-Sample Inference Network （MSIN）

The Multi-Sample Reasoning Network (MSIN) proved the interesting fact that the neural network can correctly predict multiple samples at the same time, which is worthy of theoretical research. MSIN and its variants can classify multiple samples simultaneously in one forward process. Our experiments show that this method can effectively separate multiple categories while avoiding the confusion of multiple samples. Since the MSIN can predict multiple samples without adding parameters, this can significantly reduce the forward process of the neural network, thereby reducing inference time and hardware consumption. The properties of MSIN can be used to solve the category expansion problem. It can not only make the extended network have better generalization ability for new categories, but also can maintain the prediction performance of existing categories.

<div align=center>
<img src="https://github.com/liangdaojun/MSIN/blob/master/images/msin.jpg" width="480">
</div>

Different designs of the initial and final block of the basic network will result in variants of other MSINs. The network that uses a convolution layer as an independent structure in the initial block are denoted as MSIN-I1 (the number represents the number of layers), and the network that uses one block structure (two convolution layers) as an independent structure in the final block are denoted as MSIN-F2 (the last full connection layer is not included). Using different methods for multiple samples as input to the network will also result in very different classification performance.  Various MSIN variants are shown in Fig. 2.

<div align=center>
<img src="https://github.com/liangdaojun/MSIN/blob/master/images/msin_var.jpg" width="480">
</div>

We use MSIN-B to train on different datasets to get its generalization performance for simultaneous multi-task prediction. Its training process in different datasets is shown in Fig. 3. Note that each training task is trained simultaneously, except for the CIFAR-10 and SVHN datasets in Fig. 3.c. Fig. 3 shows the training and the test accuracy of the MSIN-B network on various datasets. Each task is almost unaffected by other tasks during training, which is almost indistinguishable from training the task alone.

<div align=center>
<img src="https://github.com/liangdaojun/MSIN/blob/master/images/msin_4.jpg" width="800">
</div>

Fig. 4 shows the performance of the MSIN on a multi-sample domain. It can be found that the MSIN can separate all the samples on the four different domains. The performance of the MSIN is slightly lower when predicting the four sample domains than when predicting the three sample domains. Compared with the single-sample inference network, the performance of the MSIN is slightly decline, but the availability of the MSIN is basically guaranteed. We believe that the performance of MSIN can be enhanced by some methods, and we will leave this work for the future.

<div align=center>
<img src="https://github.com/liangdaojun/MSIN/blob/master/images/msin_mfcC.jpg" width="480">
</div>

## Training
```python
    python train_msin.py 
```

- Model was tested with Python 3.5.2 with CUDA.
- Model should work as expected with Pytorch >= 2.0

## Test
-----
Test results on various datasets. 


|Model type        |  C10  : MNIST  | C100  : MNIST | C10   : C100   |  C10   : SVHN  |
| --------         |:--------------:|:-------------:|:--------------:|:--------------:|
|original category |  5.8  :  0.8   | 25.8  : 0.8   | 5.8   : 25.8   |  5.8   : 3.7   |
|multiple category |  6.2  :  0.8   | 26.5  : 0.8   | 6.8   : 31.4   |  8.8   : 3.9   |
|single   category |  6.3  :  0.8   | 26.7  : 0.8   | 7.0   : 32.1   |  9.1   : 3.9   |

## Acknowledgement
This reimplementation are adapted from the [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) repository by [ikhlestov](https://github.com/kuangliu) and  [densenet-pytorch](https://github.com/andreasveit/densenet-pytorch) repository by [andreasveit](https://github.com/andreasveit).
