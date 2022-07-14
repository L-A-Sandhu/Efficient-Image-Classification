# Efficient-Image-Classification

 This repository contains comparisons of different convolution neural networks for the CIFAR-10 data-set.ImageNet pretrained models have been fine tuned on CIFAR-10 dataset.The script automatically downloads the cifar 10 dataset. Clone this repo please follow the following steps 
```
git clone https://github.com/L-A-Sandhu/Efficient-Image-Classification.git

```

The rest of the repository is divided as follows. 
  1. Requirements
  2. Fine Tuning
  3. Using Pretrained Model
  4. Summary
## Requirements 
This repository requires 
* **tensorflow**
* **matplotlib**
* **scipy**
* **protobuf**


For complete installation please follow the following steps
```
cd M0bile_Net/
conda create  -n <environment -name> python==3.7.4
conda activate <environment-name>
pip install -r requirements.txt
```
## Fine Tuning 
This section disccusses the fine tunning method for Mobile Net and Inception net. The keras implementations of Mobile Net and Inception Net is used in this work. Their weights are trained on Image Net dataset. However, this work fine tuned the model on cifar10 dataset. 
### Mobile Net

This section explains traning, testing and fine tuning  steps for Mobile Net. please follow the following commands 

### Fine Tune
 
```
python Mobile-Net.py  --model_dir=<Location for saving model> --inp=<tune, train, test > --b_s=< Batch size> --e=<epoch>
example command 
python Mobile-Net.py  --model_dir='./checkpoint/' --inp=tune --b_s=16 --e=100
```
### Train
 
```
python Mobile-Net.py --model_dir=<Location for saving model> --inp=<tune, train, test > --b_s=< Batch size> --e=<epoch>
example command 
python Mobile-Net.py --model_dir='./checkpoint/' --inp=train --b_s=16 --e=100
```
### Test 
```
python Mobile-Net.py  --model_dir=<Location for saving model> --inp=<tune, test, train>
example command 
python Mobile-Net.py --model_dir='./checkpoint/' --inp=test
```
### Inception-Net
This section explains Fine tunning, Traning,  testing and infrence steps for Inception Net. please follow the following commands 
```
cd ../Inception_NET/
```
### Fine Tune
 
```
python Inception-Net.py  --model_dir=<Location for saving model> --inp=<tune, train, test > --b_s=< Batch size> --e=<epoch>
example command 
python Inception-Net.py  --model_dir='./checkpoint/' --inp=tune --b_s=16 --e=100
```
### Train
 
```
python Inception-Net.py  --model_dir=<Location for saving model> --inp=<tune, train, test > --b_s=< Batch size> --e=<epoch>
example command 
python Inception-Net.py  --model_dir='./checkpoint/' --inp=train --b_s=16 --e=100
```
### Test 
```
python Inception-Net.py  --model_dir=<Location for saving model> --inp=<tune, test, train>
python Inception-Net.py  --model_dir='./checkpoint/' --inp=test

```
## Pretrained Model
 In this work the models are trained on cifar10 dataset with batch size 128 and 50 epochs. You can download the pretrained weights and place them at **./Inception_NET/checkpoint/**  or **./M0bile_Net/checkpoint/**/  for inferene. The pretrained weights for Mobile net and Inception Net can be be downloaded from the folllowing links respectively. 
```
 https://drive.google.com/file/d/1OCxDNDUbMJcoo8QbzB6hM4r4yqXOXpZU/view?usp=sharing
https://drive.google.com/file/d/144j9-G-v2x6YCTZ4u9_NDzXpVnVwT_kC/view?usp=sharing
``` 
## Results and comparision 
Test results and comparision for both models is shown in the following table 
| Model         | Parameters | Accuracy | Latency   | Size on Disk (MB)|
|---------------|------------|----------|-----------|--------|
| Mobile-Net    | 3,783,510  | 0.836    |  0.0004   | 38.8   |
| Inception-Net | 22,115,894  | 0.811    |  0.0006  | 174.2  |


### Confusion Matrix 
Thee confusion matrix for Mobile net is shown shown below

|               | 'airplane' |  'automobile' |  'bird' |  'cat' |  'deer' | 'dog' |  'frog' |  'horse' |  'ship' |  'truck' |
|---------------|------------|---------------|---------|--------|---------|-------|---------|----------|---------|----------|
| 'airplane'    | 728        | 18            | 75      | 20     | 63      | 3     | 30      | 1        | 49      | 13       |
|  'automobile' | 0          | 964           | 1       | 4      | 2       | 0     | 5       | 0        | 12      | 12       |
|  'bird'       | 10         | 0             | 898     | 23     | 31      | 5     | 30      | 1        | 2       | 0        |
|  'cat'        | 7          | 7             | 61      | 688    | 96      | 75    | 59      | 4        | 3       | 0        |
|  'deer'       | 0          | 1             | 35      | 17     | 912     | 11    | 19      | 4        | 1       | 0        |
| 'dog'         | 1          | 6             | 39      | 123    | 64      | 724   | 37      | 5        | 1       | 0        |
|  'frog'       | 0          | 1             | 16      | 19     | 11      | 3     | 949     | 0        | 1       | 0        |
|  'horse'      | 1          | 5             | 36      | 42     | 82      | 83    | 12      | 736      | 2       | 1        |
|  'ship'       | 5          | 10            | 17      | 8      | 11      | 4     | 9       | 1        | 932     | 3        |
|  'truck'      | 3          | 83            | 6       | 18     | 9       | 4     | 16      | 1        | 29      | 831      |

The confusion matrix for Inception net is shown below
|               | 'airplane' |  'automobile' |  'bird' |  'cat' |  'deer' | 'dog' |  'frog' |  'horse' |  'ship' |  'truck' |
|---------------|------------|---------------|---------|--------|---------|-------|---------|----------|---------|----------|
| 'airplane'    | 894        | 12            | 14      | 9      | 3       | 1     | 2       | 1        | 43      | 21       |
|  'automobile' | 7          | 933           | 1       | 2      | 2       | 0     | 3       | 0        | 27      | 25       |
|  'bird'       | 103        | 7             | 742     | 63     | 23      | 19    | 30      | 5        | 6       | 2        |
|  'cat'        | 27         | 3             | 50      | 765    | 33      | 66    | 19      | 5        | 19      | 13       |
|  'deer'       | 21         | 3             | 82      | 67     | 769     | 14    | 13      | 5        | 20      | 6        |
| 'dog'         | 10         | 8             | 32      | 207    | 31      | 687   | 6       | 5        | 7       | 7        |
|  'frog'       | 18         | 4             | 42      | 55     | 18      | 14    | 828     | 0        | 19      | 2        |
|  'horse'      | 19         | 4             | 27      | 70     | 66      | 73    | 8       | 715      | 3       | 15       |
|  'ship'       | 29         | 8             | 6       | 5      | 1       | 1     | 1       | 1        | 933     | 15       |
|  'truck'      | 25         | 74            | 4       | 7      | 0       | 4     | 1       | 3        | 32      | 850      |
