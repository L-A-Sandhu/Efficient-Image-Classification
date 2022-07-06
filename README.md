# Mobile-Net_Inception-Net_Cifar10_Image_classification\

 This repository contains comparisons of different convolution neural networks for the CIFAR-10 data-set. Clone this repo please follow the following steps 
```
git clone https://github.com/L-A-Sandhu/Mobile-Net_Inception-Net_Cifar10_Image_classification.git

```

The rest of the repository is divided as follows. 
  1. Requirements
  2. Mobile Net 
  3. Inception Net
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
  
## Mobile Net

This section explains traning, testing and infrence steps for Mobile Net. please follow the following commands 

### Traning 
The model is fine tunned on cifar10 dataset. The model was earlier trained on imagenet dataset. The script automatically downloads the cifa10 dataset. 
```
python Mobile-Net.py  --model_dir=<Location for saving model>  --inp=<train , test or infer> --b_s=< Batch size> --e=<epoch>
example command 
python Mobile-Net.py  --model_dir='./checkpoint/'  --inp=train --b_s=16 --e=100

```
### Test 
```
python Mobile-Net.py  --model_dir=<Location for saving model>  --inp=<train , test or infer> --b_s=< Batch size> --e=<epoch>
example command 
python Mobile-Net.py  --model_dir='./checkpoint/'  --inp=test --b_s=16 --e=100

```

## Inception-Net
This section explains traning, testing and infrence steps for Inception Net. please follow the following commands 
```
cd ../Inception_NET/
```
### Traning 
The model is fine tunned on cifar10 dataset. The model was earlier trained on imagenet dataset. The script automatically downloads the cifa10 dataset. 
```
python Inception-Net.py  --model_dir=<Location for saving model> --inp=<train , test or infer> --b_s=< Batch size> --e=<epoch>
example command 
python Inception-Net.py  --model_dir='./checkpoint/' --inp=train --b_s=16 --e=100
```
### Test 
```
python Inception-Net.py  --model_dir=<Location for saving model> --inp=<train , test or infer> --b_s=< Batch size> --e=<epoch>
example command 
python Inception-Net.py  --model_dir='./checkpoint/' --inp=test --b_s=16 --e=100

```
## Summary
