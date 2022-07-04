# Mobile-Net_Inception-Net_Cifar10_Image_classification\

Image classification is a simple case where we have to detect the presence of a particular class inside an image frame. In this repo we have used inception net and Mobile Net for image classfication on a **cifar10** benchmark dataset. Clone this repo please follow the following steps 
'''
cd git clone https://github.com/L-A-Sandhu/Mobile-Net_Inception-Net_Cifar10_Image_classification.git

```

The rest of the repository is divided as follows. 
  1. Mobile Net 
  2. Inception Net

```
  
## Mobile Net

In this section we will explain traning, testing and infrence steps for Mobile Net. please follow the following commands 
```
cd 
conda create env -n <environment -name> python==3.7.4
conda activate <environment-name>
pip install -r requirements.txt
cd M0bile_Net/
```
### Traning 
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
In this section we will explain traning, testing and infrence steps for Inception Net. please follow the following commands 
```
cd ../Inception_NET/
```
### Traning 
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

