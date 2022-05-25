## LF-DAnet: Learning a Degradation-Adaptive Network for Light Field Image Super-Resolution
<br>



https://user-images.githubusercontent.com/31008389/170273972-9a50726f-499d-45de-9527-f2f198df457c.mp4



https://user-images.githubusercontent.com/31008389/170275204-c2b14964-1112-49f1-9fba-ee54c5792dfa.mp4



https://user-images.githubusercontent.com/31008389/170274935-b5a6fb5c-2dcf-4d15-a263-8907bbc89a50.mp4


https://user-images.githubusercontent.com/31008389/170274111-6aaa7b15-e36e-4ee4-b647-fa0e6e0c73ea.mp4




https://user-images.githubusercontent.com/31008389/170274976-5eb2f4b1-4dfb-4335-b1d3-8803a47a18de.mp4




<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgSSR/master/Figs/DistgSSR.png" width="90%"> </p>

This is the PyTorch implementation of the method in our paper "Learning a Degradation-Adaptive Network for Light Field Image Super-Resolution". Please refer to our [paper](https://arxiv.org/pdf/2202.10603.pdf) for details.<br>

## News and Updates:
* 2022-05-25: Repository is created.

## Preparation:
#### 1. Requirement:
* PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.6, cuda=9.0.
* Matlab for training/test data generation and performance evaluation.
#### 2. Datasets:
* We used the EPFL, HCInew, HCIold, INRIA and STFgantry datasets for training and test. Please first download our dataset via [Baidu Drive](https://pan.baidu.com/s/1mYQR6OBXoEKrOk0TjV85Yw) (key:7nzy) or [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EpkUehGwOlFIuSSdadq9S4MBEeFkNGPD_DlzkBBmZaV_mA?e=FiUeiv), and place the 5 datasets to the folder `./Datasets/`.
#### 3. Generating training/test data:
* Run `Generate_Data_for_Train.m` to generate training data. The generated data will be saved in `./Data/train_kxSR_AxA/`.
* Run `Generate_Data_for_Test.m` to generate test data. The generated data will be saved in `./Data/test_kxSR_AxA/`.
#### 4. Download our pretrained models:
We provide the models for 4× SR. Download our models through the following links:

## Train:
* Set the hyper-parameters in `parse_args()` if needed. We have provided our default settings in the realeased codes.
* Run `train.py` to perform network training.
* Checkpoint will be saved to `./log/`.

## Test on the datasets:
* Run `test_on_dataset.py` to perform test on each dataset.
* The original result files and the metric scores will be saved to `./Results/`.

## Test on your own LFs:
* Place the input LFs into `./input` (see the attached examples).
* Run `demo_test.py` to perform spatial super-resolution. Note that, the selected pretrained model should match the input in terms of the angular resolution. 
* The super-resolved LF images will be automatically saved to `./output`.

## Results:

### Quantitative Results:
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgSSR/master/Figs/QuantitativeSSR.png" width="100%"> </p>

### Visual Comparisons:
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgSSR/master/Figs/Visual-SSR.png" width="100%"> </p>

### Efficiency:
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgSSR/master/Figs/Efficiency-SSR.png" width="50%"> </p>

### Angular Consistency:
<p align="center"> <a href="https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgLF-SpatialSR.mp4"><img src="https://raw.github.com/YingqianWang/DistgSSR/master/Figs/AngCons-SSR.png" width="80%"></a> </p>


## Citiation
**If you find this work helpful, please consider citing:**
```
@Article{LF-DAnet,
    author    = {Wang, Yingqian and Wang, Longguang and Wu, Gaochang and Yang, Jungang and An, Wei and Yu, Jingyi and Guo, Yulan},
    title     = {Learning a Degradation-Adaptive Network for Light Field Image Super-Resolution},
    journal   = {arxiv}, 
    year      = {2022},   
}
```
<br>

## Contact
**Welcome to raise issues or email to [wangyingqian16@nudt.edu.cn](wangyingqian16@nudt.edu.cn) for any question regarding this work.**
