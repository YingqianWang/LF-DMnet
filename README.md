## LF-DAnet: Learning a Degradation-Adaptive Network for Light Field Image Super-Resolution
<br>

This is the PyTorch implementation of the method in our paper "*Learning a Degradation-Adaptive Network for Light Field Image Super-Resolution*". [[project](https://yingqianwang.github.io/LF-DAnet/)], [[paper](https://arxiv.org/pdf/2206.06214.pdf)].<br>

## News and Updates:
* 2022-06-21: Codes and models are released. Welcome to try our codes and report the bugs/mistakes you meet.
* 2022-06-17: Website is [online](https://yingqianwang.github.io/LF-DAnet/), on which we provided comparative videos and an interactive demo.
* 2022-06-14: Paper is posted on [arXiv](https://arxiv.org/pdf/2206.06214.pdf). Codes are under final preparation and will be released soon.
* 2022-05-25: Repository is created.


## Demo Videos:
We show the SR results of our LF-DAnet on real LFs captured by Lytro Illum cameras. More examples are available [here](https://github.com/YingqianWang/LF-DAnet/blob/main/demo_videos.md). Note that, these videos have been compressed, and the results shown below are inferior to the original outputs of our LF-DAnet.

https://user-images.githubusercontent.com/31008389/170413144-b7ea1bbb-bf62-46a3-91b6-80cf2813bd94.mp4

https://user-images.githubusercontent.com/31008389/170413107-48568226-cebb-4bd0-8b59-93a115d03367.mp4

<br>

## Preparation:

#### 1. Requirement:
* PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.7, cuda=9.0.
* Matlab for training/validation data generation.

#### 2. Datasets:
* We used the HCInew, HCIold and STFgantry datasets for training and validation. Please first download the aforementioned datasets via [Baidu Drive](https://pan.baidu.com/s/1mYQR6OBXoEKrOk0TjV85Yw) (key:7nzy) or [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EpkUehGwOlFIuSSdadq9S4MBEeFkNGPD_DlzkBBmZaV_mA?e=FiUeiv), and place these datasets to the folder `../Datasets/`.
* We used the EPFL, INRIA and STFlytro datasets (which are developed by using Lytro cameras) to test the practical value of our method.

#### 3. Generating training/validation data:
* Run `GenerateDataForTraining.m` to generate training data. The generated data will be saved in `./Data/Train_MDSR_5x5/`.
* Run `Generate_Data_for_Validation.m` to generate validation data. The generated data will be saved in `./Data/Validation_MDSR_5x5/`.

## Train:
* Set the hyper-parameters in `parse_args()` if needed. We have provided our default settings in the realeased codes.
* Run `train.py` to perform network training.
* Checkpoint will be saved to `./log/`.

## Validation (synthetic degradation):
* Run `validation.py` to perform validation on each dataset.
* The metric scores will be printed on the screen.

## Test on your own LFs:
* Place the input LFs into `./input` (see the attached examples).
* Run `test.py` to perform SR. 
* The super-resolved LF images will be automatically saved to `./output`.

## Citiation
**If you find this work helpful, please consider citing:**
```
@Article{LF-DAnet,
    author    = {Wang, Yingqian and Liang, Zhengyu and Wang, Longguang and Yang, Jungang and An, Wei and Guo, Yulan},
    title     = {Learning a Degradation-Adaptive Network for Light Field Image Super-Resolution},
    journal   = {arXiv preprint arXiv:2206.06214}, 
    year      = {2022},   
}
```
<br>

## Contact
**Welcome to raise issues or email to [wangyingqian16@nudt.edu.cn](wangyingqian16@nudt.edu.cn) for any question regarding this work.**
