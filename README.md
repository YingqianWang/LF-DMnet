## LF-DMnet: Real-World Light Field Image Super-Resolution via Degradation Modulation
<br>

This is the PyTorch implementation of the method in our paper "*Real-World Light Field Image Super-Resolution via Degradation Modulation*". [[project](https://yingqianwang.github.io/LF-DMnet/)], [[paper](https://arxiv.org/pdf/2206.06214.pdf)].<br>

## News and Updates:
* 2023-12-01: Revised paper is posted on [arXiv](https://arxiv.org/pdf/2206.06214.pdf).
* 2022-06-21: Codes and models are released. Welcome to try our codes and report the bugs/mistakes you meet.
* 2022-06-17: Website is [online](https://yingqianwang.github.io/LF-DMnet/), on which we provided comparative videos and an interactive demo.
* 2022-06-14: Paper is posted on [arXiv](https://arxiv.org/pdf/2206.06214.pdf). Codes are under final preparation and will be released soon.
* 2022-05-25: Repository is created.


## Demo Videos:
We show the SR results of our LF-DMnet on real LFs captured by Lytro Illum cameras. More examples are available [here](https://github.com/YingqianWang/LF-DMnet/blob/main/demo_videos.md). Note that, these videos have been compressed, and the results shown below are inferior to the original outputs of our LF-DMnet.



https://github.com/YingqianWang/LF-DMnet/assets/31008389/73490c47-9a51-490a-a4b1-0794d4706d77



https://github.com/YingqianWang/LF-DMnet/assets/31008389/c41ae453-030b-4d58-8442-f59bed2cbc39



<br>

## Preparation:

#### 1. Requirement:
* PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.7, cuda=9.0.
* Matlab for training/validation data generation.

#### 2. Datasets:
* We used the HCInew, HCIold and STFgantry datasets for training and validation. Please first download the aforementioned datasets via [Baidu Drive](https://pan.baidu.com/s/1mYQR6OBXoEKrOk0TjV85Yw) (key:7nzy) or [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EpkUehGwOlFIuSSdadq9S4MBEeFkNGPD_DlzkBBmZaV_mA?e=FiUeiv), and place these datasets to the folder `../Datasets/`.
* We used the EPFL, INRIA and STFlytro datasets (which are developed by using Lytro cameras) to test the practical value of our method.

#### 3. Generating training/validation data:
* Run `GenerateDataForTraining.m` to generate training data. The generated data will be saved in `../Data/Train_MDSR_5x5/`.
* Please download the validation data via [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EgVU4b1ImNFMuchPObqZjLYBbI7zcfn_3tcM8bpXzphX5g) and place these data to the folder `../Data/Validation_MDSR_5x5/`.

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
@Article{LF-DMnet,
    author    = {Wang, Yingqian and Liang, Zhengyu and Wang, Longguang and Yang, Jungang and An, Wei and Guo, Yulan},
    title     = {Real-World Light Field Image Super-Resolution via Degradation Modulation},
    journal   = {arXiv preprint arXiv:2206.06214}, 
    year      = {2022},   
}
```
<br>

## Contact
**Welcome to raise issues or email to [wangyingqian16@nudt.edu.cn](wangyingqian16@nudt.edu.cn) for any question regarding this work.**

<details> 
<summary>statistics</summary>

![visitors](https://visitor-badge.glitch.me/badge?page_id=YingqianWang/LF-DMnet)

</details> 
