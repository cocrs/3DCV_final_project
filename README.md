# 3DCV final: Semi-spervised Learning for Reconstruction of Multiple Humans from a single image

## Installation instructions

Our project is base on the following repository. Please follow the instructions at their github to complete basic project installation first, including [Installation instructions](https://github.com/JiangWenPL/multiperson#installation-instructions) and [Fetch data](https://github.com/JiangWenPL/multiperson#fetch-data):
**Coherent Reconstruction of Multiple Humans from a Single Image**  
[Wen Jiang](https://jiangwenpl.github.io/)\*, [Nikos Kolotouros](https://www.seas.upenn.edu/~nkolot/)\*, [Georgios Pavlakos](https://www.seas.upenn.edu/~pavlakos/), [Xiaowei Zhou](http://www.cad.zju.edu.cn/home/xzhou/), [Kostas Daniilidis](http://www.cis.upenn.edu/~kostas/)  
CVPR 2020
[[paper](https://arxiv.org/pdf/2006.08586.pdf)] [[github](https://github.com/JiangWenPL/multiperson)]

After installation from the original github, you need to clone our repo and put files to the corresponding folders.

##### We encountered some problems in Fetch data and we write at bellow.
In Fetch data, we need to follow the instructions [here](https://github.com/vchoutas/smplx/tree/master/tools) to convert the models to be compatible with python3. Only the "Removing Chumpy objects" part is necessary (Notice: this command needs to run under python2.7). After processing, we have to rename the file `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `SMPL_NEUTRAL.pkl` and put them under `mmdetection/data/smpl`


## Environment
Our testing is on Ubuntu 18.04 using 2080ti.

## Prepare datasets
Please refer to [DATASETS.md](https://github.com/JiangWenPL/multiperson/blob/master/DATASETS.md) for the preparation of the dataset files. We use Panoptic for evaluation and MPI-INF-3DHP for training. You can download Panoptic only if you want to test the evaluation code because MPI-INF-3DHP is really large after extracting all frames (about 550GB). You will also need the unlabeled dataset [Cityscapes](https://www.cityscapes-dataset.com/) we used if you want to run our semi-supervised training code. Please put Cityscapses train/val/test image folders in `imgs` folder like the folder structure bellow:

```bash
./mmdetection/
    ...
    ./data/
        ...
        ./pseudo/
            ./imgs/
	    	./train
		./val
		./test
            ./annotations/
                ./anno_500.pkl
		./anno_jitter_500.pkl
```


##### We encountered some problems and we write at bellow.
Panoptic:
* While downloading Panoptic using the script from [panoptic-toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox), it is normal to see that some links are not available, and that does not affect the following processes. (Notice: the required image format in multiperson is png, so please specify `./scripts/extractAll.sh [sequence] png` while extracting frames using panoptic-toolbox)
* We were unable to use the preprocess code multiperson provided. We adjusted it ([adjusted code](./misc/preprocess_datasets/full)) to only extract the frames they need and use their processed annotation files to evaluate. The way to run our adjusted code is the same as the original code.

MPI-INF-3DHP:
* There are two links in the [official website of the dataset](https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/). The one on the left(MuCo-3DHP Scripts) is the one we want.
* We only get videos after downloading. Like Panoptic, we need to use the preprocess code to extract the frames ([adjusted code](./misc/preprocess_datasets/full)).
* Pose and Shape npz files are also needed, but not provided in this multiperson repo. We find it in another repo [link](http://visiondata.cis.upenn.edu/spin/dataset_extras.tar.gz). We need the `mpi_inf_3dhp_train.npz`. Please put it in `mmdetection/data/mpi_inf_3dhp/extras`

## Results
Method                     | Haggling  | Mafia  | Ultim.  | Pizza  | Mean  | checkpoint |
-------------------------- | ----------|--------|---------|--------|-------|------------|
multiperson (github ckpt)  | 129.1     | 132.8  | 153.0   | 153.6  | 142.1 |            |
our baseline               | 132.5     | 137.4  | 157.9   | 158.9  | 146.7 | [link](https://drive.google.com/file/d/1J7NL5Z5bqLzLgE5X5c3I2DGvWkjJvYhp/view?usp=sharing)    |
pseudo label               | 132.1     | 134.4  | 153.4   | 157.6  | 144.4 | [link](https://drive.google.com/file/d/1nGKWp84flcobT1Dcj3xNb3guzqV8k353/view?usp=sharing)    |
confident pseudo label      | 130.4     | 135.7  | 153.6   | 156.3  | 144.0 | [link](https://drive.google.com/file/d/1d0YQkXEZEMzGSY1BudiDVbPFYwjbm7zu/view?usp=sharing)    |

## Run demo code
You could use our pretrained checkpoint to run the demo on the images in the folder.
Example usage:
```
cd mmdetection
python3 tools/our_demo.py --config=configs/smpl/tune.py --image_folder=data/Panoptic --output_folder=results/ --ckpt /path/to/model --annotation=data/Panoptic/processed/annotations/160906_pizza1.pkl
```
The annotation file ```160906_pizza1.pkl``` can be replaced with other annotation files:
- ```160422_mafia2.pkl```: mafia sequence of Panoptic
- ```160422_ultimatum1.pkl```: ultimatum sequence of Panoptic
- ```160422_haggling1.pkl```: haggling sequence of Panoptic
- ```160906_pizza1.pkl```: pizza sequence of Panoptic

## Run evaluation code
You could use our pretrained checkpoint to evaluate on Panoptic.

Example usage:
```
cd mmdetection
python3 tools/full_eval.py configs/smpl/tune.py haggling --ckpt /path/to/ckpt --seed 1111
```

The ```haggling``` option can be replaced with other dataset or sequences based on the type of evaluation you want to perform:
- `mafia`: mafia sequence of Panoptic
- `ultimatum`: ultimatum sequence of Panoptic
- `haggling`: haggling sequence of Panoptic
- `pizza`: pizza sequence of Panoptic

Regarding the evaluation:
- For Panoptic, the command will compute the MPJPE for each sequence.

## Run training code

Please make sure you have downloaded checkpoint.pt from multiperson and put it in the folder `data`.
We resume from checkpoint.pt and first finetune on 500 images we sampled from MPI-INF-3DHP as our baseline. The annotations for 500 images we sampled is [here](https://drive.google.com/file/d/15MWagBYX4HUAMRuNihpA2qlAW3U-DmKx/view?usp=sharing). Please put it in `mmdetection/data/pseudo/annotations`
```bash
cd mmdetection
python3 tools/train.py configs/smpl/tune_mpi.py --load_pretrain ./data/checkpoint.pt --seed 1111
i=0
while [ $i -le 5 ]
do
    python3 tools/train.py configs/smpl/tune_mpi.py --seed 1111
    i=$(($i+1))
done
```
We use this baseline model to generate pseudo labels on Cityscapes dataset by running the following code. Add `--color_jitter` if you want to generate confident pseudo label. It will take quite a long time. We provide our generated [pseudo label](https://drive.google.com/file/d/1UOtX1d-J3smtxA3C_ygnH16GSlUnGgIA/view?usp=sharing) and [confident pseudo label](https://drive.google.com/file/d/1tSeG1O_GiYub-z_xCh8xf6NrXOt8gOBL/view?usp=sharing). Please put them in `mmdetection/data/pseudo/annotations/`.
```bash
python3 tools/pseudo_label.py --config=configs/smpl/tune.py --image_folder=/path/to/cityscapes/ --output_folder=results/ --ckpt /path/to/baseline/ckpt/
```
Again, we resume from checkpoint.pt, and use 500 labeled images from MPI-INF-3DHP + pseudo labeled data from Cityscapes dataset to train our model.
```bash
# For pseudo label model
python3 tools/train.py configs/smpl/tune_mpi_semi.py.py --load_pretrain ./data/checkpoint.pt --seed 1111
python3 tools/train.py configs/smpl/tune_mpi_semi.py --seed 1111

# For confident pseudo label model
python3 tools/train.py configs/smpl/tune_mpi_conf_semi.py --load_pretrain ./data/checkpoint.pt --seed 1111
i=0
while [ $i -le 5 ]
do
    python3 tools/train.py configs/smpl/tune_mpi_conf_semi.py --seed 1111
    i=$(($i+1))
done
```

All the checkpoints, evaluation results and logs would be saved to `./mmdetection/work_dirs`.

## References
```
@Inproceedings{jiang2020mpshape,
  Title          = {Coherent Reconstruction of Multiple Humans from a Single Image},
  Author         = {Jiang, Wen and Kolotouros, Nikos and Pavlakos, Georgios and Zhou, Xiaowei and Daniilidis, Kostas},
  Booktitle      = {CVPR},
  Year           = {2020}
}
```

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
	     Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
	     Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
	     Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
	     Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
	     and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```
