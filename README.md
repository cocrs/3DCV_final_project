# 3DCV final: Semi-spervised Learning for Reconstruction of Multiple Humans from a single image

## Installation instructions

Our project is base on the following repository. Please follow the instructions at their github to complete basic project installation first, including [Installation instructions](https://github.com/JiangWenPL/multiperson#installation-instructions) and [Fetch data](https://github.com/JiangWenPL/multiperson#fetch-data):
**Coherent Reconstruction of Multiple Humans from a Single Image**  
[Wen Jiang](https://jiangwenpl.github.io/)\*, [Nikos Kolotouros](https://www.seas.upenn.edu/~nkolot/)\*, [Georgios Pavlakos](https://www.seas.upenn.edu/~pavlakos/), [Xiaowei Zhou](http://www.cad.zju.edu.cn/home/xzhou/), [Kostas Daniilidis](http://www.cis.upenn.edu/~kostas/)  
CVPR 2020
[[paper](https://arxiv.org/pdf/2006.08586.pdf)] [[github](https://github.com/JiangWenPL/multiperson)]

##### We encountered some problems in Fetch data and we write at bellow.
In Fetch data, we need to follow the instructions [here](https://github.com/vchoutas/smplx/tree/master/tools) to convert the models to be compatible with python3. Only the "Removing Chumpy objects" part is necessary (Notice: this command needs to run under python2.7). After processing, we have to rename the file `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `SMPL_NEUTRAL.pkl` and put them under `mmdetection/data/smpl`


## Environment
Our testing is on Ubuntu 18.04 using 2080ti.

## Prepare datasets
Please refer to [DATASETS.md](https://github.com/JiangWenPL/multiperson/blob/master/DATASETS.md) for the preparation of the dataset files. We use Panoptic for evaluation and MPI-INF-3DHP for training. You can download Panoptic only if you want to test the evaluation code because MPI-INF-3DHP is really large after extracting all frames (about 550GB). You will also need the unlabeled dataset [Cityscapes](https://www.cityscapes-dataset.com/) we used if you want to run our semi-supervised training code.

##### We encountered some problems and we write at bellow.
Panoptic:
* While downloading Panoptic using the script from [panoptic-toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox), it is normal to see that some links are not available, and that does not affect the following processes. (Notice: the required image format in multiperson is png, so please specify `./scripts/extractAll.sh [sequence] png` while extracting frames using panoptic-toolbox)
* We were unable to use the preprocess code multiperson provided. We adjusted it ([adjusted code]()) to only extract the frames they need and use their processed annotation files to evaluate. The way to run our adjusted code is the same as the original code.

MPI-INF-3DHP:
* There are two links in the [official website of the dataset](https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/). The one on the left(MuCo-3DHP Scripts) is the one we want.
* We only get videos after downloading. Like Panoptic, we need to use the preprocess code to extract the frames ([adjusted code]()).
* Pose and Shape npz files are also needed, but not provided in this multiperson repo. We find it in another repo [link](http://visiondata.cis.upenn.edu/spin/dataset_extras.tar.gz). We need the `mpi_inf_3dhp_train.npz`. Please put it in `mmdetection/data/mpi_inf_3dhp/extras`

## Results


## Run evaluation code
You could use our pretrained checkpoint to evaluate on Panoptic.

Example usage:
```
cd mmdetection
python3 tools/full_eval.py configs/smpl/tune.py haggling --ckpt /path/to/ckpt
```

The ```haggling``` option can be replaced with other dataset or sequences based on the type of evaluation you want to perform:
- `mafia`: mafia sequence of Panoptic
- `ultimatum`: ultimatum sequence of Panoptic
- `haggling`: haggling sequence of Panoptic

Regarding the evaluation:
- For Panoptic, the command will compute the MPJPE for each sequence.

## Run training code

Please make sure you have downloaded checkpoint.pt from multiperson and put it in the folder `data`.
We resume from checkpoint.pt and first finetune on 500 images we sampled from MPI-INF-3DHP as our baseline. The annotations for 500 images we sampled is [here](https://drive.google.com/file/d/15MWagBYX4HUAMRuNihpA2qlAW3U-DmKx/view?usp=sharing). Please put it in `mmdetection/data/pseudo/annotations`
```bash
python3 tools/train.py configs/smpl/tune_mpi.py --load_pretrain ./data/checkpoint.pt --seed 1111
i=0
while [ $i -le 5 ]
do
    python3 tools/train.py configs/smpl/tune_mpi.py --seed 1111
    i=$(($i+1))
done
```
We use this baseline model to generate pseudo labels on Cityscapes dataset by this [code](). Add `--color_jitter` if you want to generate confident pseudo label. It will take quite a long time.
```bash
python3 tools/pseudo_label.py --config=configs/smpl/tune.py --image_folder=/path/to/cityscapes/ --output_folder=results/ --ckpt /path/to/baseline/ckpt/
```
Again, we resume from checkpoint.pt, and use 500 labeled images from MPI-INF-3DHP + pseudo labeled data from Cityscapes dataset to train our model.
```bash
# For pseudo label model
python3 tools/train.py configs/smpl/tune_mpi_semi.py.py --load_pretrain ./data/checkpoint.pt --seed 1111
python3 tools/train.py configs/smpl/tune_mpi_semi.py --seed 1111

# For confident pseudo label model
python3 tools/train.py configs/smpl/tune_mpi_cond_semi.py --load_pretrain ./data/checkpoint.pt --seed 1111
i=0
while [ $i -le 5 ]
do
    python3 tools/train.py configs/smpl/tune_mpi_cond_semi.py --seed 1111
    i=$(($i+1))
done
```

All the checkpoints, evaluation results and logs would be saved to `./mmdetection/work_dirs/tune`.

## Citing
If you find this code useful for your research or the use data generated by our method, please consider citing the following paper:

	@Inproceedings{jiang2020mpshape,
	  Title          = {Coherent Reconstruction of Multiple Humans from a Single Image},
	  Author         = {Jiang, Wen and Kolotouros, Nikos and Pavlakos, Georgios and Zhou, Xiaowei and Daniilidis, Kostas},
	  Booktitle      = {CVPR},
	  Year           = {2020}
	}

## Acknowledgements

This code uses ([mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmdetection)) as backbone.
We gratefully appreciate the impact these libraries had on our work. If you use our code, please consider citing the [original paper](https://arxiv.org/abs/1906.07155) as well.
