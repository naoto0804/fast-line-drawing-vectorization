# Fast Instance Segmentation for Line Drawing Vectorization


This page is for the paper titled above appeared in BigMM2019 (short).

Here is the example of our results.

<p align='center'>
  <img src='sample_results/input_26481.png' width="400px">
  <img src='sample_results/ours_26481_0_95.pdf' width="400px">
</p>

<p align='center'>
  <img src='sample_results/input_4657882177994752.png' width="400px">
  <img src='sample_results/ours_4657882177994752_0_97.pdf' width="400px">
</p>


## Requirements
Python 3.5+, Pytorch 1.1.0, CUDA10
Since this project heavily relies on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), building is a bit complicated process.

```
pip install -r requirements.txt

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# Build mask-RCNN
cd $INSTALL_DIR
python setup.py build develop

```

## Download
### Model
For testing our model, please download pre-trained weights from [here](https://drive.google.com/open?id=1YOQKuL323r3utgyukOZvl6th98yiGOXr)

### Data
For testing our model, please do the following;

1. Download datasets from [vectornet](https://github.com/byungsook/vectornet) and decompress it (e.g., `data/DATASET/*`).
2. Do preprocessing (SVG to PNG format, getting list of image ids.)
```
python preprocess.py --root data
```

## Usage
After the installation, please do `$cd scripts`.

### Demo
This script will test our model on some images from the test subset.

```
CUDA_VISIBLE_DEVICES=<gpu_id> python demo.py <config> <checkpoint>
```

- If you need the result in SVG format, please use `--mode svg`.
- If you want to change the directory, please use `--output_dir <output_dir>`

### Test
This script will compute evaluation metrics using our model.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python eval.py <config> <checkpoint>
```

### Train
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py <config>
```

## Citation

If you find this code or dataset useful for your research, please cite our paper:

```
@inproceedings{inoue_2019_bigmm,
  author = {Inoue, Naoto and Yamasaki, Toshihiko},
  title = {Fast Instance Segmentation for Line Drawing Vectorization},
  booktitle = {IEEE International Conference on Multimedia Big Data(BigMM)},
  month = {September},
  year = {2019}
}
```
