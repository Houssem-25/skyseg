# SkySeg

This is my implementation of the sky segmentation network

## ⚙️ Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution
The nvidia driver version is important because it selectes the cuda version. The nvidia driver version I'm using is 470 which yield a cuda version of 11.4
I have tested the compatibilty of the cuda version with torch and mmesegmentation and everything works well with this build.
```shell
conda create --name skyseg python==3.8
pip install setuptools==59.5.0
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install "mmsegmentation>=1.0.0"
pip install tensorboardX tqdm
pip install git+https://github.com/cocodataset/panopticapi.git
```

## 💾 SkySegmnetation GUI APP
To use the GUI app for infrence run the following command : 
```shell
python gui_app.py 
```

Upload a demo image and hit segement !

![1.png](assets%2F1.png)

![2.png](assets%2F2.png)


## 💾 Inference on a single image
In order to test the model on a single image run the following commend without gui
```shell
python single_img_inference.py --data_root demo.jpeg
```
The first output is for the model that was trained on Cityscapes and the 2nd output is for the model that was trained on ADE20K
## 💾 Evaluate on cityscapes
First download the dataset and extract it the files should have the following structre:

```shell
- cityscapes
  -leftImg8bit
      -train
      -val
      -test
  -gtFine
      -train
      -val
```
The initial solution for the skysegmentation network was based on ensemble methods more specifically : [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating).

Two models were considred: [SegFormer](https://arxiv.org/abs/2105.15203) trained on Cityscapes and the 2nd model trained on ADE20K.

Two methods were considered for aggregation : adding the masks or multipling the masks.

Use the following command line to run the evaluation 

### Bagging : 
```shell
python evaluate_sky_cityscapes.py --data_root [Path to cityscapes dataset]
```

### Single model
```shell
python evaluate_sky_cityscapes.py --data_root [Path to cityscapes dataset] --single_model
```


The obtained results : 


| `Model`               | Acc   | 
|-----------------------|-------|
| **Trained on CS**     | 99.56 | 
| **Trained on ADE20K** | 77.61 | 
| **Bagging OR**        | 99.39 | 
| **Bagging AND**       | 99.35 | 

Without surprise the model trained on CS and evaluated on CV yield great results. 

The model trained on ADE20K present acceptable results.

Bagging is greatly impacted by the performance of ADE20K model. 

## 💾 Evaluate on COCO 
First download the dataset and extract it the files should have the following structre:

[2017 Val images ](http://images.cocodataset.org/zips/val2017.zip)

[2017 Panoptic Train/Val annotations](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip)

```shell
- coco
  -val2017
      -000001.jpg
      ...
      -00000N.jpg
  -panoptic_val2017
      -000001.png
      ...
      -00000N.png
```
Use the following command line to run the evaluation 

### Bagging : 
```shell
python evaluate_sky_coco.py --data_root [Path to cityscapes dataset]
```

### Single model
```shell
python evaluate_sky_coco.py --data_root [Path to cityscapes dataset] --single_model
```


The obtained results : 


| `Model`               | Acc   | 
|-----------------------|-------|
| **Trained on CS**     | 93.44 | 
| **Trained on ADE20K** | 91.30 | 
| **Bagging OR**        | 90.84 | 
| **Bagging AND**       | 93.89 | 

As observed the model trained on ADE20K achieves competitive results on this dataset. 

The model trained using cityscapes dataset presents good generalization.

Bagging do not improve the further the results obtained by cityscpaes.

In conclusion : 
- The cityapes model achieves good generalization.
- Bagging solution is not required for our setting as the cityscapes model achieves good generalization

## 💾 Training a new model

In order to train a new model lauch the following command : 

```shell
python train.py --data_root /home/houssem/PhD/datasets/cityscapes --single_model
```

## ToDo: 
1- Export to onnx 
2- Retrain the model on the latest segment anything dataset and fine tune on only sky.
3- Adapt the model to the hardware to be used (More or less paramters, prunning, quantization).
