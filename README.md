## OpenPCDet Training Pipeline

This repository holds our pipeline for training 3D Object Detection models using the OpenPCDet Toolkit. 

### Setup instructions
```
git clone https://gitlab.com/autoronto-ADC2/perception/3d-object-detection/openpcdet-training-pipeline.git
```

```
cd openpcdet-training-pipeline
```

``` 
pip install -r requirements.txt
```

```
python setup.py develop
```

### Run Demo
Download pointpillars weights:
```
pip install gdown
gdown 1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm
```

To download other weights, go to the google drive page from the model zoo, then gdown <id> where <id> is the string after d/ in the url.

From the `/tools` folder:
```
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../pointpillar_7728.pth --data_path ../demo/data/kitti/000008.bin
```

```
python demo.py --cfg_file cfgs/kitti_models/pointpillars_cvat.yaml --ckpt ../output/kitti_models/pointpillars_cvat/default/ckpt/checkpoint_epoch_80.pth
```

### Training
Create dataset infos
```
python -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml
```
Start training
```
cd tools
python train.py --cfg_file cfgs/kitti_models/pointpillars_cvat.yaml --epochs 80
```
# openpcdet-baseline
