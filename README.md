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


## OpenPCDet Env
- python 3.10
- pytorch 3.10 + cu115, etc
- requirements.txt
- setup.py

What I did:
```
conda create -n open python=3.10
pip install https://download.pytorch.org/whl/cu115/torch-1.11.0%2Bcu115-cp310-cp310-linux_x86_64.whl#sha256=4f287b35e4a
c25589b1d86dc94cbf038048cea7aed547a6e7ba979915fa79d11
pip install https://download.pytorch.org/whl/cu115/torchvision-0.12.0%2Bcu115-cp310-cp310-linux_x86_64.whl#sha256=b34c7cbd68e4f86edba0aca3f08a0efb66c2f1225dfe11ec7d15f120b8eb1bbc
pip install https://download.pytorch.org/whl/cu115/torchaudio-0.11.0%2Bcu115-cp310-cp310-linux_x86_64.whl#sha256=0af08a68660f9bbe9cab050a95cb0648bb430ba0fd6e065ca7735c181063d0e0

pip install -r requirements.txt
sudo apt install gcc-9 g++-9
export CC=gcc-9
export CXX=g++-9
python setup.py develop
```

## `python3 process_cvat_data.py`
Database pedestrian: 10029
Database car: 12946
Database signs: 14681
Database barricades: 10449
Database barrels: 22990
Database railroad_bar_down: 1996
Database deers: 1634
## To train
`python3 train.py --cfg_file cfgs/kitti_models/pointpillars_half_backbone.yaml --epochs 160 --ckpt_save_interval 10`
`python demo.py --cfg_file cfgs/kitti_models/pointpillars_cvat.yaml --ckpt ../checkpoint_epoch_160.pth`



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
