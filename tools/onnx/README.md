# Uility scripts for ONNX export and inference of trained model

## 1) Convert PyTorch to ONNX (requires openpcdet setup)
I give you the onnx checkpoints in this repo so you can skip this step

If you want to do it yourself, download the trained PointPillars model checkpoint from [https://drive.google.com/file/d/16_3ly2FAgVxUMqfZ_3m_k0yl0n_OlxmJ/view?usp=sharing](https://drive.google.com/file/d/16_3ly2FAgVxUMqfZ_3m_k0yl0n_OlxmJ/view?usp=sharing) into `tools/onnx`.

`cd tools`

`PYTHONPATH=".." python onnx/pth_to_onnx.py --cfg_file onnx/pointpillars_cvat_cpu.yaml --ckpt onnx/checkpoint_epoch_80.pth`

If you want to convert your onnx models to fp16, run `python3 fp32_to_fp16.py`

## 2) Run ONNX Inference
Should not require any GPU / Cuda, only needs CPU.

First, download the data we are running inference on.
On the same level as your openpcdet folder make a folder called `datasets`.
Inside datasets, extract [https://drive.google.com/file/d/1XyGlgXEWOsKejYvC9wq5XqM5WKp3lSGX/view?usp=sharing](https://drive.google.com/file/d/1XyGlgXEWOsKejYvC9wq5XqM5WKp3lSGX/view?usp=sharing) into it.

Optionally, if you want to try inference on different bags, download the other bag and change the path in `tools/cfgs/dataset_configs/custom_dataset.yaml`.

`cd tools`

`PYTHONPATH=".." python onnx/onnx_infer.py --cfg_file onnx/pointpillars_cvat_cpu.yaml`

Note: to run inference with fp16, simply add the `--fp16` argument to the onnx inference script.

