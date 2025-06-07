import argparse
import glob
from pathlib import Path
import os
import pickle
import torch
import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.custom.custom_dataset import CustomDataset


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser for ONNX conversion')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillars.yaml',
                        help='specify the config file')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def load_model_and_data(cfg, args, logger):
    """
    Loads the model based on the configuration and data.
    """
    # Dataset initialization
    dataset = CustomDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=None,
        training=False,
        logger=logger,
    )

    # Build the model from the configuration and load the checkpoint
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    return model, dataset


def export_to_onnx(model, dataset):
    """
    Exports the given PyTorch model to ONNX format as a single unified checkpoint.
    """
    # Extract a single batch from the dataset to get the correct input size
    with torch.no_grad():
        for idx, data_dict in enumerate(dataset):
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            # Prepare the input tensors for the model
            voxel_features = data_dict['voxels']
            voxel_num_points = data_dict['voxel_num_points']
            voxel_coords = data_dict['voxel_coords']
            batch_size = data_dict['batch_size']

            # Define the input names and dynamic axes for ONNX export
            input_names = ['voxel_features', 'voxel_num_points', 'voxel_coords', 'batch_size']
            dynamic_axes = {
                'voxel_features': {0: 'pillar_num'},
                'voxel_num_points': {0: 'pillar_num'},
                'voxel_coords': {0: 'pillar_num'},
            }

            # Export the entire model to ONNX
            torch.onnx.export(
                model,
                (voxel_features, voxel_num_points, voxel_coords, batch_size),
                "onnx/pointpillars_cvat.onnx",
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=['batch_cls_preds', 'batch_box_preds', 'cls_preds_normalized'],
                dynamic_axes=dynamic_axes,
            )

            print(f"OK")
            break  # Convert using only the first sample


def main():
    args, cfg = parse_config()
    
    # Create logger
    logger = common_utils.create_logger()
    logger.info('-----------------PyTorch to ONNX Conversion-------------------------')

    # Load the model and dataset
    model, dataset = load_model_and_data(cfg, args, logger)

    # Step 1: Export the model to ONNX format
    export_to_onnx(model, dataset)

    logger.info('ONNX conversion completed successfully.')


if __name__ == '__main__':
    main()