import onnx
import onnxsim
import torch
import torch.nn as nn
import argparse
import os
import openvino as ov

from onnx.version_converter import convert_version
from onnx import TensorProto
from pathlib import Path


parser = argparse.ArgumentParser()
# parser.add_argument("input_name", type=str, help="Model file name of the input model (must be in ONNX format and include the .onnx extension)")
# parser.add_argument("output_name", type=str, help="Desired name of the output model. Do not include any file extensions")
# parser.add_argument("-c", "--conf_thresh", type=float, help="Minimum class confidence threshold across all classes for the model. Default is 0.30")


args = parser.parse_args()
# input_model_name = args.input_name
input_model_name = '/home/autoronto/dev/openpcdet-training-pipeline/tools/onnx/pointpillars_cvat_fp16.onnx'
# output_model_name = args.output_name
output_model_name = 'test'
# CONF_THRESH = 0.30 if not args.conf_thresh else args.conf_thresh

assert input_model_name.endswith(".onnx"), "Input model must be .onnx file"

Path("./intermediate_onnx").mkdir(exist_ok=True)
Path("./output_models").mkdir(exist_ok=True)

model = onnx.load(input_model_name)

# last_dim = model.graph.output[0].type.tensor_type.shape.dim[2].dim_value

# assert model.graph.output[0].type.tensor_type.shape.dim[1].dim_value == 107100, "Output shape must be (1, 107100, 3+1+num_cls) for yolov5"
# assert last_dim >= 5, "Must have at least one output class"

rem = []
for node in model.graph.output:
    if node.name != "output.1":
        rem.append(node)

for node in rem:
    model.graph.output.remove(node)
    print(f"Removed output node {node.name}")

onnx.save(model, "./intermediate_onnx/yolov5_no_int_outs.onnx")

# class NMSPreprocess(nn.Module):
#     def forward(self, x):
#         x = x[x[..., 4] > CONF_THRESH]
#         return x

import time

class PPModel(nn.Module):
    def forward(self, batch_dict):
        """
        Args:
            batch_dict: Dictionary containing input tensors.
        Returns:
            cls_preds: Tensor of class predictions.
            box_preds: Tensor of box predictions.
        """
        recall_dict = {}
        pred_dicts = []

        for index in range(1):
            start_time = time.time()

            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].dim() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].dim() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                print("a")
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]
                src_cls_preds = cls_preds
                
                if not batch_dict['cls_preds_normalized']:
                    print("c")
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                print("b")
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            cls_preds, label_preds = torch.max(cls_preds, dim=-1)

            if batch_dict.get('has_class_labels', False):
                label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                label_preds = batch_dict[label_key][index]
            else:
                label_preds = label_preds + 1 

            end_time = time.time()
            print(f"satvacha sigmoid: {end_time - start_time}")
            
            return cls_preds, box_preds

# preproc_model = NMSPreprocess()
preproc_model = PPModel()

x = torch.rand(1, 749952, 7)
y = torch.rand(1, 749952, 7)

import numpy as np

data_dict = {
                'batch_cls_preds' : x,
                'batch_box_preds' : y,
                'cls_preds_normalized': torch.tensor([False])
            }

torch.onnx.export(preproc_model, data_dict, "./intermediate_onnx/PP_pointcloud.onnx",
                input_names=["yolo_out"],
                output_names=["prediction"],
                dynamic_axes={
                    "prediction": {1: "N"}
                    })

pre_model = onnx.load("./intermediate_onnx/PP_pointcloud.onnx")
pre_model_sim, check = onnxsim.simplify(pre_model)
onnx.save(pre_model_sim, "./intermediate_onnx/PP_pointcloud_sim.onnx")
target_version = 19
yolo_model = onnx.load("./intermediate_onnx/yolov5_no_int_outs.onnx")
pre_model = onnx.load("./intermediate_onnx/PP_pointcloud_sim.onnx")

core_model = convert_version(yolo_model, target_version)
onnx.checker.check_model(core_model)

pre_model = convert_version(pre_model, target_version)
onnx.checker.check_model(pre_model)

core_model.ir_version = 8
pre_model.ir_version = 8

out_model = onnx.compose.merge_models(core_model, pre_model, [("output.1", "yolo_out")])
onnx.checker.check_model(out_model)
onnx.save(out_model, f"./output_models/{output_model_name}_confthresh{CONF_THRESH}.onnx")
print(f"\nSuccessfully saved ONNX model to {os.getcwd()}/output_models/{output_model_name}_confthresh{CONF_THRESH}.onnx")

ov_mod = ov.convert_model(f"./output_models/{output_model_name}_confthresh{CONF_THRESH}.onnx")
ov.save_model(ov_mod, f"./output_models/{output_model_name}_confthresh{CONF_THRESH}.xml")
print(f"\nSuccessfully saved OpenVINO IR model to {os.getcwd()}/output_models/{output_model_name}_confthresh{CONF_THRESH}.xml")
print(f"Successfully saved weights to {os.getcwd()}/output_models/{output_model_name}_confthresh{CONF_THRESH}.bin")
print(f"\nIntermediate ONNX model files can be found in {os.getcwd()}/intermediate_onnx/")
