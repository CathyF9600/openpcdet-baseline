import onnx
from onnxconverter_common import float16

# onnx_fp32_models = ["backbone_2d.onnx", "dense_head.onnx", "map_to_bev_module.onnx", "vfe.onnx"]
onnx_fp32_models = ["pointpillars_cvat.onnx"]

for model_path in onnx_fp32_models:
    model = onnx.load(model_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, model_path.replace(".onnx", "_fp16.onnx"))