import argparse
import time
import torch
import yaml
import numpy as np
import onnxruntime as ort
import open3d as o3d
from pathlib import Path
from CustomDatasetInference import CustomDatasetInference
from easydict import EasyDict
from pcdet.utils import common_utils

cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0


''' ---------------------------------- Visualization ---------------------------------- '''

def draw_lidar(pc, color=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
    Returns:
        pcd: open3d point cloud object
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (3, 1)).T)
    return pcd


def draw_boxes3d(gt_boxes3d, color=(1, 1, 1), color_list=None, scores_list=None, label_name_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        color: RGB value tuple in range (0,1), box line color
        color_list: a list of RGB tuple, if not None, overwrite color.
        scores_list: list of scores for each box
        label_name_list: list of label names for each box
    Returns:
        line_set_list: list of open3d line set objects
    '''
    line_set_list = []
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                 [4, 5], [5, 6], [6, 7], [7, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [color if color_list is None else color_list[n]] * len(lines)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(b)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set_list.append(line_set)
    return line_set_list


def visualize_dets(data_dict, pred_dict):
    """
    Visualizes the detections using the given data and predictions with Open3D.
    """
    # Get the point cloud data
    points = data_dict['points'][:, 1:]
    
    # Get the predicted boxes
    pred_boxes = np.array(pred_dict[0]['pred_boxes'])
    pred_scores = np.array(pred_dict[0]['pred_scores'])
    pred_labels = np.array(pred_dict[0]['pred_labels'])

    new_pred_boxes = []
    colour_list = []
    scores_list = []
    label_name_list = []

    label_to_colour = {
        0: (255/255, 255/255, 255/255),  # unk
        1: (223/255, 144/255, 134/255),  # car
        2: (190/255, 244/255, 163/255),  # pedestrian
        3: (248/255, 248/255, 168/255),  # barrels
        4: (112/255, 112/255, 235/255),  # deers
        5: (150/255, 150/255, 179/255),  # signs
        6: (224/255, 145/255, 238/255),  # barricades
        7: (200/255, 81/255, 148/255),  # railroad bar down
    }

    label_to_name = { 
        0: 'unk',
        1: 'car',
        2: 'pedestrian',
        3: 'barrel',
        4: 'deer',
        5: 'sign',
        6: 'barricade',
        7: 'railroad_bar_down'
    }

    for i in range(len(pred_boxes)):
        box = pred_boxes[i]
        score = pred_scores[i]
        label = pred_labels[i]
        
        colour_list.append(label_to_colour[label])
        scores_list.append(score)
        label_name_list.append(label_to_name[label])
        
        if score > 0.3:
            x, y, z, z_size, x_size, y_size, zrot = box

            zrot = np.deg2rad(zrot)
            # Convert to box format: [n,8,3] for XYZs of the box corners
            corners = np.array([
                [x - x_size / 2, y - y_size / 2, z - z_size / 2],
                [x + x_size / 2, y - y_size / 2, z - z_size / 2],
                [x + x_size / 2, y + y_size / 2, z - z_size / 2],
                [x - x_size / 2, y + y_size / 2, z - z_size / 2], 
                [x - x_size / 2, y - y_size / 2, z + z_size / 2],
                [x + x_size / 2, y - y_size / 2, z + z_size / 2],
                [x + x_size / 2, y + y_size / 2, z + z_size / 2],
                [x - x_size / 2, y + y_size / 2, z + z_size / 2]
            ])

            new_pred_boxes.append(np.array(corners))

    # Create Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0  # Adjust this value as needed (default is 5.0)
    vis.get_render_option().line_width = 4.0
    vis.get_render_option().background_color = np.asarray([0, 0, 0])

    # Add point cloud
    pcd = draw_lidar(points)
    vis.add_geometry(pcd)

    # Add bounding boxes
    line_sets = draw_boxes3d(new_pred_boxes, color_list=colour_list, scores_list=scores_list, label_name_list=label_name_list)
    for line_set in line_sets:
        vis.add_geometry(line_set)

    # Set camera view
    view_control = vis.get_view_control()
    view_control.set_front([-0.95838680584881764, 0.017337049394112139, 0.28494588449950803])
    view_control.set_lookat([25.853971991667837, 1.4352717527847894, 1.5323786256376781])
    view_control.set_up([0.28453597452260831, -0.022786481461139041, 0.95839451973866019])
    view_control.set_zoom(0.31999999999999962)

    # Run visualization
    vis.run()
    vis.destroy_window()


''' ---------------------------------- Inference ---------------------------------- '''

def nms_cpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """Nms function with cpu implementation."""
    def compute_iou(box1, boxes):
        """Compute IoU between a reference box and an array of boxes."""
        box1_x1 = box1[:, 0]
        box1_y1 = box1[:, 1]
        box1_x2 = box1[:, 0] + box1[:, 3]  # x1 + x_size
        box1_y2 = box1[:, 1] + box1[:, 4]  # y1 + y_size

        boxes_x1 = boxes[:, 0]
        boxes_y1 = boxes[:, 1]
        boxes_x2 = boxes[:, 0] + boxes[:, 3]
        boxes_y2 = boxes[:, 1] + boxes[:, 4]

        inter_area = (torch.min(box1_x2, boxes_x2) - torch.max(box1_x1, boxes_x1)).clamp(0) * \
                        (torch.min(box1_y2, boxes_y2) - torch.max(box1_y1, boxes_y1)).clamp(0)
        box1_area = abs(box1[:, 3] * box1[:, 4])
        boxes_area = abs(boxes[:, 3] * boxes[:, 4])

        iou = inter_area / (box1_area + boxes_area - inter_area)
        return iou
    
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order]
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        ious = compute_iou(boxes[0].unsqueeze(0), boxes[1:])
        order = order[1:][ious <= thresh]
        boxes = boxes[1:][ious <= thresh]

    keep = torch.tensor(keep, dtype=torch.long)
    return keep, None


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = nms_cpu(
            boxes=boxes_for_nms[:, 0:7], scores=box_scores_nms, thresh=nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def post_processing(model_cfg, batch_dict):
    """Post-processes the model predictions."""
    post_process_cfg = model_cfg.MODEL.POST_PROCESSING
    batch_size = batch_dict['batch_size']
    recall_dict = {}
    pred_dicts = []
    for index in range(batch_size):
        if batch_dict.get('batch_index', None) is not None:
            assert batch_dict['batch_box_preds'].shape.__len__() == 2
            batch_mask = (batch_dict['batch_index'] == index)
        else:
            assert batch_dict['batch_box_preds'].shape.__len__() == 3
            batch_mask = index

        box_preds = batch_dict['batch_box_preds'][batch_mask]
        src_box_preds = box_preds
        
        if not isinstance(batch_dict['batch_cls_preds'], list):
            cls_preds = batch_dict['batch_cls_preds'][batch_mask]

            src_cls_preds = cls_preds
            assert cls_preds.shape[1] in [1, len(model_cfg.CLASS_NAMES)]

            if not batch_dict['cls_preds_normalized']:
                cls_preds = torch.sigmoid(cls_preds)
        else:
            cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
            src_cls_preds = cls_preds
            if not batch_dict['cls_preds_normalized']:
                cls_preds = [torch.sigmoid(x) for x in cls_preds]

        if True:
            cls_preds, label_preds = torch.max(cls_preds, dim=-1)
            if batch_dict.get('has_class_labels', False):
                label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                label_preds = batch_dict[label_key][index]
            else:
                label_preds = label_preds + 1 
            
            selected, selected_scores = class_agnostic_nms(
                box_scores=cls_preds, box_preds=box_preds,
                nms_config=post_process_cfg.NMS_CONFIG,
                score_thresh=post_process_cfg.SCORE_THRESH
            )

            if post_process_cfg.OUTPUT_RAW_SCORE:
                max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                selected_scores = max_cls_preds[selected]

            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]
                    
        record_dict = {
            'pred_boxes': final_boxes,
            'pred_scores': final_scores,
            'pred_labels': final_labels
        }
        pred_dicts.append(record_dict)

    return pred_dicts, recall_dict


def onnx_inference(cfg, args, data_dict):
    """Runs inference using the ONNX models for PointPillars."""
    # Load the ONNX models
    if args.fp16:
        pointpillars_cvat_sess = ort.InferenceSession("onnx/pointpillars_cvat_fp16.onnx")
    else:
        pointpillars_cvat_sess = ort.InferenceSession("onnx/pointpillars_cvat.onnx")

    start = time.time()

    # Model inference
    voxels = data_dict['voxels']
    voxel_num_points = data_dict['voxel_num_points'].astype(np.float32)
    voxel_coords = data_dict['voxel_coords'].astype(np.float32)
    
    if args.fp16:
        voxels = voxels.astype(np.float16)
        voxel_num_points = voxel_num_points.astype(np.float16)
        voxel_coords = voxel_coords.astype(np.float16)

    inputs = {
        'voxel_features': voxels,
        'voxel_num_points': voxel_num_points,
        'voxel_coords': voxel_coords,
        'batch_size': np.array(data_dict['batch_size'], dtype=np.int64)
    }
    batch_cls_preds, batch_box_preds, cls_preds_normalized = pointpillars_cvat_sess.run(None, inputs)

    # Post-process detections (fp32)
    data_dict['batch_cls_preds'] = torch.tensor(batch_cls_preds).float()
    data_dict['batch_box_preds'] = torch.tensor(batch_box_preds).float()
    data_dict['cls_preds_normalized'] = cls_preds_normalized
    pred_dicts, _ = post_processing(cfg, data_dict)

    end = time.time()

    return pred_dicts, end - start


def benchmark_inference(cfg, args, data_dict, vis=False):
    """Benchmarks the inference and optionally visualizes the results."""
    num_trials = 20
    avg_time = 0.0
    pred_dicts = None
    for i in range(num_trials):
        pred_dicts, time_elapsed = onnx_inference(cfg, args, data_dict)
        avg_time += time_elapsed
    avg_time /= num_trials

    print(f'Average CPU inference time over {num_trials} trials: {avg_time} seconds')

    if vis:
        visualize_dets(data_dict, pred_dicts)


''' ---------------------------------- Configs ---------------------------------- '''

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser for ONNX inference')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillars.yaml',
                        help='specify the config file')
    parser.add_argument('--fp16', action='store_true', help='use float16 inference')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def load_data(cfg, args, logger):
    """Loads the dataset and returns the first data sample."""
    # Dataset initialization
    dataset = CustomDatasetInference(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=None,
        training=False,
        logger=logger,
    )
    
    # Get first data sample
    data_dict = dataset[0]
    data_dict = dataset.collate_batch([data_dict])
    return data_dict


def main():
    args, cfg = parse_config()
    
    # Create logger
    logger = common_utils.create_logger()
    logger.info('-----------------ONNX Inference-------------------------')

    # Load the dataset and retrieve data
    data_dict = load_data(cfg, args, logger)

    # Step 1: Run ONNX inference
    benchmark_inference(cfg, args, data_dict, vis=True)

    logger.info('ONNX inference completed successfully.')


if __name__ == '__main__':
    main()