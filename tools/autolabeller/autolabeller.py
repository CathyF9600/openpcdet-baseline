import argparse
from pathlib import Path
import os
import struct

from rosbags.highlevel import AnyReader

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import yaml
import shutil

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# --------------- UTILS ---------------
CLASS_TO_LABEL_ID = {
    "unk": 0,
    "car": 1,
    "pedestrian": 2,
    "construction_barrel": 3,
    "deer": 4,
    "other_sign": 5,
    "type_3_barricade": 6,
    "railroad_bar_down": 8
}

TYPE_TO_CVAT_CLASS = { 
    0: 'unk',
    1: 'car',
    2: 'pedestrian',
    3: 'construction_barrel',
    4: 'deer',
    5: 'other_sign',
    6: 'type_3_barricade',
    8: 'railroad_bar_down'
}

# Define the fixed categories
CATEGORIES = [
    {"name": "unknown", "parent": "", "attributes": []},
    {"name": "car", "parent": "", "attributes": []},
    {"name": "pedestrian", "parent": "", "attributes": []},
    {"name": "construction_barrel", "parent": "", "attributes": []},
    {"name": "deer", "parent": "", "attributes": []},
    {"name": "other_sign", "parent": "", "attributes": []},
    {"name": "type_3_barricade", "parent": "", "attributes": []},
    {"name": "railroad_bar_down", "parent": "", "attributes": []}
]

ATTRIBUTES = ["occluded"]

def generate_annotation(frame_id, tracks):
    """
    Generates a JSON annotation for a given frame.
    
    Parameters:
    - frame_id: The ID of the frame.
    - tracks: A list of dictionaries where each dictionary contains the track information for an object.
    
    Returns:
    - A dictionary representing the annotation for the frame.
    """
    annotations = []
    for i, track in tracks.items():
        xyz = track['xyz']
        lwh = track['lwh']
        yaw = track['yaw']
        type = track['type']
        annotation = {
            "id": i,
            "type": "cuboid_3d",
            "attributes": {
                "occluded": False,
                # "track_id": i,
                "keyframe": True,
                "outside": False,
            },
            "group": 0,
            "label_id": CLASS_TO_LABEL_ID[TYPE_TO_CVAT_CLASS[type]],
            "position": xyz,
            "rotation": [0.0, 0.0, yaw],
            "scale": lwh,
        }
        annotations.append(annotation)
    
    return {
        "id": str(frame_id).zfill(6),
        "annotations": annotations,
        "attr": {
            "frame": frame_id
        },
        "point_cloud": {
            "path": ""
        },
        "media": {
            "path": ""
        }
    }

def generate_datumaro_annotations(annotations):
    """
    Generates annotations for multiple frames.
    
    Parameters:
    - frames: A list of dictionaries, each containing 'frame_id' and 'tracks' information.
    
    Returns:
    - A JSON object with the complete annotation data.
    """
    
    json_data = {
        "info": {},
        "categories": {
            "label": {
                "labels": CATEGORIES,
                "attributes": ATTRIBUTES
            },
            "points": {
                "items": []
            }
        },
        "items": annotations
    }
    
    return json_data


def process_tracked_box(pred_boxes, pred_scores, pred_labels):
    cur_frame_dict = {}
    for idx, b in enumerate(pred_boxes):
        idx = int(idx)
        cur_frame_dict[idx] = {}
        cur_frame_dict[idx]['type'] = pred_labels[idx]
        cur_frame_dict[idx]['confidence'] = pred_scores[idx]
        # center_x, center_y, center_z = b[0], b[1], b[2]
        center_x, center_y, center_z = b[0], b[1], b[2]

        # Model output has a strange order
        dx, dy, dz = b[4], b[5], b[3]
        yaw = b[6]


        cur_frame_dict[idx]['xyz'] = [c.item() for c in [center_x, center_y, center_z]]
        cur_frame_dict[idx]['yaw'] = yaw.item()
        cur_frame_dict[idx]['lwh'] = [d.item() for d in [dx, dy, dz]]

    return cur_frame_dict

def get_centroid_and_stdev(point_cloud, search_box):
    '''
    Compute the centroid and stdev of the points within the search box

    Input:
        point_cloud: np.array of shape (N, 4)
        search_box: [x, y, z, l, w, h, 0, 0, yaw]
    Output:
        centroid: np.array of shape (3,)
        stdev: float
    '''

    # Extract bounding box parameters
    box_x, box_y, box_z, l, w, h, _, _, yaw = search_box

    # Compute rotation matrix for the yaw (rotation about the z-axis)
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,        0,       1]
    ])

    # Translate point cloud to bounding box coordinate frame
    translated_points = point_cloud[:, :3] - np.array([box_x, box_y, box_z])

    # Rotate the points into the bounding box's frame
    rotated_points = translated_points @ rotation_matrix.T

    # Check if the points are within the bounding box dimensions
    in_box_x = np.abs(rotated_points[:, 0]) <= l / 2
    in_box_y = np.abs(rotated_points[:, 1]) <= w / 2
    in_box_z = np.abs(rotated_points[:, 2]) <= h / 2

    in_box = in_box_x & in_box_y  # not incoporating z works better since z floats

    # Get the points inside the bounding box
    points_in_box = rotated_points[in_box]

    if len(points_in_box) == 0:
        return np.array([box_x, box_y, box_z]), 0.0
    
    # Compute the centroid of the points
    centroid = np.mean(points_in_box, axis=0)

    # Compute the variance of the points
    variance = np.var(points_in_box, axis=0)
    variance = np.sqrt(variance[0]**2 + variance[1]**2 + variance[2]**2)
    stdev = np.sqrt(variance)

    # debugging only: plot the points in the bounding box in 3d
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points_in_box[:, 0], points_in_box[:, 1], points_in_box[:, 2])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()
    # # save the figure
    # fig.savefig('points_in_box.png')

    # transform centroid back to the original frame
    centroid = np.dot(centroid, rotation_matrix) + np.array([box_x, box_y, box_z])

    return centroid, stdev

def num_pts_in_bbox(point_cloud, lidar_box):
    '''
    Gets the number of points from point cloud contained in the bounding box

    Input:
        point_cloud: np.array of shape (N, 4)
        lidar_box: [x, y, z, l, w, h, 0, 0, yaw]
    Output:
        int
    '''

    # Extract bounding box parameters
    box_x, box_y, box_z, l, w, h, _, _, yaw = lidar_box

    # Compute rotation matrix for the yaw (rotation about the z-axis)
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,        0,       1]
    ])

    # Translate point cloud to bounding box coordinate frame
    translated_points = point_cloud[:, :3] - np.array([box_x, box_y, box_z])

    # plot the translated points (bev)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(translated_points[:, 0], translated_points[:, 1], s=1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # plt.show()
    # # save the figure
    # fig.savefig('translated_points.png')

    # Rotate the points into the bounding box's frame
    rotated_points = translated_points @ rotation_matrix.T

    # plot the rotated points with the box
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(rotated_points[:, 0], rotated_points[:, 1], s=1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # # plot the box
    # box_corners_2d = np.array([
    #     [l/2, w/2],
    #     [l/2, -w/2],
    #     [-l/2, -w/2],
    #     [-l/2, w/2],
    # ])
    # ax.add_patch(plt.Polygon(box_corners_2d,
    #                   edgecolor='black', fill=False))
    # plt.show()
    # # save the figure
    # fig.savefig('rotated_points.png')

    # Check if the points are within the bounding box dimensions
    in_box_x = np.abs(rotated_points[:, 0]) <= l / 2
    in_box_y = np.abs(rotated_points[:, 1]) <= w / 2
    in_box_z = np.abs(rotated_points[:, 2]) <= h / 2

    # Combine the conditions for x, y, and z axes
    in_box = in_box_x & in_box_y  # not incoporating z works better since z floats
    in_box = in_box
    # Count the number of points inside the bounding box
    return np.sum(in_box)

def point_cloud_msg_to_np_array(msg):
    num_points = msg.width * msg.height
    # Create a NumPy array to hold the point cloud data
    points = np.zeros((num_points, 4), dtype=np.float32)
    # Iterate over each point in the point cloud
    for i in range(num_points):
        # Calculate the offset in the data array
        offset = i * msg.point_step
        # Extract the point data (x, y, z) from the message
        # Assume that the point cloud is in the format of (x, y, z, ...)
        # Adjust the offsets (0, 4, 8) if the format is different
        x, y, z, intensity = struct.unpack_from('ffff', msg.data, offset)
        points[i] = [x, y, z, intensity]
    return points

# --------------- Dataset ---------------

class MyDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, use_raw_pc=False, rosbag_path=None, rosbag_cfg=None, raw_pc_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=False, logger=logger, root_path="../"
        )
        self.use_raw_pc = use_raw_pc
        if (use_raw_pc and (raw_pc_path is None)):
            self.logger.error("Chose to use raw pc data but none provided")
            exit(-1)
        if ((not use_raw_pc) and (rosbag_path is None)):
            self.logger.error("Chose to use rosbag but none provided or rosbag cfg missing")
            exit(-1)

        if self.use_raw_pc:
            pc_path = Path(raw_pc_path)
            if not (pc_path.is_dir() and os.path.exists(pc_path)):
                raise FileNotFoundError("Point cloud data is not a path or is not found")
            self.pc_path = pc_path
            self.length = sum(1 for item in os.listdir(self.pc_path) if os.path.isfile(os.path.join(self.pc_path, item)))

        else:
            rosbag_path = Path(rosbag_path)
            if not (rosbag_path.is_dir() and os.path.exists(rosbag_path)):
                raise FileNotFoundError("Rosbag not found")
            self.rosbag_reader = AnyReader([Path(rosbag_path)])
            self.rosbag_reader.open()

            connections = [c for c in self.rosbag_reader.connections if c.topic == rosbag_cfg['fused_lidar_topic']]
            self.rosbag_iter = self.rosbag_reader.messages(connections=connections)
            with AnyReader([Path(rosbag_path)]) as temp_reader:
                temp_iter = temp_reader.messages(connections=connections)
                self.length = sum(1 for _ in temp_iter)
            

    def __len__(self):
        return self.length
    
    def readRosBag(self, frame_idx):
        # Keep reading rosbag until all necessary topics are extracted
        connection, timestamp, rawdata = next(self.rosbag_iter)
        msg = self.rosbag_reader.deserialize(rawdata, connection.msgtype)
        pc = point_cloud_msg_to_np_array(msg)

        # Ensure file name has 6 digits (e.g. 000013.bin)
        target_num_digits = 6
        num_digits = len(str(frame_idx))
        diff = target_num_digits - num_digits
        new_name = (str('0') * diff) + str(frame_idx)
        output_folder = "autolabeller/result/velodyne_points/data"

        # Save point cloud data (without ground plane)
        bin_file_path = os.path.join(output_folder, f"{new_name}.bin")
        with open(bin_file_path, 'wb+', buffering=4096) as bin_file:
            bin_file.write(pc.tobytes())
        return pc

    def readPcBin(self, frame_idx):
        target_num_digits = 6
        num_digits = len(str(frame_idx))
        diff = target_num_digits - num_digits
        new_name = (str('0') * diff) + str(frame_idx)
        bin_file_path = os.path.join(self.pc_path, f"{new_name}.bin")
        with open(bin_file_path, 'rb') as bin_file:
            byte_data = bin_file.read()
        
        output_folder = "autolabeller/result/velodyne_points/data"
        new_bin_file_path = os.path.join(output_folder, f"{new_name}.bin")
        shutil.copy(bin_file_path, new_bin_file_path)

        shape = (-1, 4)
        pc = np.frombuffer(byte_data, dtype=np.float32).reshape(shape)

        return pc

    def __getitem__(self, index):
        try:
            if self.use_raw_pc:
                points = self.readPcBin(index)
            else:
                points = self.readRosBag(index)

            input_dict = {
                'points': points,
                'frame_id': index,
            }

            data_dict = self.prepare_data(data_dict=input_dict)
            return data_dict
        
        except:
            self.logger.info("All Rosbag messages have been consumed")
            raise IndexError("Index out of bounds.")

        
# --------------- Main ---------------


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/voxel_rcnn_cvat.yaml', help='Specify the model config')
    parser.add_argument('--rosbag_path', type=str, required=False,default=None, help='Specify path to rosbag')
    parser.add_argument('--ckpt', type=str, required=True, default=None, help='Specify the pretrained model')
    parser.add_argument("--use_raw_pc", action='store_true', default=False, help='Choose to use a rosbag or bin files')
    parser.add_argument("--raw_pc_path", default=None, help='Specify the path of the point cloud bin files')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    logger = common_utils.create_logger()
    if (os.path.basename(os.getcwd()) != "tools"):
        logger.error("Must call script from tools folder. Exiting")
        exit(-1)

    args, model_cfg = parse_config()
    rosbag_cfg = 'autolabeller/configs/rosbag_config.yaml'
    rosbag_cfg = yaml.load(open(rosbag_cfg), Loader=yaml.FullLoader)
    
    os.makedirs("autolabeller/result/annotations", exist_ok=True)
    os.makedirs("autolabeller/result/velodyne_points/data", exist_ok=True)
    
    logger.info('-----------------Labelling Bag-------------------------')
    dataset = MyDataset( 
        dataset_cfg=model_cfg.DATA_CONFIG,
        class_names=model_cfg.CLASS_NAMES,
        rosbag_path=args.rosbag_path,
        rosbag_cfg=rosbag_cfg,
        logger=logger,
        use_raw_pc=args.use_raw_pc,
        raw_pc_path=args.raw_pc_path
    )

    datumaro_detections = []

    model = build_network(model_cfg=model_cfg.MODEL, num_class=len(model_cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        # for idx, data_dict in enumerate(dataset):
        for data_dict in dataset:
            # We havent collected info from all topics yet. Skip till we have
            if not data_dict: continue
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            frame_idx = int(data_dict['frame_id'].item())
            logger.info(f'Visualized sample index: \t{frame_idx + 1}')

            for pd in pred_dicts:
                pred_boxes = pd['pred_boxes'].cpu().numpy()
                pred_scores = pd['pred_scores'].cpu().numpy()
                pred_labels = pd['pred_labels'].cpu().numpy()

                tracked_dict = process_tracked_box(pred_boxes, pred_scores, pred_labels)

                datumaro_detections.append(generate_annotation(frame_idx, tracked_dict))
        
    datumaro_annotations = generate_datumaro_annotations(datumaro_detections)

    with open("autolabeller/result/annotations/default.json", "w+") as f:
        json.dump(datumaro_annotations, f, indent=4)
    logger.info(f'Finished processing {frame_idx+1} frames.')

    logger.info('Labelling done.')


if __name__ == '__main__':
    main()
