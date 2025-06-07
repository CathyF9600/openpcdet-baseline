"""
Converts CVAT KITTI format for OpenPCDet training

Usage: python3 tools/process_cvat_data.py
Specify the dataset you are processing in custom_dataset.yaml
"""

import xml.etree.ElementTree as ET
import os
import numpy as np
import yaml
import subprocess
import argparse
import re
import shutil
from tqdm import tqdm


def parse_xml(xml_file, frame_idx_to_name):
    try:
        tree = ET.parse(xml_file)
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None

    root = tree.getroot()
    tracklets = root.find('tracklets')
    items = tracklets.findall('item')
    
    annotations_per_frame = {}
    
    for item in items:
        object_type = item.find('objectType').text
        object_type = object_type.replace(" ", "_").lower()
        poses = item.find('poses')
        
        for pose in poses.findall('item'):
            frame_id = frame_idx_to_name[int(item.find('first_frame').text)]
            
            if frame_id not in annotations_per_frame:
                annotations_per_frame[frame_id] = []
            
            x = float(pose.find('tx').text)
            y = float(pose.find('ty').text)
            z = float(pose.find('tz').text)
            dz = float(item.find('h').text)
            dy = float(item.find('w').text)
            dx = float(item.find('l').text)
            heading_angle = float(pose.find('rz').text) # TODO: Validate if this is correct?

            # NOTE: Refer to https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md#label-format
            # in order to understand the KITTI label format. We use the same format here. We have set a lot of things we do not need to 0.
            # annotation = f"{object_type} 0.00 0 0 0 0 0 0 {h:.2f} {w:.2f} {l:.2f} {tx:.2f} {ty:.2f} {tz:.2f} {ry:.2f} 0"
            annotation = f"{x:.2f} {y:.2f} {z:.2f} {dx:.2f} {dy:.2f} {dz:.2f} {heading_angle:.2f} {object_type}"
            
            annotations_per_frame[frame_id].append(annotation)
    
    return annotations_per_frame

def write_annotations_txt(folder_path, annotations_per_frame):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for frame_id, annotations in annotations_per_frame.items():
        file_path = os.path.join(folder_path, f'{frame_id:06d}.txt')
        
        with open(file_path, 'w', encoding='utf-8') as file:
            for annotation in annotations:
                file.write(annotation + '\n')

def parse_xml_to_txt(xml_file, folder_path, frame_idx_to_name):
    annotations_per_frame = parse_xml(xml_file, frame_idx_to_name)
    
    if annotations_per_frame is not None:
        write_annotations_txt(folder_path, annotations_per_frame)

def parse_numpy(directory):

    # make points folder (two levels up) and move all the npy files to that folder
    points_folder = os.path.join(directory, '..', '..')
    points_folder = os.path.join(points_folder, 'points')
    if not os.path.exists(points_folder):
        os.makedirs(points_folder)

    for filename in os.listdir(directory):
        if filename.endswith('.bin'):
            filepath = os.path.join(directory, filename)

            # Read the .bin file into a numpy array
            # Assuming each point (x, y, z, intensity) is stored as float32
            arr = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)

            np.save(filepath.replace('.bin', '.npy'), arr)

            # move the npy file to the points folder
            new_filepath = os.path.join(points_folder, filename.replace('.bin', '.npy'))
            os.rename(filepath.replace('.bin', '.npy'), new_filepath)


def create_imagesets_from_label_txt(directory, split, train_split=0.85):
    # frame list, if you leave switch outside property after the final labelled frame
    # would result in extra entires in frame_list but those don't have label txts

    # create ImageSets folder
    imagesets_folder = os.path.join(directory, 'ImageSets')
    if not os.path.exists(imagesets_folder):
        os.makedirs(imagesets_folder)

    # create train.txt and val.txt
    train_file = os.path.join(imagesets_folder, 'train.txt')
    val_file = os.path.join(imagesets_folder, 'val.txt')

    # TODO: for now, just put all files in both train and validation
    filenames = sorted([filename for filename in os.listdir(os.path.join(directory, 'labels')) if filename.endswith('.txt')])


    if not split:
        with open(train_file, 'w') as file:
            for filename in filenames:
                file.write(filename.split('.')[0] + '\n')
        with open(val_file, 'w') as file:
            for filename in filenames:
                file.write(filename.split('.')[0] + '\n')
        return

    # Shuffle the filenames
    np.random.shuffle(filenames)

    # Write sorted filenames to the train file
    with open(train_file, 'w') as file:
        for filename in filenames[:int(train_split * len(filenames))]:
            file.write(filename.split('.')[0] + '\n')

    # Write sorted filenames (without extension) to the val file
    with open(val_file, 'w') as file:
        for filename in filenames[int(train_split * len(filenames)):]:
            file.write(filename.split('.')[0] + '\n')

def remap_to_known_classes(known_classes, map_class_to_kitti, label):

    # read all labels and remap classes to known classes
    invalids = []
    for filename in os.listdir(label):
        if filename.endswith('.txt'):
            filepath = os.path.join(label, filename)

            with open(filepath, 'r') as file:
                lines = file.readlines()

            with open(filepath, 'w') as file:
                num_valid_annotations = 0
                for line in lines:
                    x, y, z, dx, dy, dz, heading_angle, object_type = line.split()
                    if object_type not in known_classes:
                        if object_type in map_class_to_kitti:
                            object_type = map_class_to_kitti[object_type]
                        else:
                            print(f"Unknown class, ignoring label: {object_type}")
                            continue
                    file.write(f"{x} {y} {z} {dx} {dy} {dz} {heading_angle} {object_type}\n")
                    num_valid_annotations += 1
                
                if num_valid_annotations == 0:
                    invalids.append(int(filename.split('.')[0]))
                    os.remove(filepath)  # we will inform the train val splits via the updated frame list

    return invalids

def merge_tracklet_items(source_file, tracklets_element, frame_offset):
    tree = ET.parse(source_file)
    root = tree.getroot()
    source_tracklets = root.find("tracklets")
    firstIteration = True
    frame_offset_final = 0
    #iterate through <item> elements and add em in
    for item in source_tracklets.findall("item"):
        #adjust the first frame
        first_frame_element = item.find('first_frame')
        first_frame = int(first_frame_element.text)
        if firstIteration: 
            frame_offset_final = first_frame
            firstIteration = False
        first_frame += frame_offset
        first_frame_element.text = str(first_frame)
        #print("inital, final, offset", (first_frame - frame_offset, first_frame, frame_offset))

        tracklets_element.append(item)

    #print("frame_offset_final value ", frame_offset_final)
    return (frame_offset_final+1)


def drop_empty_labels(invalids, frame_list_file_path):
    # remove all blank labels from frame_list.txt
    with open(frame_list_file_path, 'r') as f:
        lines = f.readlines()
    with open(frame_list_file_path, 'w') as f:
        idx = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                _, name = map(int, parts)
                if name not in invalids:
                    f.write(f"{idx} {name:06d}\n")
                    idx += 1

def parse_args():
    parser = argparse.ArgumentParser(description='Process CVAT data for OpenPCDet training')
    parser.add_argument('--data_yaml', type=str, default='tools/cfgs/dataset_configs/custom_dataset.yaml', help='Path to the data yaml file')
    parser.add_argument('--class_yaml', type=str, default='tools/cfgs/kitti_models/pointpillars_cvat.yaml', help='Path to the class yaml file')
    parser.add_argument('--split', action='store_true', help='Split the data into train and validation sets')
    return parser.parse_args()


if __name__ == "__main__":


    args = parse_args()
    print(args)
    
    # get the directory from yaml
    with open(args.data_yaml) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(args.class_yaml) as file:
        cvat = yaml.load(file, Loader=yaml.FullLoader)

    #pre-work
    merge_dirs = config['MERGE_PATHS']

    current_image_idx, current_velodyne_idx = 0, 0
    frame_cnt = 0
    item_count = 0
    save_dir = config['SAVE_PATH']

    #new directory which will store all images, velodyne points, frame list and tracklet labels
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
        
    # new_images_directory = os.path.join(save_dir, 'IMAGE_00/data')
    new_velodyne_directory = os.path.join(save_dir, 'velodyne_points/data')
    # os.makedirs(new_images_directory, exist_ok=True)
    os.makedirs(new_velodyne_directory, exist_ok=True)

    merged_file_path = os.path.join(save_dir, 'tracklet_labels.xml')
    merged_root = ET.Element("boost_serialization", version="9", signature="serialization::archive")
    tracklets = ET.SubElement(merged_root, "tracklets", version="0", tracking_level="0", class_id="0")
    ET.SubElement(tracklets, "count").text = "0"
    ET.SubElement(tracklets, "item_version").text = "1"
    
    interbag_name_offset = 0
    starting_frame_idx = 0
    frame_idx_to_name = dict()

    for directory in tqdm(merge_dirs):

        # #change name of images
        # images_directory_path = os.path.join(directory, 'IMAGE_00/data')
        # for imageFile in sorted(os.listdir(images_directory_path)):
        #     source_file_path = os.path.join(images_directory_path, imageFile)
        #     new_file_name = f"{current_image_idx:06d}{os.path.splitext(imageFile)[1]}"
        #     new_file_path = os.path.join(new_images_directory, new_file_name)
        #     shutil.copy(source_file_path, new_file_path)
        #     current_image_idx += 1

        #change name of velodyne_points
        velodyne_points_directory_path = os.path.join(directory, 'velodyne_points/data')
        for binFile in sorted(os.listdir(velodyne_points_directory_path)):
            if not binFile.endswith('.bin'):
                continue
            source_file_path = os.path.join(velodyne_points_directory_path, binFile)
            new_file_name = f"{current_velodyne_idx:06d}{os.path.splitext(binFile)[1]}"
            new_file_path = os.path.join(new_velodyne_directory, new_file_name)
            shutil.copy(source_file_path, new_file_path)
            current_velodyne_idx += 1

        # assert(current_image_idx == current_velodyne_idx)

        #made file with correct frame_list values
        frame_list_file_path = os.path.join(directory, 'frame_list.txt')
        new_frame_list_file_path = os.path.join(save_dir, 'frame_list.txt')
        with open(frame_list_file_path, 'r') as f_in, open(new_frame_list_file_path, 'a') as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) == 2:
                    in_idx, in_name = map(int, parts)
                    upd_name = in_name + interbag_name_offset
                    # print(f"{frame_cnt} {upd_name:06d}")
                    f_out.write(f"{frame_cnt} {upd_name:06d}\n")
                    frame_idx_to_name[frame_cnt] = upd_name
                    frame_cnt += 1
        
        tracklet_labels_path = os.path.join(directory, 'tracklet_labels.xml')
        if os.path.exists(tracklet_labels_path):
            _ = merge_tracklet_items(tracklet_labels_path, tracklets, frame_offset=starting_frame_idx)
            interbag_name_offset = current_velodyne_idx
            starting_frame_idx = frame_cnt
            item_count += len(ET.parse(tracklet_labels_path).getroot().find("tracklets").findall("item"))


    tracklets.find("count").text = str(item_count)

    #save
    merged_tree = ET.ElementTree(merged_root)
    merged_tree.write(merged_file_path, encoding="utf-8", xml_declaration=True)


    
    directory = config['DATA_PATH']
    xml_file_path = os.path.join(directory, 'tracklet_labels.xml')
    labels_file_path = os.path.join(directory, 'labels')
    pc_file_path = os.path.join(directory, 'velodyne_points/data')

    parse_xml_to_txt(xml_file_path, labels_file_path, frame_idx_to_name)
    parse_numpy(pc_file_path)
    invalids = remap_to_known_classes(cvat['CLASS_NAMES'], config['MAP_CLASS_TO_KITTI'], labels_file_path)   
    print(f"[Warning] Blank labels will be purged from frame_list.txt: {invalids}")
    drop_empty_labels(invalids, new_frame_list_file_path)
    create_imagesets_from_label_txt(directory, args.split)
    

    # build gt dictionary
    subprocess.run(['python3', '-m', 'pcdet.datasets.custom.custom_dataset', 
                    'create_custom_infos', args.data_yaml])
    
    
