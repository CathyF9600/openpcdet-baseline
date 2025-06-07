import numpy as np
import open3d as o3d
import os

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
        line_width: box line width
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

# Load the data
FRAME = 0


# Create Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window()

vis.get_render_option().point_size = 1.0  # Adjust this value as needed (default is 5.0)
vis.get_render_option().line_width = 4.0
vis.get_render_option().background_color = np.asarray([0, 0, 0])

while True:
    try:
        data_dict_file = f'volume/results/data_dict_{FRAME:06d}.pkl'
        pred_dict_file = f'volume/results/pred_dicts_{FRAME:06d}.pkl'

        with open(data_dict_file, 'rb') as f:
            data_dict = np.load(f, allow_pickle=True)
        with open(pred_dict_file, 'rb') as f:
            pred_dict = np.load(f, allow_pickle=True)

        print("Num points: ", len(data_dict['points']))

        # vis point cloud
        points = np.array(data_dict['points'])
        points = points[:, 1:]

        # get boxes, scores, and labels
        boxes = np.array(pred_dict)
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
            7: (152/255, 181/255, 148/255),  # fire_hydrant
            8: (200/255, 81/255, 148/255),  # railroad bar down
        }

        label_to_name = { 
            0: 'unk',
            1: 'car',
            2: 'pedestrian',
            3: 'barrel',
            4: 'deer',
            5: 'sign',
            6: 'barricade',
            7: 'fire_hydrant',
            8: 'railroad_bar_down'
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
                Rz = np.array([
                    [np.cos(zrot), -np.sin(zrot), 0],
                    [np.sin(zrot),  np.cos(zrot), 0],
                    [0, 0, 1]
                ])
                
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

        vis.clear_geometries()

        # Add point cloud
        pcd = draw_lidar(points)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)

        # Add bounding boxes
        line_sets = draw_boxes3d(new_pred_boxes, color_list=colour_list, scores_list=scores_list, label_name_list=label_name_list)
        for line_set in line_sets:
            vis.add_geometry(line_set)
            vis.update_geometry(line_set)

        # Set Camera View from JSON
        camera_params = {
            "boundingbox_max": [50.061504364013672, 39.664722442626953, 9.3240251541137695],
            "boundingbox_min": [1.6045398712158203, -35.483657836914062, -6.4799299240112305],
            "field_of_view": 60.0,
            "front": [ -0.95838680584881764, 0.017337049394112139, 0.28494588449950803 ],
            "lookat": [ 25.853971991667837, 1.4352717527847894, 1.5323786256376781 ],
            "up": [ 0.28453597452260831, -0.022786481461139041, 0.95839451973866019 ],
            "zoom": 0.31999999999999962
        }

        # Apply camera settings
        view_control = vis.get_view_control()
        view_control.set_front(camera_params["front"])
        view_control.set_lookat(camera_params["lookat"])
        view_control.set_up(camera_params["up"])
        view_control.set_zoom(camera_params["zoom"])

        # Ensure rendering updates before capturing
        vis.poll_events()
        vis.update_renderer()

        # Save visualization
        output_dir = 'output/saved_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f'{FRAME}.png')
        vis.capture_screen_image(output_file, do_render=True)

        # Run visualization
        vis.run()
        

        FRAME += 1

    except FileNotFoundError:
        print('file not found')
        break

vis.destroy_window()