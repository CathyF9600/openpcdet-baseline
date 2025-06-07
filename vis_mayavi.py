import numpy as np
import mayavi.mlab
import os

# Suppress visualization display pop-ups
mayavi.mlab.options.offscreen = True

def draw_lidar(pc, color=None, fig=None, bgcolor=(0,0,0), pts_scale=1, pts_mode='point', pts_color=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    if fig is None: fig = mayavi.mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,2]
    mayavi.mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=pts_color, mode=pts_mode, colormap ='gnuplot', scale_factor=pts_scale, figure=fig)
    # colormaps: 'bone', 'copper', 'gnuplot', 'spectral'
    # color=(0, 1, 0) : Used a fixed (r,g,b) instead
    
    #draw origin
    mayavi.mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.1)
    
    #draw axis
    axes=np.array([
        [1.,0.,0.,0.],
        [0.,1.,0.,0.],
        [0.,0.,1.,0.],
    ],dtype=np.float64)
    mayavi.mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mayavi.mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mayavi.mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)

    mayavi.mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def draw_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=1, text_scale=(0.2,0.2,0.2), color_list=None, scores_list=None, label_name_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    ''' 
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n] 
            mayavi.mlab.text3d(b[4,0], b[4,1], b[4,2] - 0.5, '%.3f'%scores_list[n], scale=text_scale, color=color, figure=fig)
        if scores_list is not None: 
            mayavi.mlab.text3d(b[4,0], b[4,1], b[4,2], label_name_list[n], scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mayavi.mlab_helper_functions.html
            i,j=k,(k+1)%4
            mayavi.mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mayavi.mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mayavi.mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    #mayavi.mlab.show(1)
    mayavi.mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

########
########

# Load the data
FRAME = 0

while True:

    try:
        data_dict_file = f'results/data_dict_{FRAME:06d}.pkl'
        pred_dict_file = f'results/pred_dicts_{FRAME:06d}.pkl'

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

        fig = mayavi.mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))

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
            
            # print('box: ', box)
            # print('score: ', score)
            # print('label: ', label)

            colour_list.append(label_to_colour[label])
            scores_list.append(score)
            label_name_list.append(label_to_name[label])
            
            if score > 0.3:
                x, y, z, z_size, x_size, y_size, zrot = box

                zrot = np.deg2rad(zrot)
                # convert to box format: [n,8,3] for XYZs of the box corners
                # box format: [x, y, z, x_size, y_size, z_size, zrot]
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

                # Rotate each corner around the Z-axis
                # rotated_corners = []
                # for corner in corners:
                #     # Shift the corner so the rotation is around the center of the box
                #     shifted_corner = corner - np.array([x, y, z])
                #     # Apply rotation
                #     rotated_corner = Rz @ shifted_corner
                #     # Shift back
                #     rotated_corner += np.array([x, y, z])
                #     rotated_corners.append(rotated_corner)

                # new_pred_boxes.append(np.array(rotated_corners))
                new_pred_boxes.append(np.array(corners))

        draw_lidar(points, fig=fig)
        draw_boxes3d(new_pred_boxes, fig=fig, color_list=colour_list, scores_list=scores_list, label_name_list=label_name_list)

        output_dir = 'output/saved_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f'{FRAME}.png')
        mayavi.mlab.savefig(output_file)

        # mayavi.mlab.show()
        mayavi.mlab.close()

        FRAME += 1

    except FileNotFoundError:
        break