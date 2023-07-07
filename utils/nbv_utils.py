import os
import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import open3d
import json
import pickle
import distinctipy

#read json file
def read_json(path):
    with open(path, 'r') as f:
        json_to_read = json.loads(f.read())
    
    return json_to_read

#read yaml file
def read_yaml(path):
    with open(path, 'r') as f:
        yaml_to_read = yaml.safe_load(f)

    return yaml_to_read

#read pkl file
def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

#write pkl file
def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

#get intrinsics struck from dict
def get_intrinsics(camera_info):
    P = camera_info['projection_matrix']['data']
    f_norm = P[0]
    baseline = P[3] / P[0]
    cx = P[2]
    cy = P[6]
    intrinsics = (baseline, f_norm, cx, cy)

    return intrinsics

#get transform from yml path
def get_transform(transform_path):
    transform_dict = read_yaml(transform_path)
    quat = transform_dict['quaternion']
    trans = transform_dict['translation']

    q = [quat['qx'], quat['qy'], quat['qz'], quat['qw']]
    t = [trans['x'], trans['y'], trans['z']]

    R = Rotation.from_quat(q).as_matrix()
    t = np.array(t)

    return R, t

#bilateral filter
def bilateral_filter(disparity, intrinsics, args):
    baseline, f_norm, _, _ = intrinsics
    stub = -baseline / disparity
    z = stub * f_norm
    z_new = cv2.bilateralFilter(z, args.bilateral_d, args.bilateral_sc, args.bilateral_ss)

    stub_new = z_new / f_norm
    disparity_new = -baseline / stub_new

    return disparity_new

#extract depth discontinuities
def extract_depth_discontuinities(disparity, intrinsics, args):
    baseline, f_norm, _, _ = intrinsics
    stub = -baseline / disparity
    z = stub * f_norm

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(z, element)
    erosion = cv2.erode(z, element)

    dilation -= z
    erosion = z - erosion

    max_image = np.max((dilation, erosion), axis=0)

    if args.disc_use_rat:
        ratio_image = max_image / z
        _, discontinuity_map = cv2.threshold(ratio_image, args.disc_rat_thresh, 1.0, cv2.THRESH_BINARY)
    else:
        _, discontinuity_map = cv2.threshold(max_image, args.disc_dist_thresh, 1.0, cv2.THRESH_BINARY)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    discontinuity_map = cv2.morphologyEx(discontinuity_map, cv2.MORPH_CLOSE, element)

    return discontinuity_map

#compute points using our method
def compute_points(disparity, intrinsics):
    baseline, f_norm, cx, cy = intrinsics
    stub = -baseline / disparity #*0.965

    x_pts, y_pts = np.meshgrid(np.arange(disparity.shape[1]), np.arange(disparity.shape[0]))

    x = stub * (x_pts - cx)
    y = stub * (y_pts - cy)
    z = stub*f_norm

    points = np.stack((x, y, z), axis=2)

    return points

#compute points using opencv - same as above
def compute_points_opencv(disparity, intrinsics):
    baseline, f_norm, cx, cy = intrinsics
    
    cm1 = np.array([[f_norm, 0, cx],
                   [0, f_norm, cy],
                   [0, 0, 1]])
    
    cm2 = np.array([[f_norm, 0, cx],
                   [0, f_norm, cy],
                   [0, 0, 1]])
    
    distortion1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    distortion2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    T = np.array([baseline, 0, 0])
    R = np.eye(3)


    res = cv2.stereoRectify(cm1, distortion1, cm2, distortion2, (1440, 1080), R, T)
    Q = res[4]

    points = cv2.reprojectImageTo3D(disparity, Q)

    return points

#save point cloud file
def create_point_cloud(cloud_path, points, colors, normals=None, estimate_normals=False):
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(points)
    cloud.colors = open3d.utility.Vector3dVector(colors)

    if normals is not None:
        cloud.normals = open3d.utility.Vector3dVector(normals)
    elif estimate_normals:
        cloud.estimate_normals(
            search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

    open3d.io.write_point_cloud(
        cloud_path,
        cloud
    ) 

#extract point cloud using filters
def extract_point_cloud(left_path, disparity_path, 
                        camera_info_path, transform_path, args):
    camera_info = read_yaml(camera_info_path)
    intrinsics = get_intrinsics(camera_info)

    disparity = np.load(disparity_path)
    assert np.min(disparity) > 0

    im = cv2.imread(left_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if args.bilateral_filter:
        disparity = bilateral_filter(disparity, intrinsics, args)
    
    discontinuity_map = extract_depth_discontuinities(disparity, intrinsics, args)
    points = compute_points(disparity, intrinsics)
    colors = im.astype(float) / 255

    nan_inds = np.where(discontinuity_map > 0) 
    points[nan_inds] = np.nan
    colors[nan_inds] = np.nan

    nan_inds = np.where(points[:, :, 2] > args.max_dist)
    points[nan_inds] = np.nan
    colors[nan_inds] = np.nan

    R, t = get_transform(transform_path)
    world_points = ((R @ points.reshape((-1, 3)).T).T + t).reshape(points.shape)

    return points, world_points, colors, R, t

#get paths following directory structure
def get_paths(data_dir, indices, single=False):
    left_dir = os.path.join(data_dir, 'rect_images', 'left')
    disparities_dir = os.path.join(data_dir, 'disparities')
    #use right camera info for baseline
    camera_info_dir = os.path.join(data_dir, 'camera_info', 'right')
    transform_dir = os.path.join(data_dir, 'ee_states')
    segmentation_dir = os.path.join(data_dir, 'segmentations')

    left_paths = []
    disparity_paths = []
    camera_info_paths = []
    transform_paths = []
    segmentation_paths = []
    path_inds = []

    if indices is None:
        inds_path = os.path.join(data_dir, 'indices.json')
        indices = read_json(inds_path)

    for filename in os.listdir(left_dir):
        if not filename.endswith('.png'):
            continue

        index = int(filename.split('.png')[0])

        if not index in indices:
            continue

        left_path = os.path.join(left_dir, filename)
        disparity_path = os.path.join(disparities_dir, filename.replace('.png', '.npy'))
        camera_info_path = os.path.join(camera_info_dir, filename.replace('.png', '.yml'))
        transform_path = os.path.join(transform_dir, filename.replace('.png', '.yml'))
        segmentation_path = os.path.join(segmentation_dir, filename.replace('.png', '.pkl'))

        if not os.path.exists(camera_info_path):
            raise RuntimeError('No camera_info_path for: ' + camera_info_path)
        
        if not os.path.exists(transform_path):
            raise RuntimeError('No transform_path for: ' + transform_path)
        
        left_paths.append(left_path)
        disparity_paths.append(disparity_path)
        camera_info_paths.append(camera_info_path)
        transform_paths.append(transform_path)
        segmentation_paths.append(segmentation_path)
        path_inds.append(index)

    if not single:
        return path_inds, left_paths, disparity_paths, camera_info_paths, transform_paths, segmentation_paths
    else:
        return 0, left_paths[0], disparity_paths[0], camera_info_paths[0], transform_paths[0], segmentation_paths[0]

#warp 2D points using homography
def warp_points(points, H):
    points_homo = np.ones((points.shape[0], 3))
    points_homo[:, 0:2] = points
    perspective_points_homo = (H @ points_homo.T).T
    perspective_points = perspective_points_homo[:, 0:2] / perspective_points_homo[:, 2:]

    return perspective_points

def drop_points(valid_inds_bool, drop_thresh):
    should_drop = (np.random.random(size=valid_inds_bool.shape[0]) < drop_thresh)
    valid_inds_bool[should_drop] = False

def select_valid_points(valid_inds_bool, num_points):
    valid_inds_int = np.arange(valid_inds_bool.shape[0])[valid_inds_bool]

    if valid_inds_int.shape[0] <= num_points:
        return valid_inds_bool
    
    random_inds_subset = np.random.choice(valid_inds_int.shape[0], size=(num_points,), replace=False)
    
    valid_inds_bool_out = np.zeros(valid_inds_bool.shape[0], dtype=bool)
    valid_inds_bool_out[valid_inds_int[random_inds_subset]] = True

    return valid_inds_bool_out

def select_points(valid_inds_bool, seg_inds, bgr_vals):
    return seg_inds[valid_inds_bool], bgr_vals[valid_inds_bool]

#visualize feature matches
def vis_lines(torch_im_0, torch_im_1, torch_points_0, torch_points_1, output_path, padding=20):
    cv_im_0 = torch_im_0.permute(1, 2, 0).numpy().copy()
    cv_im_1 = torch_im_1.permute(1, 2, 0).numpy().copy()

    width = cv_im_0.shape[1]
    height = cv_im_0.shape[0]

    points_0 = torch_points_0.numpy().copy()
    points_1 = torch_points_1.numpy().copy()

    comb_im = np.zeros((height, 2*width+ padding, 3))
    comb_im[:, 0:width] = cv_im_0
    comb_im[:, width+padding:] = cv_im_1

    num_colors = 20
    colors = distinctipy.get_colors(num_colors)
    for i in range(points_0.shape[0]):
        color = colors[i % num_colors]
        color = ([int(255*color[0]), int(255*color[1]), int(255*color[2])])

        x0, y0 = points_0[i]
        x1, y1 = points_1[i]
        x1 += width + padding

        cv2.line(comb_im, (x0, y0), (x1, y1), color, thickness=1)

    cv2.imwrite(output_path, comb_im)

def vis_segs(torch_im_0, torch_im_1, torch_points_0, torch_points_1, output_path, padding=20):
    cv_im_0 = torch_im_0.permute(1, 2, 0).numpy().copy()
    cv_im_1 = torch_im_1.permute(1, 2, 0).numpy().copy()

    width = cv_im_0.shape[1]
    height = cv_im_0.shape[0]

    points_0 = torch_points_0.numpy().copy()
    points_1 = torch_points_1.numpy().copy()

    comb_im = np.zeros((height, 2*width+ padding, 3))
    comb_im[:, 0:width] = cv_im_0
    comb_im[:, width+padding:] = cv_im_1

    for i in range(points_0.shape[0]):
        x, y = points_0[i]

        cv2.circle(comb_im, (x, y), 1, (255, 0, 0), thickness=-1)

    for i in range(points_1.shape[0]):
        x, y = points_1[i]
        x += width + padding

        cv2.circle(comb_im, (x, y), 1, (255, 0, 0), thickness=-1)

    cv2.imwrite(output_path, comb_im)
