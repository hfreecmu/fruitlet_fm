import os
import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import open3d
import json
import pickle

def read_json(path):
    with open(path, 'r') as f:
        json_to_read = json.loads(f.read())
    
    return json_to_read

def read_yaml(path):
    with open(path, 'r') as f:
        yaml_to_read = yaml.safe_load(f)

    return yaml_to_read

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def get_intrinsics(camera_info):
    P = camera_info['projection_matrix']['data']
    f_norm = P[0]
    baseline = P[3] / P[0]
    cx = P[2]
    cy = P[6]
    intrinsics = (baseline, f_norm, cx, cy)

    return intrinsics

def get_transform(transform_path):
    transform_dict = read_yaml(transform_path)
    quat = transform_dict['quaternion']
    trans = transform_dict['translation']

    q = [quat['qx'], quat['qy'], quat['qz'], quat['qw']]
    t = [trans['x'], trans['y'], trans['z']]

    R = Rotation.from_quat(q).as_matrix()
    t = np.array(t)

    return R, t

def bilateral_filter(disparity, intrinsics, args):
    baseline, f_norm, _, _ = intrinsics
    stub = -baseline / disparity
    z = stub * f_norm
    z_new = cv2.bilateralFilter(z, args.bilateral_d, args.bilateral_sc, args.bilateral_ss)

    stub_new = z_new / f_norm
    disparity_new = -baseline / stub_new

    return disparity_new

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

def compute_points(disparity, intrinsics):
    baseline, f_norm, cx, cy = intrinsics
    stub = -baseline / disparity #*0.965

    x_pts, y_pts = np.meshgrid(np.arange(disparity.shape[1]), np.arange(disparity.shape[0]))

    x = stub * (x_pts - cx)
    y = stub * (y_pts - cy)
    z = stub*f_norm

    points = np.stack((x, y, z), axis=2)

    return points

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

def warp_points(points, H):
    points_homo = np.ones((points.shape[0], 3))
    points_homo[:, 0:2] = points
    perspective_points_homo = (H @ points_homo.T).T
    perspective_points = perspective_points_homo[:, 0:2] / perspective_points_homo[:, 2:]

    return perspective_points
# def sample_homography_dep(
#         shape, perspective=True, scaling=True, rotation=True, translation=True,
#         n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
#         perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=pi/2,
#         allow_artifacts=False, translation_overflow=0.):
#     """Sample a random valid homography.
#     Computes the homography transformation between a random patch in the original image
#     and a warped projection with the same image size.
#     As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
#     transformed input point (original patch).
#     The original patch, which is initialized with a simple half-size centered crop, is
#     iteratively projected, scaled, rotated and translated.
#     Arguments:
#         shape: A tuple specifying the height and width of the original image.
#         perspective: A boolean that enables the perspective and affine transformations.
#         scaling: A boolean that enables the random scaling of the patch.
#         rotation: A boolean that enables the random rotation of the patch.
#         translation: A boolean that enables the random translation of the patch.
#         n_scales: The number of tentative scales that are sampled when scaling.
#         n_angles: The number of tentatives angles that are sampled when rotating.
#         scaling_amplitude: Controls the amount of scale.
#         perspective_amplitude_x: Controls the perspective effect in x direction.
#         perspective_amplitude_y: Controls the perspective effect in y direction.
#         patch_ratio: Controls the size of the patches used to create the homography.
#         max_angle: Maximum angle used in rotations.
#         allow_artifacts: A boolean that enables artifacts when applying the homography.
#         translation_overflow: Amount of border artifacts caused by translation.
#     Returns:
#         A tensor of shape `[1, 8]` corresponding to the flattened homography transform.
#     """

#     # Corners of the output image
#     margin = (1 - patch_ratio) / 2
#     pts1 = margin + torch.tensor([[0, 0], [0, patch_ratio],
#                                   [patch_ratio, patch_ratio], [patch_ratio, 0]],
#                                  dtype=torch.float32)
    
#     # Corners of the input patch
#     pts2 = pts1

#     # Random perspective and affine perturbations
#     if perspective:
#         if not allow_artifacts:
#             perspective_amplitude_x = min(perspective_amplitude_x, margin)
#             perspective_amplitude_y = min(perspective_amplitude_y, margin)
#         perspective_displacement = torch.tensor([1.])
#         h_displacement_left = torch.tensor([1.])
#         h_displacement_right = torch.tensor([1.])
#         torch.nn.init.trunc_normal_(perspective_displacement, 0., perspective_amplitude_y/2)
#         torch.nn.init.trunc_normal_(h_displacement_left, 0., perspective_amplitude_y/2)
#         torch.nn.init.trunc_normal_(h_displacement_right, 0., perspective_amplitude_y/2)
#         pts2 += torch.stack([torch.cat([h_displacement_left, perspective_displacement], 0),
#                              torch.cat([h_displacement_left, -perspective_displacement], 0),
#                              torch.cat([h_displacement_right, perspective_displacement], 0),
#                              torch.cat([h_displacement_right, -perspective_displacement],
#                                        0)])
#     # Random scaling
#     # sample several scales, check collision with borders, randomly pick a valid one
#     if scaling:
#         n_scales_tf_norm = torch.tensor([1.])
#         torch.nn.init.trunc_normal_(n_scales_tf_norm, 0., scaling_amplitude/2)
#         scales = torch.cat(
#             [torch.tensor([1.]), n_scales_tf_norm], 0)
#         center = torch.mean(pts2, dim=0, keepdim=True)
#         scaled = (pts2 - center).unsqueeze(0) * scales.unsqueeze(1).unsqueeze(1) + center
#         if allow_artifacts:
#             valid = torch.arange(1, n_scales + 1)  # all scales are valid except scale=1
#         else:
#             #valid = torch.nonzero(torch.all((scaled >= 0.) & (scaled <= 1.), dim=[1, 2]))[:, 0]
#             scaled_np = scaled.numpy()
#             valid = np.all((scaled_np >= 0.) & (scaled_np <= 1.), axis=(1, 2))
#             valid = torch.nonzero(torch.from_numpy(valid))[:, 0]

#         idx = valid[torch.randint(0, valid.shape[0], ())]
#         pts2 = scaled[idx]

#     # Random translation
#     if translation:
#         t_min, t_max = torch.min(pts2, dim=0)[0], torch.min(1 - pts2, dim=0)[0]
#         if allow_artifacts:
#             t_min += translation_overflow
#             t_max += translation_overflow
#         pts2 += torch.cat([torch.tensor([torch.rand(()).item() * (-t_min[0] + t_max[0]) - t_max[0]]),
#                            torch.tensor([torch.rand(()).item() * (-t_min[1] + t_max[1]) - t_max[1]])]).unsqueeze(0)

#     # Random rotation
#     # sample several rotations, check collision with borders, randomly pick a valid one
#     if rotation:
#         angles = torch.linspace(-max_angle, max_angle, n_angles)
#         angles = torch.cat([torch.tensor([0.]), angles], dim=0)  # in case no rotation is valid
#         center = torch.mean(pts2, dim=0, keepdim=True)
#         rot_mat = torch.reshape(torch.stack([torch.cos(angles), -torch.sin(angles), torch.sin(angles),
#                                               torch.cos(angles)], dim=1), (-1, 2, 2))
#         rotated = torch.matmul(
#             torch.tile((pts2 - center).unsqueeze(0), (n_angles+1, 1, 1)),
#             rot_mat) + center
#         if True or allow_artifacts:
#             valid = torch.arange(1, n_angles + 1)  # all angles are valid, except angle=0
#         else:
#             # valid = torch.nonzero(torch.all((rotated >= 0.) & (rotated <= 1.), dim=[1, 2]))[:, 0]
#             rotated_np = rotated.numpy()
#             valid = np.all((rotated_np >= 0.) & (rotated_np <= 1.), axis=(1, 2))
#             valid = torch.nonzero(torch.from_numpy(valid))[:, 0]

#         idx = valid[torch.randint(0, valid.shape[0], ())]
#         pts2 = rotated[idx]

#     # Rescale to actual size
#     shape = torch.tensor(shape[::-1], dtype=torch.float32)  # different convention [y, x]
#     pts1 *= shape.unsqueeze(0)
#     pts2 *= shape.unsqueeze(0)

#     def ax(p, q): return torch.tensor([p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]])

#     def ay(p, q): return torch.tensor([0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]])

#     a_mat = torch.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], dim=0)
#     p_mat = torch.transpose(torch.stack(
#         [torch.tensor([pts2[i][j] for i in range(4) for j in range(2)])], dim=0), 0, 1)
    
#     solution= torch.linalg.lstsq(p_mat, a_mat).solution
#     homography = torch.transpose(solution, 0, 1)
#     return homography
