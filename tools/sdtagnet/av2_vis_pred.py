import argparse
import mmcv
import os
import shutil
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
import sys
sys.path.append('')
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle
from shapely.geometry import LineString
import cv2
import copy
from adjustText import adjust_text
from time import time


caption_by_cam={
    'ring_front_center':'CAM_FRONT_CENTER',
    'ring_front_right':'CAM_FRONT_RIGHT',
    'ring_front_left': 'CAM_FRONT_LEFT',
    'ring_rear_right': 'CAM_REAR_RIGHT',
    'ring_rear_left': 'CAM_REAT_LEFT',
    'ring_side_right': 'CAM_SIDE_RIGHT',
    'ring_side_left': 'CAM_SIDE_LEFT',
}
COLOR_MAPS_BGR = {
    # bgr colors
    'divider': (54,137,255),
    #'divider_solid': (5,53,73),
    #'divider_dashed': (255,255,0),
    'divider_dashed': (255,178,4),
    'divider_solid': (108,121,255),
    'divider_mixed': (50,0,50),
    'divider_virtual': (17,17,17),
    'ped_crossing': (255, 0, 0),
    'boundary': (0, 0, 255),
    'centerline': (0,255,0),
    'drivable_area': (171, 255, 255),
    'masked': (190, 190, 190)
}

# label_names = {
# 0: 'divider_dashed',
# 1: 'divider_solid',
# 2: 'divider_mixed',
# 3: 'divider_virtual',
# 4: 'ped_crossing',
# 5: 'boundary',
# 6: 'centerline',
# }

# LABEL_NAMES = {
# 0: 'divider_dashed',
# 1: 'divider_solid',
# 2: 'divider_virtual',
# 3: 'ped_crossing',
# 4: 'boundary',
# 5: 'centerline',
# }

data_path_prefix = '/home/users/yunchi.zhang/project/MapTR' # project root

def remove_nan_values(uv):
    is_u_valid = np.logical_not(np.isnan(uv[:, 0]))
    is_v_valid = np.logical_not(np.isnan(uv[:, 1]))
    is_uv_valid = np.logical_and(is_u_valid, is_v_valid)

    uv_valid = uv[is_uv_valid]
    return uv_valid

def interp_fixed_dist(line, sample_dist):
        ''' Interpolate a line at fixed interval.
        
        Args:
            line (LineString): line
            sample_dist (float): sample interval
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''

        distances = list(np.arange(sample_dist, line.length, sample_dist))
        # make sure to sample at least two points when sample_dist > line.length
        distances = [0,] + distances + [line.length,] 
        
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points

def draw_visible_polyline_cv2(line, valid_pts_bool, image, color, thickness_px,map_class):
    """Draw a polyline onto an image using given line segments.
    Args:
        line: Array of shape (K, 2) representing the coordinates of line.
        valid_pts_bool: Array of shape (K,) representing which polyline coordinates are valid for rendering.
            For example, if the coordinate is occluded, a user might specify that it is invalid.
            Line segments touching an invalid vertex will not be rendered.
        image: Array of shape (H, W, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        thickness_px: thickness (in pixels) to use when rendering the polyline.
    """
    line = np.round(line).astype(int)  # type: ignore
#     if map_class == 'centerline':
#         instance = LineString(line).simplify(0.2, preserve_topology=True)
#         line = np.array(list(instance.coords))
#         line = np.round(line).astype(int)
    for i in range(len(line) - 1):

        if (not valid_pts_bool[i]) or (not valid_pts_bool[i + 1]):
            continue

        x1 = line[i][0]
        y1 = line[i][1]
        x2 = line[i + 1][0]
        y2 = line[i + 1][1]

        # Use anti-aliasing (AA) for curves
        if map_class == 'divider_dashed' or map_class == 'divider_solid':
            image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=int(1.8*thickness_px), lineType=cv2.LINE_AA)
        elif map_class != 'centerline':
            image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness_px, lineType=cv2.LINE_AA)
        else:
            image = cv2.arrowedLine(image,(x1, y1),(x2,y2),color,int(0.7*thickness_px),8,0,0.7)


def points_ego2img(pts_ego, lidar2img):
    pts_ego_4d = np.concatenate([pts_ego, np.ones([len(pts_ego), 1])], axis=-1)
    pts_img_4d = lidar2img @ pts_ego_4d.T
    
    
    uv = pts_img_4d.T
    uv = remove_nan_values(uv)
    depth = uv[:, 2]
    uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)

    return uv, depth
def draw_polyline_ego_on_img(polyline_ego, img_bgr, lidar2img, map_class, thickness):
    # if 2-dimension, assume z=0
    if polyline_ego.shape[1] == 2:
        zeros = np.zeros((polyline_ego.shape[0], 1))
        polyline_ego = np.concatenate([polyline_ego, zeros], axis=1)

    polyline_ego = interp_fixed_dist(line=LineString(polyline_ego), sample_dist=0.2)
    
    uv, depth = points_ego2img(polyline_ego, lidar2img)

    # print("IMG SHAPE: ")
    # print(img_bgr.shape)
    # print("POLYLINE EGO:")
    # print(polyline_ego)
    # print("POLYLINE IMG: ")
    # print(uv)

    h, w, c = img_bgr.shape

    is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < w - 1)
    is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < h - 1)
    is_valid_z = depth > 0
    is_valid_points = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])

    if is_valid_points.sum() == 0:
        return
    
    tmp_list = []
    for i, valid in enumerate(is_valid_points):
        
        if valid:
            tmp_list.append(uv[i])
        else:
            if len(tmp_list) >= 2:
                tmp_vector = np.stack(tmp_list)
                tmp_vector = np.round(tmp_vector).astype(np.int32)
                draw_visible_polyline_cv2(
                    copy.deepcopy(tmp_vector),
                    valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
                    image=img_bgr,
                    color=COLOR_MAPS_BGR[map_class],
                    thickness_px=thickness,
                    map_class=map_class
                )
            tmp_list = []
    if len(tmp_list) >= 2:
        tmp_vector = np.stack(tmp_list)
        tmp_vector = np.round(tmp_vector).astype(np.int32)
        draw_visible_polyline_cv2(
            copy.deepcopy(tmp_vector),
            valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
            image=img_bgr,
            color=COLOR_MAPS_BGR[map_class],
            thickness_px=thickness,
            map_class=map_class,
        )

def render_anno_on_pv(cam_img, anno, lidar2img):
    for key, value in anno.items():
        for pts in value:
            draw_polyline_ego_on_img(pts, cam_img, lidar2img, 
                       key, thickness=10)

def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords

def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--score-thresh', default=0.3, type=float, help='samples to visualize')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--show-cam', action='store_true', help='show camera pic')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['fixed_num_pts',],
        help='vis format, default should be "points",'
        'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')
    parser.add_argument('--show-masked', default=True, type=bool, help='whether to draw masked gt elements in the visualization')
    parser.add_argument('--clip-text', default=False, type=bool, help='whether to clip osm tag text')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.show_dir is None:
        args.show_dir = osp.join('./work_dirs', 
                                osp.splitext(osp.basename(args.config))[0],
                                'vis_pred')
    # create vis_label dir
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    logger = get_root_logger()
    logger.info(f'DONE create vis_pred dir: {args.show_dir}')


    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True #TODO, this is a hack
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info('Done build test data set')

    # build the model and load checkpoint
    # import pdb;pdb.set_trace()
    cfg.model.train_cfg = None
    # cfg.model.pts_bbox_head.bbox_coder.max_num=15 # TODO this is a hack
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    logger.info('loading check point')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    logger.info('DONE load check point')
    model = MMDataParallel(model, device_ids=[0])

    # import pdb;pdb.set_trace()

    model.eval()

    img_norm_cfg = cfg.img_norm_cfg

    # get denormalized param
    mean = np.array(img_norm_cfg['mean'],dtype=np.float32)
    std = np.array(img_norm_cfg['std'],dtype=np.float32)
    to_bgr = img_norm_cfg['to_rgb']

    # get pc_range
    pc_range = cfg.point_cloud_range

    # get car icon
    car_img = Image.open('./figs/car.png')

    # get color map: divider_dashed->orange, divider_solid->darkgoldenrod, divider_mixed->purple, divider_virtual->dimgrey, ped_crossing->blue, boundary->red, centerline->green
    # colors_plt = ['orange', 'darkgoldenrod', 'purple', 'dimgrey', 'blue', 'red','green']
    colors_plt = ['orange', 'darkgoldenrod', 'dimgrey', 'blue', 'red','green']
    #colors_plt_dict = {'divider_dashed': 'orange', 'divider_solid': 'darkgoldenrod', 'divider_mixed': 'purple', 'divider_virtual': 'dimgrey', 'ped_crossing': 'blue', 'boundary': 'red', 'centerline': 'green', 'masked': 'lightgray'}
    colors_plt_dict = {'divider': 'goldenrod', 'divider_dashed': 'xkcd:aquamarine', 'divider_solid': 'xkcd:salmon', 'divider_mixed': 'purple', 'divider_virtual': 'dimgrey', 'ped_crossing': 'blue', 'boundary': 'red', 'centerline': 'green', 'masked': 'lightgray'}

    label_names = {i: map_class for i, map_class in enumerate(cfg.map_classes)}

    logger.info('BEGIN vis test dataset samples gt label & pred')

    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    have_mask = False
    # prog_bar = mmcv.ProgressBar(len(CANDIDATE))
    prog_bar = mmcv.ProgressBar(len(dataset))
    # import pdb;pdb.set_trace()
    final_dict = {}

    total_runtime = 0
    no_runs = 0

    for i, data in enumerate(data_loader):

        # if i < 1310:
        #     prog_bar.update()
        #     continue

        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            # prog_bar.update()  
            continue
       
        
        img = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]
        gt_masked = data['gt_masked'].data[0]

        # import pdb;pdb.set_trace()

        osm_map_nodes_pts = data['osm_map_nodes_pts'].data[0]
        osm_map_ways_pts = data['osm_map_ways_pts'].data[0]
        
        osm_map_nodes_tags_input_ids = data['osm_map_nodes_tags_input_ids'].data[0]
        osm_map_ways_tags_input_ids = data['osm_map_ways_tags_input_ids'].data[0]
        osm_map_relations_tags_input_ids = data['osm_map_relations_tags_input_ids'].data[0]
        
        
        osm_map_relations_node_member_indices = data['osm_map_relations_node_member_indices'].data[0]
        osm_map_relations_way_member_indices = data['osm_map_relations_way_member_indices'].data[0]
        osm_map_relations_relation_member_indices = data['osm_map_relations_relation_member_indices'].data[0]
        #osm_map_relations_node_member_tags = data['osm_map_relations_node_member_tags'].data[0]
        #osm_map_relations_way_member_tags = data['osm_map_relations_way_member_tags'].data[0]
        #osm_map_relations_relation_member_tags = data['osm_map_relations_relation_member_tags'].data[0]

        pts_filename = img_metas[0]['pts_filename']
        pts_filename = osp.basename(pts_filename)
        pts_filename = pts_filename.split('.')[0]
        # import pdb;pdb.set_trace()
        # if pts_filename not in CANDIDATE:
        #     continue
        sample_dict = {}
        with torch.no_grad():

            # start_time = time()
            # runtime = timeit.timeit(lambda: model(return_loss=False, rescale=True, **data))
            result = model(return_loss=False, rescale=True, **data)
            # end_time = time()
            # runtime = end_time - start_time
 
            # if i < 5: continue

            # no_runs += 1
            # total_runtime += runtime
            # avg_runtime = total_runtime / no_runs
            # print("Sample No.: " + str(no_runs))
            # print("current FPS: " + str(1.0/runtime))
            # print("avg. FPS: " + str(1.0/avg_runtime))
            # 
            # if no_runs == 10:
            #     print("avg. FPS after 10 samples: " + str(1.0/avg_runtime))
 
            # continue

            # result = model(return_loss=False, rescale=True, **data)

        sample_dir = osp.join(args.show_dir, pts_filename)
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))

        filename_list = img_metas[0]['filename']
        img_path_dict = {}
        # save cam img for sample
        # import ipdb;ipdb.set_trace() 
        for filepath, lidar2img, img_aug in zip(filename_list,img_metas[0]['lidar2img'],img_metas[0]['img_aug_matrix']):
            inv_aug = np.linalg.inv(img_aug)
            lidar2orimg = np.dot(inv_aug, lidar2img)
            cam_name = os.path.dirname(filepath).split('/')[-1]
            img_path_dict[cam_name] = dict(
                filepath=filepath,
                lidar2img = lidar2orimg)
        sample_dict['imgs_path'] = img_path_dict
        gt_dict = {'divider':[], 'divider_dashed':[],'divider_solid':[],'divider_mixed':[],'divider_virtual':[],'ped_crossing':[],'boundary':[],'centerline':[], 'masked':[]}
        # import ipdb;ipdb.set_trace() 
        gt_lines_instance = gt_bboxes_3d[0].instance_list
        # import pdb;pdb.set_trace()
        for gt_line_instance, gt_label_3d, masked in zip(gt_lines_instance, gt_labels_3d[0], gt_masked[0]):
            # if masked:
            #     if args.show_masked:
            #         gt_dict['masked'].append(np.array(list(gt_line_instance.coords)))
            #     continue
            if label_names[gt_label_3d.item()] == 'divider_dashed':
                gt_dict['divider_dashed'].append(np.array(list(gt_line_instance.coords)))
            elif label_names[gt_label_3d.item()] == 'divider_solid':
                gt_dict['divider_solid'].append(np.array(list(gt_line_instance.coords)))
            elif label_names[gt_label_3d.item()] == 'divider_mixed':
                gt_dict['divider_mixed'].append(np.array(list(gt_line_instance.coords)))
            elif label_names[gt_label_3d.item()] == 'divider_virtual':
                print("found virtual divider!")
                gt_dict['divider_virtual'].append(np.array(list(gt_line_instance.coords)))
            elif label_names[gt_label_3d.item()] == 'divider':
                gt_dict['divider'].append(np.array(list(gt_line_instance.coords)))
            elif label_names[gt_label_3d.item()] == 'ped_crossing':
                gt_dict['ped_crossing'].append(np.array(list(gt_line_instance.coords)))
            elif label_names[gt_label_3d.item()] == 'boundary':
                gt_dict['boundary'].append(np.array(list(gt_line_instance.coords)))
            elif label_names[gt_label_3d.item()] == 'centerline':
                gt_dict['centerline'].append(np.array(list(gt_line_instance.coords)))
            else:
                raise NotImplementedError
        sample_dict['gt_map'] = gt_dict

        result_dict = result[0]['pts_bbox']
        sample_dict['pred_map'] = result_dict

        # visualize gt
        plt.figure(figsize=(4, 2))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')
        gt_centerlines = []

        for key, pts_list in gt_dict.items():
            for pts in pts_list:
                if key == 'centerline':
                    instance = LineString(pts).simplify(0.2, preserve_topology=True) 
                    pts = np.array(list(instance.coords))
                    gt_centerlines.append(pts)
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy',scale=1,   color=colors_plt_dict[key],headwidth=5,headlength=6,width=0.006,alpha=0.8,zorder=-1)
                elif key == 'boundary':
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.plot(x, y, color=colors_plt_dict[key],linewidth=1,alpha=0.8,zorder=-10)
                elif key == 'divider_dashed' or key == 'divider_solid':
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.plot(x, y, color=colors_plt_dict[key],linewidth=1.25,alpha=0.8,zorder=-1)
                else:
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.plot(x, y, color=colors_plt_dict[key],linewidth=1,alpha=0.8,zorder=-1)
                                        
        plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
        gt_map_path = osp.join(sample_dir, 'GT_MAP.png')
        plt.savefig(gt_map_path, bbox_inches='tight', format='png',dpi=1200)
        plt.close()

        plt.figure(figsize=(4, 2))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')

        # import pdb;pdb.set_trace()

        osm_nodes_tags = dataset.nlp_tokenizer.batch_decode([t.tolist() for t in osm_map_nodes_tags_input_ids[0]], skip_special_tokens=True)
        osm_ways_tags = dataset.nlp_tokenizer.batch_decode([t.tolist() for t in osm_map_ways_tags_input_ids[0]], skip_special_tokens=True)
        osm_rels_tags = dataset.nlp_tokenizer.batch_decode([t.tolist() for t in osm_map_relations_tags_input_ids[0]], skip_special_tokens=True)

        drawn_texts = []
        
        for i, el in enumerate(osm_nodes_tags):
            if el == '':
                continue

            if args.clip_text and len(el) > 60:
                el = el[:60] + '...'

            text = plt.text(osm_map_nodes_pts[0][i][0], osm_map_nodes_pts[0][i][1], s=el, wrap=True, color='black', 
                     horizontalalignment='center', verticalalignment='center', fontsize=3,  
                     bbox=dict(boxstyle="square", ec=(0.3, 0.3, 0.3, 0.3), fc=(0.3, 0.3, 0.3, 0.3)))
            text._get_wrap_line_width = lambda : 1000
            drawn_texts.append(text)

        for i, el in enumerate(osm_ways_tags):
            if el == '':
                continue

            if args.clip_text and len(el) > 60:
                el = el[:60] + '...'

            text = plt.text(osm_map_ways_pts[0][i][9][0], osm_map_ways_pts[0][i][9][1], el, wrap=True, 
                     color='black', horizontalalignment='center', verticalalignment='center', fontsize=3,
                     bbox=dict(boxstyle="square", ec=(0.3, 0.3, 0.3, 0.3), fc=(0.3, 0.3, 0.3, 0.3)))
            text._get_wrap_line_width = lambda : 1000
            drawn_texts.append(text)

        
        # print(osm_rels_tags)
        # print(osm_map_relations_node_member_indices)
        # print(osm_map_relations_way_member_indices)
        # print(osm_map_relations_relation_member_indices)

        for i, el in enumerate(osm_rels_tags):
            if el == '':
                continue

            if args.clip_text and len(el) > 60:
                el = el[:60] + '...'

            member_pts = []
            if osm_map_relations_node_member_indices[0][i].numel():
                for idx in osm_map_relations_node_member_indices[0][i]:
                    member_pts.append(osm_map_nodes_pts[0][idx].squeeze().numpy())
            if osm_map_relations_way_member_indices[0][i].numel():
                for idx in osm_map_relations_way_member_indices[0][i]:
                    member_pts.append(np.average(osm_map_ways_pts[0][idx].squeeze().numpy(), axis=0))
            if osm_map_relations_relation_member_indices[0][i].numel():
                for idx in osm_map_relations_relation_member_indices[0][i]:
                    member_pts.append(np.array([0, 0]))

            # import pdb;pdb.set_trace()
            
            member_pts = np.vstack(member_pts)
            center = np.average(member_pts, axis=0)

            for pt in member_pts:
                direction = center - pt
                plt.arrow(pt[0], pt[1], direction[0], direction[1], color='black',linewidth=0.6, alpha=0.8, zorder=5)
            
            text = plt.text(center[0], center[1], el, wrap=True, color='black', 
                     horizontalalignment='center', verticalalignment='center', fontsize=3,
                     bbox=dict(boxstyle="square", ec=(0.3, 0.3, 0.3, 0.3), fc=(0.3, 0.3, 0.3, 0.3)))
            text._get_wrap_line_width = lambda : 1000
            drawn_texts.append(text)

        if osm_map_nodes_pts[0].numel():
            plt.scatter(osm_map_nodes_pts[0][:, 0], osm_map_nodes_pts[0][:, 1],linewidth=1.5, color='blue')

        if osm_map_ways_pts[0].numel():
            for i, line in enumerate(osm_map_ways_pts[0]):
                if 'highway' in osm_ways_tags[i]:
                    plt.plot(line[:, 0], line[:, 1],linewidth=2,alpha=0.8, color='green')
                else:
                    plt.plot(line[:, 0], line[:, 1],linewidth=2,alpha=0.8, color='red')
        
        # adjust_text(drawn_texts, force_text=(0.6,0.6), force_explode=(0.2, 1.0), max_move=(20,40), ensure_inside_axes=False, arrowprops=dict(arrowstyle="->", color='grey', lw=0.5))
        
        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        osm_map_path = osp.join(sample_dir, 'OSM_MAP.png')
        # bbox_inches='tight'
        plt.savefig(osm_map_path, format='png',dpi=1200)
        plt.close()
        
        # visualize pred
        scores_3d = result_dict['scores_3d']
        labels_3d = result_dict['labels_3d']
        pts_3d = result_dict['pts_3d']
        keep = scores_3d > args.score_thresh

        plt.figure(figsize=(4, 2))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')
        pred_centerlines=[]
        pred_anno = {'divider':[],'divider_dashed':[],'divider_solid':[],'divider_mixed':[],'divider_virtual':[],'ped_crossing':[],'boundary':[],'centerline':[]}
        for pred_score_3d,  pred_label_3d, pred_pts_3d in zip(scores_3d[keep], labels_3d[keep], pts_3d[keep]):
            if label_names[pred_label_3d.item()] == 'centerline':
                instance = LineString(pred_pts_3d.numpy()).simplify(0.2, preserve_topology=True)
                pts = np.array(list(instance.coords))
                pred_anno['centerline'].append(pts)
                pred_centerlines.append(pts)
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color='green',headwidth=5,headlength=6,width=0.006,alpha=0.8,zorder=-1)
            elif label_names[pred_label_3d.item()] == 'divider_dashed' or label_names[pred_label_3d.item()] == 'divider_solid': 
                pred_pts_3d = pred_pts_3d.numpy()
                pred_anno[label_names[pred_label_3d.item()]].append(pred_pts_3d)
                pts_x = pred_pts_3d[:,0]
                pts_y = pred_pts_3d[:,1]
                plt.plot(pts_x, pts_y, color=colors_plt_dict[cfg.map_classes[pred_label_3d]],linewidth=1.25,alpha=0.8,zorder=-1)
            else: 
                pred_pts_3d = pred_pts_3d.numpy()
                pred_anno[label_names[pred_label_3d.item()]].append(pred_pts_3d)
                pts_x = pred_pts_3d[:,0]
                pts_y = pred_pts_3d[:,1]
                plt.plot(pts_x, pts_y, color=colors_plt_dict[cfg.map_classes[pred_label_3d]],linewidth=1,alpha=0.8,zorder=-1)
        #         plt.scatter(pts_x, pts_y, color=colors_plt_dict[pred_label_3d],s=1,alpha=0.8,zorder=-1)

        plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
        map_path = osp.join(sample_dir, 'PRED_MAP.png')
        plt.savefig(map_path, bbox_inches='tight', format='png',dpi=1200)
        plt.close()

        rendered_cams_dict = {}
        for key, cam_dict in img_path_dict.items():
            cam_img = cv2.imread(osp.join(data_path_prefix,cam_dict['filepath']))
            render_anno_on_pv(cam_img,pred_anno,cam_dict['lidar2img'])
            if 'front' not in key:
        #         cam_img = cam_img[:,::-1,:]
                cam_img = cv2.flip(cam_img, 1)
            lw = 8
            tf = max(lw - 1, 1)
            w, h = cv2.getTextSize(caption_by_cam[key], 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            p1 = (0,0)
            p2 = (w,h+3)
            color=(0, 0, 0)
            txt_color=(255, 255, 255)
            cv2.rectangle(cam_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(cam_img,
                        caption_by_cam[key], (p1[0], p1[1] + h + 2),
                        0,
                        lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)
            rendered_cams_dict[key] = cam_img

        new_image_height = 2048
        new_image_width = 1550+2048*2
        color = (255,255,255)
        first_row_canvas = np.full((new_image_height,new_image_width, 3), color, dtype=np.uint8)
        first_row_canvas[(2048-1550):, :2048,:] = rendered_cams_dict['ring_front_left']
        first_row_canvas[:,2048:(2048+1550),:] = rendered_cams_dict['ring_front_center']
        first_row_canvas[(2048-1550):,3598:,:] = rendered_cams_dict['ring_front_right']

        new_image_height = 1550
        new_image_width = 2048*4
        color = (255,255,255)
        second_row_canvas = np.full((new_image_height,new_image_width, 3), color, dtype=np.uint8)
        second_row_canvas[:,:2048,:] = rendered_cams_dict['ring_side_left']
        second_row_canvas[:,2048:4096,:] = rendered_cams_dict['ring_rear_left']
        second_row_canvas[:,4096:6144,:] = rendered_cams_dict['ring_rear_right']
        second_row_canvas[:,6144:,:] = rendered_cams_dict['ring_side_right']

        resized_first_row_canvas = cv2.resize(first_row_canvas,(8192,2972))
        full_canvas = np.full((2972+1550,8192,3),color,dtype=np.uint8)
        full_canvas[:2972,:,:] = resized_first_row_canvas
        full_canvas[2972:,:,:] = second_row_canvas
        cams_img_path = osp.join(sample_dir,'surround_view.jpg')
        cv2.imwrite(cams_img_path, full_canvas,[cv2.IMWRITE_JPEG_QUALITY, 70])

        rendered_cams_dict = {}
        for key, cam_dict in img_path_dict.items():
            cam_img = cv2.imread(osp.join(data_path_prefix,cam_dict['filepath']))
            # render_anno_on_pv(cam_img,pred_anno,cam_dict['lidar2img'])
            if 'front' not in key:
        #         cam_img = cam_img[:,::-1,:]
                cam_img = cv2.flip(cam_img, 1)
            lw = 8
            tf = max(lw - 1, 1)
            w, h = cv2.getTextSize(caption_by_cam[key], 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            p1 = (0,0)
            p2 = (w,h+3)
            color=(0, 0, 0)
            txt_color=(255, 255, 255)
            cv2.rectangle(cam_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(cam_img,
                        caption_by_cam[key], (p1[0], p1[1] + h + 2),
                        0,
                        lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)
            rendered_cams_dict[key] = cam_img

        new_image_height = 2048
        new_image_width = 1550+2048*2
        color = (255,255,255)
        first_row_canvas = np.full((new_image_height,new_image_width, 3), color, dtype=np.uint8)
        first_row_canvas[(2048-1550):, :2048,:] = rendered_cams_dict['ring_front_left']
        first_row_canvas[:,2048:(2048+1550),:] = rendered_cams_dict['ring_front_center']
        first_row_canvas[(2048-1550):,3598:,:] = rendered_cams_dict['ring_front_right']

        new_image_height = 1550
        new_image_width = 2048*4
        color = (255,255,255)
        second_row_canvas = np.full((new_image_height,new_image_width, 3), color, dtype=np.uint8)
        second_row_canvas[:,:2048,:] = rendered_cams_dict['ring_side_left']
        second_row_canvas[:,2048:4096,:] = rendered_cams_dict['ring_rear_left']
        second_row_canvas[:,4096:6144,:] = rendered_cams_dict['ring_rear_right']
        second_row_canvas[:,6144:,:] = rendered_cams_dict['ring_side_right']

        resized_first_row_canvas = cv2.resize(first_row_canvas,(8192,2972))
        full_canvas = np.full((2972+1550,8192,3),color,dtype=np.uint8)
        full_canvas[:2972,:,:] = resized_first_row_canvas
        full_canvas[2972:,:,:] = second_row_canvas
        cams_img_path = osp.join(sample_dir,'surround_view_raw.jpg')
        cv2.imwrite(cams_img_path, full_canvas,[cv2.IMWRITE_JPEG_QUALITY, 70])

        final_dict[pts_filename] = sample_dict
        prog_bar.update()

    mmcv.dump(final_dict, osp.join(args.show_dir, 'final_dict.pkl'))
    logger.info('\n DONE vis test dataset samples gt label & pred')
if __name__ == '__main__':
    main()