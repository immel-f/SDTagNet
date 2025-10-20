from lanelet2.ml_converter import MapDataInterface, LineStringType, toPointMatrix
from lanelet2.core import (BasicPoint3d, Lanelet, LaneletMap,
                           LineString3d, Point2d, Point3d, getId)
import lanelet2
from collections import defaultdict
from copy import deepcopy
import signal
from functools import partial
from multiprocessing import Pool
import multiprocessing
from random import sample
import random
import time
import mmcv
import logging
from pathlib import Path
from os import path as osp
import os
import av2.geometry.utm
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.lane_segment import LaneMarkType, LaneSegment
from av2.map.map_api import ArgoverseStaticMap
from tqdm import tqdm
import time
import argparse
import networkx as nx
from av2.map.map_primitives import Polyline
from nuscenes.map_expansion.map_api import NuScenesMapExplorer
from shapely import affinity, ops
from shapely.geometry import Polygon, LineString, box, MultiPolygon, MultiLineString
from shapely.strtree import STRtree
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from av2.geometry.se3 import SE3
import numpy as np
import math
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from scipy.spatial import distance, KDTree
import warnings
warnings.filterwarnings("ignore")

import osm_parser
import av2_to_wgs_conversion

from shapely.geometry import LineString, box, Polygon, LinearRing
from shapely.geometry.base import BaseGeometry
from shapely import ops
import numpy as np
from scipy.spatial import distance
from typing import List, Optional, Tuple
from numpy.typing import NDArray


try:
    from tools.sdtagnet.map_element_utils import calc_masked_elements
except:
    from map_element_utils import calc_masked_elements


CAM_NAMES = ['ring_front_center', 'ring_front_right', 'ring_front_left',
             'ring_rear_right', 'ring_rear_left', 'ring_side_right', 'ring_side_left',
             # 'stereo_front_left', 'stereo_front_right',
             ]
# some fail logs as stated in av2
# https://github.com/argoverse/av2-api/blob/05b7b661b7373adb5115cf13378d344d2ee43906/src/av2/map/README.md#training-online-map-inference-models
FAIL_LOGS = [
    # official
    '75e8adad-50a6-3245-8726-5e612db3d165',
    '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
    'af170aac-8465-3d7b-82c5-64147e94af7d',
    '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
    # observed
    '01bb304d-7bd8-35f8-bbef-7086b688e35e',
    '453e5558-6363-38e3-bf9b-42b5ba0a6f1d',
    # observed ll2_custom
    '8940f5f1-13e0-3094-99ba-da2d17639774',
    'c08279c0-10b4-3d21-b13f-a1c1a0b87f8b',
    'c96a09c8-46ed-391f-8a66-c46fa8b76029'
]

AV2_LANEMARKTYPE_TO_LL2 = {
    LaneMarkType.DASH_SOLID_YELLOW: 'dashed_solid',
    LaneMarkType.DASH_SOLID_WHITE: 'dashed_solid',

    LaneMarkType.DASHED_WHITE: 'dashed',
    LaneMarkType.DASHED_YELLOW: 'dashed',

    LaneMarkType.DOUBLE_SOLID_YELLOW: 'solid',
    LaneMarkType.DOUBLE_SOLID_WHITE: 'solid',

    LaneMarkType.DOUBLE_DASH_YELLOW: 'dashed',
    LaneMarkType.DOUBLE_DASH_WHITE: 'dashed',

    LaneMarkType.SOLID_YELLOW: 'solid',
    LaneMarkType.SOLID_WHITE: 'solid',

    LaneMarkType.SOLID_DASH_WHITE: 'solid_dashed',
    LaneMarkType.SOLID_DASH_YELLOW: 'solid_dashed',

    LaneMarkType.SOLID_BLUE: 'solid',

    LaneMarkType.NONE: 'virtual',

    LaneMarkType.UNKNOWN: 'unknown'
}

RANDOM_MASKING_LIST = [
    ['ego_lane'],
    ['ego_road'],
    ['centerline'],
    ['ped_crossing'],
    ['boundary'],
    ['divider_solid', 'divider_dashed'],
    ['boundary', 'divider_dashed', 'divider_solid', 'ped_crossing'],
    ['centerline', 'divider_dashed', 'divider_solid', 'ped_crossing'],
    ['boundary', 'centerline', 'divider_dashed', 'divider_solid', 'ped_crossing'],
]

OSM_MAX_NUM_WAYS = 0
OSM_MAX_NUMEL = 0


def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        help='specify the root path of dataset')
    parser.add_argument(
        '--osm-map-root',
        type=str,
        help='specify the root path of osm maps')
    parser.add_argument(
        '--out-root',
        type=str,
        help='specify the output path of the generated annotations')
    parser.add_argument(
        '--pc-range',
        type=float,
        nargs='+',
        default=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0],
        help='specify the perception point cloud range')
    parser.add_argument(
        '--nproc',
        type=int,
        default=64,
        required=False,
        help='workers to process data')
    parser.add_argument(
        '--use-mixed',
        type=bool,
        default=False,
        required=False,
        help='Use the mixed divider type (solid dashed or dashed solid) for labels. If false (default), mixed dividers will be classified as solid')
    parser.add_argument(
        '--use-virtual',
        type=bool,
        default=False,
        required=False,
        help='Use the virtual divider type for labels. If false, virtual dividers will be excluded from labels')
    parser.add_argument(
        '--remove-not-relevant-keys',
        type=bool,
        default=False,
        required=False,
        help='If true, not relevant keys will be removed from osm tags')
    parser.add_argument(
        '--masked-elements',
        nargs='+',
        type=str,
        default=None,
        required=False,
        help="""
             Elements that should be masked from the map information given to the network. 
             If none are given, the network will not recieve any map information during training.
             Possible Options:
             ego_lane: masks out all labels associated with the ego lane
             ego_road: masks out all labels associated with the ego road
             random: randomly selects a masking type for a sample
             random_whole_dataset: duplicates each sample for each available masking type (e.g. 8 masking types = 
             8x the dataset annotations stacked, once for each masking type)
             <list of element types>: masks out all elements with the specified type, e.g. divider_solid, divider_dashed, centerline etc.
             """)
    args = parser.parse_args()
    return args

# def track_job(job, update_interval=2):
#     while job._number_left > 0:
#         print("Tasks remaining = {0}".format(
#         job._number_left * job._chunksize))
#         time.sleep(update_interval)


def timeout_handler(timeout_msg, signum, frame):
    print(timeout_msg)
    raise Exception("end of time")


def create_av2_infos_mp(root_path,
                        osm_root_path,
                        info_prefix,
                        dest_path=None,
                        split='train',
                        num_multithread=96,
                        pc_range=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0],
                        use_mixed=False,
                        use_virtual=True,
                        masked_elements=None,
                        remove_not_relevant_keys=False):
    """Create info file of av2 dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        dest_path (str): Path to store generated file, default to root_path
        split (str): Split of the data.
            Default: 'train'
    """
    root_path = osp.join(root_path, split)
    if dest_path is None:
        dest_path = root_path

    loader = AV2SensorDataLoader(Path(root_path), Path(root_path))
    log_ids = list(loader.get_log_ids())
    # import pdb;pdb.set_trace()
    for l in FAIL_LOGS:
        if l in log_ids:
            log_ids.remove(l)

    print('collecting samples...')
    start_time = time.time()
    print('num cpu:', multiprocessing.cpu_count())
    print(f'using {num_multithread} threads')
    print("iterations needed in total: " + str(len(log_ids)))

    print("range: " + str(pc_range))

    # to supress logging from av2.utils.synchronization_database
    sdb_logger = logging.getLogger('av2.utils.synchronization_database')
    prev_level = sdb_logger.level
    sdb_logger.setLevel(logging.CRITICAL)

    log_ids_track = deepcopy(log_ids)

    results = []
    fn = partial(get_data_from_logid, loader=loader, data_root=root_path, osm_root_path=osm_root_path, pc_range=pc_range,
                 use_mixed=use_mixed, use_virtual=use_virtual, masked_elements=masked_elements, remove_not_relevant_keys=remove_not_relevant_keys)
    pool = multiprocessing.Pool(num_multithread)
    try:
        for samples, discarded, log_id in tqdm(pool.imap(fn, log_ids)):
            results.append((samples, discarded))
            log_ids_track.remove(log_id)
            if len(log_ids_track) < 5:
                print(log_ids_track)
        # for log_id in tqdm(log_ids):
        #     samples, discarded, log_id = fn(log_id)
        #     results.append((samples, discarded))
        #     log_ids_track.remove(log_id)
        #     if len(log_ids_track) < 5:
        #         print(log_ids_track)
    except KeyboardInterrupt:
        logging.warning("got Ctrl+C")
    finally:
        pool.terminate()
        pool.join()

    samples = []
    discarded = 0
    sample_idx = 0
    for _samples, _discarded in results:
        for i in range(len(_samples)):
            _samples[i]['sample_idx'] = sample_idx
            sample_idx += 1
        samples += _samples
        discarded += _discarded

    sdb_logger.setLevel(prev_level)
    print(f'{len(samples)} available samples, {discarded} samples discarded')

    print('collected in {}s'.format(time.time()-start_time))
    infos = dict(samples=samples)

    info_path = osp.join(dest_path,
                         '{}_map_infos_{}.pkl'.format(info_prefix, split))
    print(f'saving results to {info_path}')
    mmcv.dump(infos, info_path)
    # mmcv.dump(samples, info_path)


def get_data_from_logid(log_id,
                        loader: AV2SensorDataLoader,
                        data_root,
                        osm_root_path,
                        pc_range=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0],
                        use_mixed=False,
                        use_virtual=True,
                        masked_elements=None,
                        remove_not_relevant_keys=False):
    samples = []
    discarded = 0

    timeout_msg = "TIMEOUT ON get_data_from_logid WITH LOG_ID: " + str(log_id)
    signal.signal(signal.SIGALRM, partial(timeout_handler, timeout_msg))
    signal.alarm(70000)

    log_map_dirpath = Path(osp.join(data_root, log_id, "map"))
    vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
    if not len(vector_data_fnames) == 1:
        raise RuntimeError(
            f"JSON file containing vector map data is missing (searched in {log_map_dirpath})")
    vector_data_fname = vector_data_fnames[0]
    vector_data_json_path = vector_data_fname
    avm = ArgoverseStaticMap.from_json(vector_data_json_path)

    osm_file_path = osp.join(osm_root_path, avm.log_id + '.osm')
    with open(osm_file_path, 'r') as osm_file:
        osm_map = osm_parser.parse(osm_file)
    av2_city_name = avm.log_id[-14:-11]
    av2_city_name = av2.geometry.utm.CityName(av2_city_name)  
    osm_map.build_node_way_lists(av2_city_name)

    # We use lidar timestamps to query all sensors.
    # The frequency is 10Hz
    cam_timestamps = loader._sdb.per_log_lidar_timestamps_index[log_id]

    for ts in cam_timestamps:
        cam_ring_fpath = [loader.get_closest_img_fpath(
            log_id, cam_name, ts
        ) for cam_name in CAM_NAMES]
        lidar_fpath = loader.get_closest_lidar_fpath(log_id, ts)

        # If bad sensor synchronization, discard the sample
        if None in cam_ring_fpath or lidar_fpath is None:
            discarded += 1
            continue

        cams = {}
        for i, cam_name in enumerate(CAM_NAMES):
            pinhole_cam = loader.get_log_pinhole_camera(log_id, cam_name)
            cam_timestamp_ns = int(cam_ring_fpath[i].stem)
            cam_city_SE3_ego = loader.get_city_SE3_ego(
                log_id, cam_timestamp_ns)
            cams[cam_name] = dict(
                img_fpath=str(cam_ring_fpath[i]),
                intrinsics=pinhole_cam.intrinsics.K,
                extrinsics=pinhole_cam.extrinsics,
                e2g_translation=cam_city_SE3_ego.translation,
                e2g_rotation=cam_city_SE3_ego.rotation,
            )

        city_SE3_ego = loader.get_city_SE3_ego(log_id, int(ts))
        e2g_translation = city_SE3_ego.translation
        e2g_rotation = city_SE3_ego.rotation
        info = dict(
            e2g_translation=e2g_translation,
            e2g_rotation=e2g_rotation,
            cams=cams,
            lidar_path=str(lidar_fpath),
            # map_fpath=map_fname,
            timestamp=str(ts),
            log_id=log_id,
            token=str(log_id+'_'+str(ts)))

        if 'random_whole_dataset' in masked_elements:
            for masked_elements_random in RANDOM_MASKING_LIST:
                map_anno = extract_local_map(
                    avm, osm_map, e2g_translation, e2g_rotation, pc_range, use_mixed, use_virtual, masked_elements_random, remove_not_relevant_keys)
                info_cpy = deepcopy(info)
                info_cpy["annotation"] = map_anno
                samples.append(info_cpy)
        elif 'random' in masked_elements:
            masked_elements_random = random.choice(RANDOM_MASKING_LIST)
            map_anno = extract_local_map(
                avm, osm_map, e2g_translation, e2g_rotation, pc_range, use_mixed, use_virtual, masked_elements_random, remove_not_relevant_keys)
            info["annotation"] = map_anno
            samples.append(info)
        else:
            map_anno = extract_local_map(
                avm, osm_map, e2g_translation, e2g_rotation, pc_range, use_mixed, use_virtual, masked_elements, remove_not_relevant_keys)
            info["annotation"] = map_anno
            samples.append(info)

    signal.alarm(0)

    return samples, discarded, log_id


def extract_local_map(avm, osm_map, e2g_translation, e2g_rotation, pc_range, use_mixed, use_virtual, masked_elements, remove_not_relevant_keys):
    
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    patch_size = (patch_h, patch_w)
    map_pose = e2g_translation[:2]
    rotation = Quaternion._from_matrix(e2g_rotation)
    patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])
    patch_angle = quaternion_yaw(rotation) / np.pi * 180

    city_SE2_ego = SE3(e2g_rotation, e2g_translation)
    ego_SE3_city = city_SE2_ego.inverse()

    result_dict_osm_map = generate_osm_map_info(avm, osm_map, patch_box, patch_angle, ego_SE3_city, remove_not_relevant_keys)

    map_anno = dict(
        divider=[],
        ped_crossing=[],
        boundary=[],
        divider_masked=[],
        boundary_masked=[],
        ped_crossing_masked=[],
        osm_map_nodes_pts=[],
        osm_map_nodes_tags=[],
        osm_map_ways_pts=[],
        osm_map_ways_tags=[],
        osm_map_relations_tags=[],
        osm_map_relations_node_member_indices=[],
        osm_map_relations_way_member_indices=[],
        osm_map_relations_relation_member_indices=[],
        osm_map_relations_node_member_tags=[],
        osm_map_relations_way_member_tags=[],
        osm_map_relations_relation_member_tags=[]
    )

    map_anno['osm_map_nodes_pts'] = result_dict_osm_map['osm_map_nodes_pts']
    map_anno['osm_map_nodes_tags'] = result_dict_osm_map['osm_map_nodes_tags']
    map_anno['osm_map_ways_pts'] = result_dict_osm_map['osm_map_ways_pts']
    map_anno['osm_map_ways_tags'] = result_dict_osm_map['osm_map_ways_tags']
    map_anno['osm_map_relations_tags'] = result_dict_osm_map['osm_map_relations_tags']
    map_anno['osm_map_relations_node_member_indices'] = result_dict_osm_map['osm_map_relations_node_member_indices']
    map_anno['osm_map_relations_way_member_indices'] = result_dict_osm_map['osm_map_relations_way_member_indices']
    map_anno['osm_map_relations_relation_member_indices'] = result_dict_osm_map['osm_map_relations_relation_member_indices']
    map_anno['osm_map_relations_node_member_tags'] = result_dict_osm_map['osm_map_relations_node_member_tags']
    map_anno['osm_map_relations_way_member_tags'] = result_dict_osm_map['osm_map_relations_way_member_tags']
    map_anno['osm_map_relations_relation_member_tags'] = result_dict_osm_map['osm_map_relations_relation_member_tags']

    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    nearby_dividers = generate_nearby_dividers(avm, e2g_translation,e2g_rotation,patch)
    map_anno['ped_crossing'] = get_scene_ped_crossings(avm,e2g_translation,e2g_rotation, (patch_w, patch_h), polygon_ped=True)  
    map_anno['boundary'] = extract_local_boundary(avm, ego_SE3_city, patch_box, patch_angle,patch_size)

    all_dividers = extract_local_divider(nearby_dividers, ego_SE3_city, patch_box, patch_angle,patch_size)
    map_anno['divider'] = remove_boundary_dividers(all_dividers,map_anno['boundary'])

    for el_type in ['boundary', 'divider', 'ped_crossing']:
        map_anno[el_type + '_masked'] = np.array([True for bd in map_anno[el_type]])

    return map_anno


def split_collections(geom: BaseGeometry) -> List[Optional[BaseGeometry]]:
    ''' Split Multi-geoms to list and check is valid or is empty.
        
    Args:
        geom (BaseGeometry): geoms to be split or validate.
    
    Returns:
        geometries (List): list of geometries.
    '''
    assert geom.geom_type in ['MultiLineString', 'LineString', 'MultiPolygon', 
        'Polygon', 'GeometryCollection'], f"got geom type {geom.geom_type}"
    if 'Multi' in geom.geom_type:
        outs = []
        for g in geom.geoms:
            if g.is_valid and not g.is_empty:
                outs.append(g)
        return outs
    else:
        if geom.is_valid and not geom.is_empty:
            return [geom,]
        else:
            return []

def get_drivable_area_contour(drivable_areas: List[Polygon], 
                              roi_size: Tuple) -> List[LineString]:
    ''' Extract drivable area contours to get list of boundaries.

    Args:
        drivable_areas (list): list of drivable areas.
        roi_size (tuple): bev range size
    
    Returns:
        boundaries (List): list of boundaries.
    '''
    max_x = roi_size[0] / 2
    max_y = roi_size[1] / 2

    # a bit smaller than roi to avoid unexpected boundaries on edges
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    
    exteriors = []
    interiors = []
    
    for poly in drivable_areas:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)
    
    results = []
    for ext in exteriors:
        # NOTE: we make sure all exteriors are clock-wise
        # such that each boundary's right-hand-side is drivable area
        # and left-hand-side is walk way
        
        if ext.is_ccw:
            ext = LinearRing(list(ext.coords)[::-1])
        lines = ext.intersection(local_patch)
        if lines.geom_type == 'GeometryCollection' and len(lines) == 0:
            continue
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        assert lines.geom_type in ['MultiLineString', 'LineString']
        
        results.extend(split_collections(lines))

    for inter in interiors:
        # NOTE: we make sure all interiors are counter-clock-wise
        if not inter.is_ccw:
            inter = LinearRing(list(inter.coords)[::-1])
        lines = inter.intersection(local_patch)
        if lines.geom_type == 'GeometryCollection' and len(lines) == 0:
            continue
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        assert lines.geom_type in ['MultiLineString', 'LineString']
        
        results.extend(split_collections(lines))

    return results

def get_ped_crossing_contour(polygon: Polygon, 
                             local_patch: box) -> Optional[LineString]:
    ''' Extract ped crossing contours to get a closed polyline.
    Different from `get_drivable_area_contour`, this function ensures a closed polyline.

    Args:
        polygon (Polygon): ped crossing polygon to be extracted.
        local_patch (tuple): local patch params
    
    Returns:
        line (LineString): a closed line
    '''

    ext = polygon.exterior
    if not ext.is_ccw:
        ext = LinearRing(list(ext.coords)[::-1])
    lines = ext.intersection(local_patch)
    if lines.type != 'LineString':
        # remove points in intersection results
        lines = [l for l in lines.geoms if l.geom_type != 'Point']
        lines = ops.linemerge(lines)
        
        # same instance but not connected.
        if lines.type != 'LineString':
            ls = []
            for l in lines.geoms:
                ls.append(np.array(l.coords))
            
            lines = np.concatenate(ls, axis=0)
            lines = LineString(lines)

    if not lines.is_empty:
        start = list(lines.coords[0])
        end = list(lines.coords[-1])
        if not np.allclose(start, end, atol=1e-3):
            new_line = list(lines.coords)
            new_line.append(start)
            lines = LineString(new_line) # make ped cross closed
        return lines
    
    return None

def remove_repeated_lines(lines: List[LineString]) -> List[LineString]:
    ''' Remove repeated dividers since each divider in argoverse2 is mentioned twice
    by both left lane and right lane.

    Args:
        lines (List): list of dividers

    Returns:
        lines (List): list of left dividers
    '''

    new_lines = []
    for line in lines:
        repeated = False
        for l in new_lines:
            length = min(line.length, l.length)
            
            # hand-crafted rule to check overlap
            # if line.buffer(0.01).intersection(l.buffer(0.01)).area \
            #         > 0.2 * length:
            #     repeated = True
            #     break
            area1 = line.buffer(0.1)
            area2 = l.buffer(0.1)
            inter = area1.intersection(area2).area
            union = area1.union(area2).area
            iou = inter / union
            if iou >= 0.90:
                repeated = True
                break
        
        if not repeated:
            new_lines.append(line)
    
    return new_lines

def remove_repeated_lanesegment(lane_dict):
    ''' Remove repeated dividers since each divider in argoverse2 is mentioned twice
    by both left lane and right lane.

    Args:
        lines (List): list of dividers

    Returns:
        lines (List): list of left dividers
    '''

    new_lane_dict = {}
    # for line in lines:
    for key, value in lane_dict.items():
        repeated = False
        # for l in new_lines:
        for new_key, new_value in new_lane_dict.items():
            # length = min(line.length, l.length)
            line = LineString(value['polyline'].xyz)
            l = LineString(new_value['polyline'].xyz)
            
            area1 = line.buffer(0.01)
            area2 = l.buffer(0.01)
            inter = area1.intersection(area2).area
            union = area1.union(area2).area
            iou = inter / union
            if iou >= 0.90:
                repeated = True
                break
        
        if not repeated:
            new_lane_dict[key] = value
    
    return new_lane_dict


def reassign_graph_attribute(lane_dict):
    for key, value in lane_dict.items():
        if len(value['predecessors']) > 0:
            if value['predecessors'][0] not in lane_dict.keys() or value['predecessors'][0]==key:
                value['predecessors'] = []
            else:
                lane_dict[value['predecessors'][0]]['successors']  = [key]
    for key, value in lane_dict.items():
        if len(value['successors']) > 0:
            if value['successors'][0] not in lane_dict.keys() or value['successors'][0]==key:
                value['successors'] = []
            else:
                lane_dict[value['successors'][0]]['predecessors']  = [key]

    return lane_dict


def remove_boundary_dividers(dividers: List[LineString], 
                             boundaries: List[LineString]) -> List[LineString]:
    ''' Some dividers overlaps with boundaries in argoverse2 dataset so
    we need to remove these dividers.

    Args:
        dividers (list): list of dividers
        boundaries (list): list of boundaries

    Returns:
        left_dividers (list): list of left dividers
    '''

    for idx in range(len(dividers))[::-1]:
        divider = dividers[idx]
        
        for bound in boundaries:
            length = min(divider.length, bound.length)

            # hand-crafted rule to check overlap
            if divider.buffer(0.3).intersection(bound.buffer(0.3)).area \
                    > 0.2 * length:
                # the divider overlaps boundary
                dividers.pop(idx)
                break

    return dividers

def connect_lines(lines: List[LineString]) -> List[LineString]:
    ''' Some dividers are split into multiple small parts
    so we need to connect these lines.

    Args:
        dividers (list): list of dividers
        boundaries (list): list of boundaries

    Returns:
        left_dividers (list): list of left dividers
    '''

    new_lines = []
    eps = 0.1 # threshold to identify continuous lines
    while len(lines) > 1:
        line1 = lines[0]
        merged_flag = False
        for i, line2 in enumerate(lines[1:]):
            # hand-crafted rule
            begin1 = list(line1.coords)[0]
            end1 = list(line1.coords)[-1]
            begin2 = list(line2.coords)[0]
            end2 = list(line2.coords)[-1]

            dist_matrix = distance.cdist([begin1, end1], [begin2, end2])
            if dist_matrix[0, 0] < eps:
                coords = list(line2.coords)[::-1] + list(line1.coords)
            elif dist_matrix[0, 1] < eps:
                coords = list(line2.coords) + list(line1.coords)
            elif dist_matrix[1, 0] < eps:
                coords = list(line1.coords) + list(line2.coords)
            elif dist_matrix[1, 1] < eps:
                coords = list(line1.coords) + list(line2.coords)[::-1]
            else: continue

            new_line = LineString(coords)
            lines.pop(i + 1)
            lines[0] = new_line
            merged_flag = True
            break
        
        if merged_flag: continue

        new_lines.append(line1)
        lines.pop(0)

    if len(lines) == 1:
        new_lines.append(lines[0])

    return new_lines

def transform_from(xyz: NDArray, 
                   translation: NDArray, 
                   rotation: NDArray) -> NDArray:
    ''' Transform points between different coordinate system.

    Args:
        xyz (array): original point coordinates
        translation (array): translation
        rotation (array): rotation matrix

    Returns:
        left_dividers (list): list of left dividers
    '''
    
    new_xyz = xyz @ rotation.T + translation
    return new_xyz

def generate_osm_map_info(avm, osm_map, patch_box, patch_angle, ego_SE3_city, remove_not_relevant_keys):

    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    result_dict = osm_map.get_elements_in_patch(patch, remove_not_relevant_keys)

    if result_dict['osm_map_nodes_pts'].size:
        result_dict['osm_map_nodes_pts'] = np.hstack([result_dict['osm_map_nodes_pts'], np.zeros([result_dict['osm_map_nodes_pts'].shape[0], 1])])
        result_dict['osm_map_nodes_pts'][:, 2] = ego_SE3_city.inverse().translation[2]
        result_dict['osm_map_nodes_pts'] = ego_SE3_city.transform_point_cloud(result_dict['osm_map_nodes_pts'])

    transformed_ways_points = []
    for lstring in result_dict['osm_map_ways_pts']:
        pts_3d = np.hstack([np.array(list(lstring.coords)), np.zeros([np.array(list(lstring.coords)).shape[0], 1])])
        pts_3d[:, 2] = ego_SE3_city.inverse().translation[2]
        transformed_ways_points.append(proc_line(LineString(pts_3d), ego_SE3_city))

    result_dict['osm_map_ways_pts'] = transformed_ways_points

    # import pdb;pdb.set_trace()

    # num_nodes = len(result_dict['osm_map_nodes_pts'])
    # num_ways = len(result_dict['osm_map_ways_pts'])
    # num_rels = len(result_dict['osm_map_relations_tags'])
    # num_rel_node_members = sum([len(l) for l in result_dict['osm_map_relations_node_member_indices']])
    # num_rel_way_members = sum([len(l) for l in result_dict['osm_map_relations_way_member_indices']])
    # num_rel_rel_members = sum([len(l) for l in result_dict['osm_map_relations_relation_member_indices']])
    # total_numel = num_nodes + num_ways + num_rels + num_rel_node_members + num_rel_way_members + num_rel_rel_members
    # global OSM_MAX_NUMEL, OSM_MAX_NUM_WAYS
    # if total_numel > OSM_MAX_NUMEL:
    #     OSM_MAX_NUMEL = total_numel
    # if num_ways > OSM_MAX_NUM_WAYS:
    #     OSM_MAX_NUM_WAYS = num_ways
    # 
    return result_dict


def generate_nearby_dividers(avm, e2g_translation, e2g_rotation,patch):
        def get_path(ls_dict):
            pts_G = nx.DiGraph()
            junction_pts_list = []
            tmp=ls_dict
            for key, value in tmp.items():
                centerline_geom = LineString(value['polyline'].xyz)
                centerline_pts = np.array(centerline_geom.coords).round(3)
                start_pt = centerline_pts[0]
                end_pt = centerline_pts[-1]

                for idx, pts in enumerate(centerline_pts[:-1]):
                    pts_G.add_edge(tuple(centerline_pts[idx]),tuple(centerline_pts[idx+1]))

                valid_incoming_num = 0
                for idx, pred in enumerate(value['predecessors']):
                    if pred in tmp.keys():
                        valid_incoming_num += 1
                        pred_geom = LineString(tmp[pred]['polyline'].xyz)
                        pred_pt = np.array(pred_geom.coords).round(3)[-1]

                        if pred_pt[0] == start_pt[0] and pred_pt[1] == start_pt[1] and pred_pt[2] == start_pt[2]:
                            pass
                        else:
                            pts_G.add_edge(tuple(pred_pt), tuple(start_pt))

                if valid_incoming_num > 1:
                    junction_pts_list.append(tuple(start_pt))
                
                valid_outgoing_num = 0
                for idx, succ in enumerate(value['successors']):
                    if succ in tmp.keys():
                        valid_outgoing_num += 1
                        succ_geom = LineString(tmp[succ]['polyline'].xyz)
                        succ_pt = np.array(succ_geom.coords).round(3)[0]

                        if end_pt[0] == succ_pt[0] and end_pt[1] == succ_pt[1] and end_pt[2] == succ_pt[2]:
                            pass
                        else:
                            pts_G.add_edge(tuple(end_pt), tuple(succ_pt))

                if valid_outgoing_num > 1:
                    junction_pts_list.append(tuple(end_pt))
            
            roots = (v for v, d in pts_G.in_degree() if d == 0)
            roots_list = [v for v, d in pts_G.in_degree() if d == 0]
            
            notroot_list = [v for v in pts_G.nodes if v not in roots_list]
            leaves = [v for v,d in pts_G.out_degree() if d==0]
            ### find path from each root to leaves

            all_paths = []
            for root in roots:
                for leave in leaves:
                    paths = nx.all_simple_paths(pts_G, root, leave)
                    all_paths.extend(paths)

            for single_path in all_paths:
                for single_node in single_path:
                    if single_node in notroot_list:
                        notroot_list.remove(single_node)

            final_centerline_paths = []
            for path in all_paths:
                merged_line = LineString(path)
                # pdb.set_trace()
                merged_line = merged_line.simplify(0.2, preserve_topology=True)
                final_centerline_paths.append(merged_line)

            local_centerline_paths = final_centerline_paths
            return local_centerline_paths
        
        left_lane_dict = {}
        right_lane_dict = {}

        scene_ls_list = avm.get_scenario_lane_segments()
        scene_ls_dict = dict()
        for ls in scene_ls_list:
            scene_ls_dict[ls.id] = dict(
                ls=ls,
                polygon = Polygon(ls.polygon_boundary),
                predecessors=ls.predecessors,
                successors=ls.successors
            )
        
        nearby_ls_dict = dict()
        for key, value in scene_ls_dict.items():
            polygon = value['polygon']
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    nearby_ls_dict[key] = value['ls']

        ls_dict = nearby_ls_dict
        divider_ls_dict = dict()
        for key, value in ls_dict.items():
            if not value.is_intersection:
                divider_ls_dict[key] = value

        left_lane_dict = {}
        right_lane_dict = {}
        for key,value in divider_ls_dict.items():
            if value.left_neighbor_id is not None:
                left_lane_dict[key] = dict(
                    polyline=value.left_lane_boundary,
                    predecessors = value.predecessors,
                    successors = value.successors,
                    left_neighbor_id = value.left_neighbor_id,
                )
            if value.right_neighbor_id is not None:
                right_lane_dict[key] = dict(
                    polyline = value.right_lane_boundary,
                    predecessors = value.predecessors,
                    successors = value.successors,
                    right_neighbor_id = value.right_neighbor_id,
                )

        for key, value in left_lane_dict.items():
            if value['left_neighbor_id'] in right_lane_dict.keys():
                del right_lane_dict[value['left_neighbor_id']]

        for key, value in right_lane_dict.items():
            if value['right_neighbor_id'] in left_lane_dict.keys():
                del left_lane_dict[value['right_neighbor_id']]

        left_lane_dict = remove_repeated_lanesegment(left_lane_dict)
        right_lane_dict = remove_repeated_lanesegment(right_lane_dict)

        left_lane_dict = reassign_graph_attribute(left_lane_dict)
        right_lane_dict = reassign_graph_attribute(right_lane_dict)

        left_paths = get_path(left_lane_dict)
        right_paths = get_path(right_lane_dict)
        local_dividers = left_paths + right_paths

        return local_dividers

def proc_polygon(polygon, ego_SE3_city):
    interiors = []
    exterior_cityframe = np.array(list(polygon.exterior.coords))
    exterior_egoframe = ego_SE3_city.transform_point_cloud(exterior_cityframe)
    for inter in polygon.interiors:
        inter_cityframe = np.array(list(inter.coords))
        inter_egoframe = ego_SE3_city.transform_point_cloud(inter_cityframe)
        interiors.append(inter_egoframe[:,:3])

    new_polygon = Polygon(exterior_egoframe[:,:3], interiors)
    return new_polygon
 
def proc_line(line,ego_SE3_city):
    new_line_pts_cityframe = np.array(list(line.coords))
    new_line_pts_egoframe = ego_SE3_city.transform_point_cloud(new_line_pts_cityframe)
    line = LineString(new_line_pts_egoframe[:,:3]) #TODO
    return line

def extract_local_divider(nearby_dividers, ego_SE3_city, patch_box, patch_angle,patch_size):
    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    # pdb.set_trace()
    # final_pgeom = remove_repeated_lines(nearby_dividers)
    line_list = []
    # pdb.set_trace()
    for line in nearby_dividers:
        if line.is_empty:  # Skip lines without nodes.
            continue
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            if new_line.geom_type == 'MultiLineString':
                for single_line in new_line.geoms:
                    if single_line.is_empty:
                        continue
                    single_line = proc_line(single_line,ego_SE3_city)
                    line_list.append(single_line)
            else:
                new_line = proc_line(new_line, ego_SE3_city)
                line_list.append(new_line)
    centerlines = line_list
    
    poly_centerlines = [line.buffer(0.1,
                cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre) for line in centerlines]
    index_by_id = dict((id(pt), i) for i, pt in enumerate(poly_centerlines))
    tree = STRtree(poly_centerlines)
    final_pgeom = []
    remain_idx = [i for i in range(len(centerlines))]
    for i, pline in enumerate(poly_centerlines):
        if i not in remain_idx:
            continue
        remain_idx.pop(remain_idx.index(i))

        final_pgeom.append(centerlines[i])
        for o in tree.query(pline):
            o_idx = index_by_id[id(o)]
            if o_idx not in remain_idx:
                continue
            inter = o.intersection(pline).area
            union = o.union(pline).area
            iou = inter / union
            if iou >= 0.90:
                remain_idx.pop(remain_idx.index(o_idx))

    # return [np.array(line.coords) for line in final_pgeom]
    final_pgeom = connect_lines(final_pgeom)
    return final_pgeom

def extract_local_boundary(avm, ego_SE3_city, patch_box, patch_angle,patch_size):
    boundary_list = []
    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    for da in avm.get_scenario_vector_drivable_areas():
        boundary_list.append(da.xyz)

    polygon_list = []
    for da in boundary_list:
        exterior_coords = da
        interiors = []
    #     polygon = Polygon(exterior_coords, interiors)
        polygon = Polygon(exterior_coords, interiors)
        if polygon.is_valid:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                if new_polygon.geom_type is 'Polygon':
                    if not new_polygon.is_valid:
                        continue
                    new_polygon = proc_polygon(new_polygon,ego_SE3_city)
                    if not new_polygon.is_valid:
                        continue
                elif new_polygon.geom_type is 'MultiPolygon':
                    polygons = []
                    for single_polygon in new_polygon.geoms:
                        if not single_polygon.is_valid or single_polygon.is_empty:
                            continue
                        new_single_polygon = proc_polygon(single_polygon,ego_SE3_city)
                        if not new_single_polygon.is_valid:
                            continue
                        polygons.append(new_single_polygon)
                    if len(polygons) == 0:
                        continue
                    new_polygon = MultiPolygon(polygons)
                    if not new_polygon.is_valid:
                        continue
                else:
                    raise ValueError('{} is not valid'.format(new_polygon.geom_type))

                if new_polygon.geom_type is 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                polygon_list.append(new_polygon)

    union_segments = ops.unary_union(polygon_list)
    max_x = patch_size[1] / 2
    max_y = patch_size[0] / 2
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    exteriors = []
    interiors = []
    if union_segments.geom_type != 'MultiPolygon':
        union_segments = MultiPolygon([union_segments])
    for poly in union_segments.geoms:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)


    results = []
    for ext in exteriors:
        if ext.is_ccw:
            ext.coords = list(ext.coords)[::-1]
        lines = ext.intersection(local_patch)
        if isinstance(lines, MultiLineString):
            lines = ops.linemerge(lines)
        results.append(lines)

    for inter in interiors:
        if not inter.is_ccw:
            inter.coords = list(inter.coords)[::-1]
        lines = inter.intersection(local_patch)
        if isinstance(lines, MultiLineString):
            lines = ops.linemerge(lines)
        results.append(lines)

    boundary_lines = []
    for line in results:
        if not line.is_empty:
            if line.geom_type == 'MultiLineString':
                for single_line in line.geoms:
                    boundary_lines.append(single_line)
            elif line.geom_type == 'LineString':
                boundary_lines.append(line)
            else:
                raise NotImplementedError
    return boundary_lines

def get_scene_dividers(avm,patch_box,patch_angle):
    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    scene_ls_list = avm.get_scenario_lane_segments()
    # pdb.set_trace()
    scene_ls_dict = dict()
    for ls in scene_ls_list:
        scene_ls_dict[ls.id] = dict(
            ls=ls,
            polygon = Polygon(ls.polygon_boundary),
            predecessors=ls.predecessors,
            successors=ls.successors
        )
    nearby_ls_dict = dict()
    for key, value in scene_ls_dict.items():
        polygon = value['polygon']
        if polygon.is_valid:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                nearby_ls_dict[key] = value['ls']

    ls_dict = nearby_ls_dict
    divider_ls_dict = dict()
    for key, value in ls_dict.items():
        if not value.is_intersection:
            divider_ls_dict[key] = value

    return divider_ls_dict

def get_scene_ped_crossings(avm,e2g_translation,e2g_rotation,roi_size,polygon_ped=True):

    g2e_translation = e2g_rotation.T.dot(-e2g_translation)
    g2e_rotation = e2g_rotation.T

    roi_x, roi_y = roi_size[:2]
    local_patch = box(-roi_x / 2, -roi_y / 2, roi_x / 2, roi_y / 2)
    ped_crossings = [] 
    for _, pc in avm.vector_pedestrian_crossings.items():
        edge1_xyz = pc.edge1.xyz
        edge2_xyz = pc.edge2.xyz
        ego1_xyz = transform_from(edge1_xyz, g2e_translation, g2e_rotation)
        ego2_xyz = transform_from(edge2_xyz, g2e_translation, g2e_rotation)

        # if True, organize each ped crossing as closed polylines. 
        if polygon_ped:
            vertices = np.concatenate([ego1_xyz, ego2_xyz[::-1, :]])
            p = Polygon(vertices)
            line = get_ped_crossing_contour(p, local_patch)
            if line is not None:
                if len(line.coords) < 3 or Polygon(line).area < 1:
                    continue
                ped_crossings.append(line)
        # Otherwise organize each ped crossing as two parallel polylines.
        else:
            line1 = LineString(ego1_xyz)
            line2 = LineString(ego2_xyz)
            line1_local = line1.intersection(local_patch)
            line2_local = line2.intersection(local_patch)

            # take the whole ped cross if all two edges are in roi range
            if not line1_local.is_empty and not line2_local.is_empty:
                ped_crossings.append(line1_local)
                ped_crossings.append(line2_local)

    return ped_crossings    


if __name__ == '__main__':
    args = parse_args()
    for name in ['train', 'val', 'test']:
        create_av2_infos_mp(
            root_path=args.data_root,
            osm_root_path=args.osm_map_root,
            split=name,
            info_prefix='av2',
            dest_path=args.out_root,
            pc_range=args.pc_range,
            num_multithread=args.nproc,
            use_mixed=args.use_mixed,
            use_virtual=args.use_virtual,
            masked_elements=args.masked_elements,
            remove_not_relevant_keys=args.remove_not_relevant_keys)