import copy
from collections import defaultdict

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
import os
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random

from .nuscenes_dataset import CustomNuScenesDataset
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
from shapely.geometry import Polygon, LineString, box, MultiPolygon, MultiLineString
from mmdet.datasets.pipelines import to_tensor
import json

from pathlib import Path
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.lane_segment import LaneMarkType, LaneSegment
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.se3 import SE3
import av2.geometry.interpolate as interp_utils
import cv2

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

import shapely
from shapely.geometry import LineString

from .vectorized_map import LiDARInstanceLines, VectorizedLocalMap


NOT_RELEVANT_KEYS = {'addr', 'comment', 'contact', 'source', 'name', 'tiger',
                     'ref', 'created_by', 'nysgissam', 'wikidata', 'operator',
                     'lacounty', 'osak', 'source_ref', 'nhd', 'admin_level',
                     'wikipedia', 'yh', 'gnis', 'at_bev', 'mml', 'postal_code',
                     'raba', 'nycdoitt', 'maaamet', 'pmfsefin', 'old_name', 'official_name',
                     'chicago', 'linz', 'it', 'destination', 'date', 'lojic'
                     'geobase', 'mapillary', 'clc', 'ssr', 'unsigned_ref', 'naptan'
                     'mvdgis', 'linz2osm', 'gns', 'note', 'metcouncil', 'url',
                     'route_ref', 'gtfs', 'uic', 'attribution', 'date', 'ts',
                     'id', 'survey', 'stif', 'network', 'naptan', 'location',
                     'tmc', 'fixme', 'wabe', 'object', 'description', 'check_date',
                     'tec', 'qroti', 'dcgis', 'website', 'short_name', 'image',
                     'NaPTANAreaCode', 'vrs', 'cxx', 'in', 'code', 'massgis', 'original_osm_id', 
                     'bbr', 'shape', 'lnam', 'redwood_city_ca', 'email', 'KSJ2',
                     'canvec', 'uuid', 'sorting_name', 'phone', 'inegi', 'ine', 
                     'brand', 'cesena', 'geobase', 'mobile', 'strazakosm', 'ipp',
                     'fhrs', 'alt_name', 'old_street', 'ksj2', 'unocha',
                     'wikimedia_commons', 'lojic', 'brn', 'fid', 'notas',
                     'fax', 'sangis', 'okato', 'nhd-shp', 'surrey', 'statscan',
                     'panoramax'}


@DATASETS.register_module()
class CustomAV2OfflineLocalMapDataset(CustomNuScenesDataset):
    r"""NuScenes Dataset.

    This datset add static map elements
    """
    MAPCLASSES = ('divider',)
    def __init__(self,
                 map_ann_file=None, 
                 queue_length=4, 
                 z_cfg = dict(
                    pred_z_flag=True,
                    gt_z_flag=True,
                 ),
                 bev_size=(200, 200), 
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 overlap_test=False, 
                 fixed_ptsnum_per_line=-1,
                 fixed_ptsnum_per_line_osm=-1,
                 eval_use_same_gt_sample_num_flag=False,
                 padding_value=-10000,
                 map_classes=None,
                 aux_seg = dict(
                    use_aux_seg=False,
                    bev_seg=False,
                    pv_seg=False,
                    seg_classes=1,
                    feat_down_sample=32,
                 ),
                 prev_bev_idx_skip_interval=1,
                 map_completion_mode=True,
                 eval_only_masked_elements=False, 
                 nlp_model_path=None,
                 smerf_osm_classes_mode=False, 
                 osm_highways_only=False, 
                 osm_augment_drop_els_rate=None,
                 osm_augment_random_offset_stddev=None,
                 osm_augment_random_angle_stddev=None,
                 osm_augment_global_shift=None,
                 osm_nlp_augment_drop_tags_el_rate=None,
                 osm_nlp_augment_drop_tags_rate=None, 
                 osm_nlp_augment_drop_non_relevant_tags_only=None, 
                 use_streammapnet_map = False
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.map_ann_file = map_ann_file
        self.z_cfg = z_cfg
        if z_cfg['gt_z_flag']:
            self.code_size = 3
        else:
            self.code_size = 2
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size

        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4]-pc_range[1]
        patch_w = pc_range[3]-pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.min_z = pc_range[2]
        self.max_z = pc_range[5]
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.fixed_num_osm = fixed_ptsnum_per_line_osm
        self.eval_use_same_gt_sample_num_flag = eval_use_same_gt_sample_num_flag
        self.aux_seg = aux_seg
        self.vector_map = VectorizedLocalMap(canvas_size=bev_size,
                                             patch_size=self.patch_size, 
                                             map_classes=self.MAPCLASSES, 
                                             fixed_ptsnum_per_line=fixed_ptsnum_per_line,
                                             padding_value=self.padding_value,
                                             code_size=self.code_size,
                                             min_z=self.min_z,
                                             max_z=self.max_z,
                                             aux_seg=aux_seg)
        self.is_vis_on_test = False
        self.map_completion_mode = map_completion_mode
        self.formatted_gt_this_run = False
        self.prev_bev_idx_skip_interval = prev_bev_idx_skip_interval
        self.eval_only_masked_elements = eval_only_masked_elements
        self.nlp_model_path = nlp_model_path
        self.nlp_tokenizer = AutoTokenizer.from_pretrained(nlp_model_path)
        self.smerf_osm_classes_mode = smerf_osm_classes_mode 
        self.osm_highways_only = osm_highways_only

        self.osm_augment_drop_els_rate = osm_augment_drop_els_rate
        self.osm_augment_random_offset_stddev = osm_augment_random_offset_stddev
        self.osm_augment_random_angle_stddev = osm_augment_random_angle_stddev
        self.osm_augment_global_shift = osm_augment_global_shift

        self.osm_nlp_augment_drop_tags_rate = osm_nlp_augment_drop_tags_rate
        self.osm_nlp_augment_drop_non_relevant_tags_only = osm_nlp_augment_drop_non_relevant_tags_only
        self.osm_nlp_augment_drop_tags_el_rate = osm_nlp_augment_drop_tags_el_rate

        self.use_streammapnet_map = use_streammapnet_map

        # from smerf code

        self.highway_val_to_smerf_osm_classes = {
            'highway: motorway': 'truck_road',
            'highway: trunk': 'truck_road',
            'highway: motorway_link': 'truck_road',
            'highway: trunk_link': 'truck_road',
            'highway: primary': 'highway',
            'highway: primary_link': 'highway',
            'highway: secondary': 'highway',
            'highway: secondary_link': 'highway',
            'highway: tertiary': 'highway',
            'highway: tertiary_link': 'highway',
            'highway: residential': 'residential',
            'highway: living_street': 'residential',
            'highway: service': 'service',
            'highway: pedestrian': 'pedestrian',
            'highway: footway': 'pedestrian',
            'highway: path': 'pedestrian',
            'highway: steps': 'pedestrian',
            'highway: road': 'road',
            'highway: busway': 'bus_way',
            'highway: bus_guideway': 'bus_way'
        }

        self.smerf_osm_classes_to_id = {
            'truck_road': 1,
            'highway': 2,
            'residential': 3,
            'service': 4,
            'pedestrian': 5,
            'road': 6,
            'bus_way': 7
        }

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        
        data = mmcv.load(ann_file, file_format='pkl')
        # import pdb;pdb.set_trace()
        data_infos = list(sorted(data['samples'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        # data_infos = [ data_info.update(dict(token= str(data_info['timestamp']+data_info['log_id'])))  for data_info in data_infos]
        self.metadata = None
        self.version = None
        return data_infos

    @classmethod
    def get_map_classes(cls, map_classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if map_classes is None:
            return cls.MAPCLASSES

        if isinstance(map_classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(map_classes)
        elif isinstance(map_classes, (tuple, list)):
            class_names = map_classes
        else:
            raise ValueError(f'Unsupported type {type(map_classes)} of map classes.')

        return class_names
    
    def osm_map_pipeline(self, example, input_dict, use_augmentation=False):

        # print(use_augmentation)

        def fixed_num_sampled_points(lstring_list, patch_size, min_z=-1, max_z=1):

            max_x = patch_size[1] / 2
            max_y = patch_size[0] / 2

            instance_points_list = []
            for instance in lstring_list:
                # instance_array = np.array(list(instance.coords))
                # interpolated_instance = interp_utils.interp_arc(t=self.fixed_num, points=instance_array)
                distances = np.linspace(0, instance.length, self.fixed_num_osm)
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances])
                if instance.has_z:
                    sampled_points = sampled_points.reshape(-1,3)
                else:
                    sampled_points = sampled_points.reshape(-1,2)
                # import pdb;pdb.set_trace()
                instance_points_list.append(sampled_points)
            instance_points_array = np.array(instance_points_list)
            instance_points_tensor = to_tensor(instance_points_array)
            instance_points_tensor = instance_points_tensor.to(
                                dtype=torch.float32)

            if instance.has_z:
                instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-max_x,max=max_x)
                instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-max_y,max=max_y)
                instance_points_tensor[:,:,2] = torch.clamp(instance_points_tensor[:,:,2], min=min_z,max=max_z)
            else:
                instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-max_x,max=max_x)
                instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-max_y,max=max_y)
            return instance_points_tensor
        
        def has_not_relevant_key(string, not_relevant_keys):
            split = [string]
            if ':' in string:
                split = string.split(':')
                for el in split:
                    if '_' in el:
                        split.extend(el.split('_'))
                        split.remove(el)
                        break
                    
            elif '_' in string:
                split = string.split('_')
            
            for str in split:
                if str.strip().lower() in not_relevant_keys:
                    return True
            return False
        
        def drop_tags(tags, drop_rate, non_relevant_tags_only, drop_els_rate):
            keep_tags = []
            no_els_map = len(tags)
            keep_indices_els = np.sort(np.random.choice(no_els_map, int(no_els_map*(1 - drop_els_rate)), replace=False))

            for i, tagset in enumerate(tags):

                if i in keep_indices_els:
                    keep_tags.append(tagset)
                    continue

                tagset_list = tagset.split(',')[:-1] # last element is always ' '
                
                # print(tagset_list)
                
                if non_relevant_tags_only:
                    tagset_list_rel = []
                    tagset_list_non_rel = []
                    for tag in tagset_list:
                        if has_not_relevant_key(tag, NOT_RELEVANT_KEYS):
                            tagset_list_non_rel.append(tag)
                        else:
                            tagset_list_rel.append(tag)
                    tagset_list = tagset_list_non_rel
                
                # print(tagset_list_rel)
                # print(tagset_list)
                
                no_els = len(tagset_list)
                keep_indices = np.sort(np.random.choice(no_els, round(no_els*(1 - drop_rate)), replace=False))
                if non_relevant_tags_only:
                    keep_tagset = ''
                    if tagset_list_rel:
                        keep_tagset = keep_tagset + ','.join(tagset_list_rel) + ', '
                    if tagset_list and len(keep_indices) > 0:
                        keep_tagset = keep_tagset + ','.join([tagset_list[i] for i in keep_indices]) + ', '
                    keep_tags.append(keep_tagset)
                else:
                    keep_tags.append(','.join([tagset_list[i] for i in keep_indices]) + ', ')

            return keep_tags

        # import pdb;pdb.set_trace()
        
        #============================================================================================================================================
        
        if input_dict['ann_info']['osm_map_nodes_tags'] and not self.osm_highways_only:

            if use_augmentation and self.osm_nlp_augment_drop_tags_rate:
                nodes_keep_tags = drop_tags(input_dict['ann_info']['osm_map_nodes_tags'], 
                                            self.osm_nlp_augment_drop_tags_rate, 
                                            self.osm_nlp_augment_drop_non_relevant_tags_only, 
                                            self.osm_nlp_augment_drop_tags_el_rate)
                osm_map_nodes_tags_tokenized = self.nlp_tokenizer(nodes_keep_tags, padding=True)
            else:
                osm_map_nodes_tags_tokenized = self.nlp_tokenizer(input_dict['ann_info']['osm_map_nodes_tags'], padding=True)
            
            osm_map_nodes_tags_tokenized_tensor = {}
            for key, val in osm_map_nodes_tags_tokenized.items():
                osm_map_nodes_tags_tokenized_tensor[key] = [to_tensor(entry) for entry in val]
        else:
            osm_map_nodes_tags_tokenized_tensor = defaultdict(list)

        if use_augmentation and self.osm_augment_drop_els_rate:
            no_els = len(osm_map_nodes_tags_tokenized_tensor['input_ids'])
            keep_indices_node = np.sort(np.random.choice(no_els, round(no_els*(1 - self.osm_augment_drop_els_rate)), replace=False))
            drop_indices_node = [i for i in range(0, no_els) if i not in keep_indices_node]
            for key, val in osm_map_nodes_tags_tokenized_tensor.items():
                osm_map_nodes_tags_tokenized_tensor[key] = [val[i] for i in keep_indices_node]

        if input_dict['ann_info']['osm_map_ways_tags']:

            if use_augmentation and self.osm_nlp_augment_drop_tags_rate:
                ways_keep_tags = drop_tags(input_dict['ann_info']['osm_map_ways_tags'], 
                                           self.osm_nlp_augment_drop_tags_rate, 
                                           self.osm_nlp_augment_drop_non_relevant_tags_only,
                                           self.osm_nlp_augment_drop_tags_el_rate)
                osm_map_ways_tags_tokenized = self.nlp_tokenizer(ways_keep_tags, padding=True)
            else:
                osm_map_ways_tags_tokenized = self.nlp_tokenizer(input_dict['ann_info']['osm_map_ways_tags'], padding=True)

            osm_map_ways_tags_tokenized_tensor = {}
            for key, val in osm_map_ways_tags_tokenized.items():
                osm_map_ways_tags_tokenized_tensor[key] = [to_tensor(entry) for entry in val]
        else:
            osm_map_ways_tags_tokenized_tensor = defaultdict(list)

        if use_augmentation and self.osm_augment_drop_els_rate:
            no_els = len(osm_map_ways_tags_tokenized_tensor['input_ids'])
            keep_indices_way = np.sort(np.random.choice(no_els, round(no_els*(1 - self.osm_augment_drop_els_rate)), replace=False))
            drop_indices_way = [i for i in range(0, no_els) if i not in keep_indices_way]
            for key, val in osm_map_ways_tags_tokenized_tensor.items():
                osm_map_ways_tags_tokenized_tensor[key] = [val[i] for i in keep_indices_way]

        if input_dict['ann_info']['osm_map_relations_tags'] and not self.osm_highways_only:

            if use_augmentation and self.osm_nlp_augment_drop_tags_rate:
                rels_keep_tags = drop_tags(input_dict['ann_info']['osm_map_relations_tags'], 
                                           self.osm_nlp_augment_drop_tags_rate, 
                                           self.osm_nlp_augment_drop_non_relevant_tags_only,
                                           self.osm_nlp_augment_drop_tags_el_rate)
                osm_map_relations_tags_tokenized = self.nlp_tokenizer(rels_keep_tags, padding=True)
            else:
                osm_map_relations_tags_tokenized = self.nlp_tokenizer(input_dict['ann_info']['osm_map_relations_tags'], padding=True)
            osm_map_relations_tags_tokenized_tensor = {}
            for key, val in osm_map_relations_tags_tokenized.items():
                osm_map_relations_tags_tokenized_tensor[key] = [to_tensor(entry) for entry in val]
        else:
            osm_map_relations_tags_tokenized_tensor = defaultdict(list)

        if use_augmentation and self.osm_augment_drop_els_rate:
            no_els = len(osm_map_relations_tags_tokenized_tensor['input_ids'])
            keep_indices_rel = np.sort(np.random.choice(no_els, round(no_els*(1 - self.osm_augment_drop_els_rate)), replace=False))
            drop_indices_rel = [i for i in range(0, no_els) if i not in keep_indices_rel]
            for key, val in osm_map_relations_tags_tokenized_tensor.items():
                osm_map_relations_tags_tokenized_tensor[key] = [val[i] for i in keep_indices_rel]

        #============================================================================================================================================

        if self.smerf_osm_classes_mode and input_dict['ann_info']['osm_map_ways_tags']:

            def get_smerf_class(tags):
                for key, val in self.highway_val_to_smerf_osm_classes.items():
                    if key in tags:
                        return self.smerf_osm_classes_to_id[val]
                # print('No fittting SMERF class found!')
                return 0

            osm_map_ways_smerf_classes = [get_smerf_class(tags) for tags in input_dict['ann_info']['osm_map_ways_tags']]
            example['osm_map_ways_smerf_classes'] = DC(to_tensor(osm_map_ways_smerf_classes), cpu_only=False)

        else:
            example['osm_map_ways_smerf_classes'] = DC(to_tensor([]), cpu_only=False)
        #============================================================================================================================================
        
        osm_map_relations_node_member_tags_tokenized_tensor = defaultdict(list)
        if input_dict['ann_info']['osm_map_relations_node_member_tags'] and not self.osm_highways_only:
            for i, el in enumerate(input_dict['ann_info']['osm_map_relations_node_member_tags']):
                if use_augmentation and self.osm_augment_drop_els_rate and i not in keep_indices_rel:
                    continue
                if use_augmentation and self.osm_augment_drop_els_rate:
                    keep = [True if idx in keep_indices_node else False for idx in input_dict['ann_info']['osm_map_relations_node_member_indices'][i]]
                else:
                    keep = [True for idx in input_dict['ann_info']['osm_map_relations_node_member_indices'][i]]

                if any(keep):

                    if use_augmentation and self.osm_nlp_augment_drop_tags_rate:
                        rels_member_keep_tags = drop_tags([it for keep_val, it in zip(keep, el) if keep_val], 
                                                          self.osm_nlp_augment_drop_tags_rate, 
                                                          self.osm_nlp_augment_drop_non_relevant_tags_only,
                                                          self.osm_nlp_augment_drop_tags_el_rate)
                        osm_map_relations_node_member_tags_tokenized = self.nlp_tokenizer(rels_member_keep_tags)
                    else:
                        osm_map_relations_node_member_tags_tokenized = self.nlp_tokenizer([it for keep_val, it in zip(keep, el) if keep_val])

                    for key, val in osm_map_relations_node_member_tags_tokenized.items():
                        osm_map_relations_node_member_tags_tokenized_tensor[key].append([to_tensor(entry) for entry in val])
                else:
                    osm_map_relations_node_member_tags_tokenized_tensor['input_ids'].append([])
                    osm_map_relations_node_member_tags_tokenized_tensor['token_type_ids'].append([])
                    osm_map_relations_node_member_tags_tokenized_tensor['attention_mask'].append([])

         #============================================================================================================================================
        
        osm_map_relations_way_member_tags_tokenized_tensor = defaultdict(list)
        if input_dict['ann_info']['osm_map_relations_way_member_tags'] and not self.osm_highways_only:
            for i, el in enumerate(input_dict['ann_info']['osm_map_relations_way_member_tags']):
                if use_augmentation and self.osm_augment_drop_els_rate and i not in keep_indices_rel:
                    continue
                if use_augmentation and self.osm_augment_drop_els_rate:
                    keep = [True if idx in keep_indices_way else False for idx in input_dict['ann_info']['osm_map_relations_way_member_indices'][i]]
                else:
                    keep = [True for idx in input_dict['ann_info']['osm_map_relations_way_member_indices'][i]]  

                if any(keep):

                    if use_augmentation and self.osm_nlp_augment_drop_tags_rate:
                        rels_member_keep_tags = drop_tags([it for keep_val, it in zip(keep, el) if keep_val], 
                                                          self.osm_nlp_augment_drop_tags_rate, 
                                                          self.osm_nlp_augment_drop_non_relevant_tags_only,
                                                          self.osm_nlp_augment_drop_tags_el_rate)
                        osm_map_relations_way_member_tags_tokenized = self.nlp_tokenizer(rels_member_keep_tags)
                    else:
                        osm_map_relations_way_member_tags_tokenized = self.nlp_tokenizer([it for keep_val, it in zip(keep, el) if keep_val])

                    for key, val in osm_map_relations_way_member_tags_tokenized.items():
                        osm_map_relations_way_member_tags_tokenized_tensor[key].append([to_tensor(entry) for entry in val])
                else:
                    osm_map_relations_way_member_tags_tokenized_tensor['input_ids'].append([])
                    osm_map_relations_way_member_tags_tokenized_tensor['token_type_ids'].append([])
                    osm_map_relations_way_member_tags_tokenized_tensor['attention_mask'].append([])

         #============================================================================================================================================
        
        osm_map_relations_relation_member_tags_tokenized_tensor = defaultdict(list)
        if input_dict['ann_info']['osm_map_relations_relation_member_tags'] and not self.osm_highways_only:
            for i, el in enumerate(input_dict['ann_info']['osm_map_relations_relation_member_tags']):
                if use_augmentation and self.osm_augment_drop_els_rate and i not in keep_indices_rel:
                    continue
                if use_augmentation and self.osm_augment_drop_els_rate:
                    keep = [True if idx in keep_indices_rel else False for idx in input_dict['ann_info']['osm_map_relations_relation_member_indices'][i]]
                else:
                    keep = [True for idx in input_dict['ann_info']['osm_map_relations_relation_member_indices'][i]]

                if any(keep):

                    if use_augmentation and self.osm_nlp_augment_drop_tags_rate:
                        rels_member_keep_tags = drop_tags([it for keep_val, it in zip(keep, el) if keep_val], 
                                                          self.osm_nlp_augment_drop_tags_rate, 
                                                          self.osm_nlp_augment_drop_non_relevant_tags_only)
                        osm_map_relations_relation_member_tags_tokenized = self.nlp_tokenizer(rels_member_keep_tags)
                    else:
                        osm_map_relations_relation_member_tags_tokenized = self.nlp_tokenizer([it for keep_val, it in zip(keep, el) if keep_val])

                    for key, val in osm_map_relations_relation_member_tags_tokenized.items():
                        osm_map_relations_relation_member_tags_tokenized_tensor[key].append([to_tensor(entry) for entry in val])
                else:
                    osm_map_relations_relation_member_tags_tokenized_tensor['input_ids'].append([])
                    osm_map_relations_relation_member_tags_tokenized_tensor['token_type_ids'].append([])
                    osm_map_relations_relation_member_tags_tokenized_tensor['attention_mask'].append([])
        
        # import pdb;pdb.set_trace()
        
        #============================================================================================================================================
        
        example['osm_map_nodes_tags_input_ids'] = DC(osm_map_nodes_tags_tokenized_tensor['input_ids'], cpu_only=False)
        example['osm_map_ways_tags_input_ids'] = DC(osm_map_ways_tags_tokenized_tensor['input_ids'], cpu_only=False)
        example['osm_map_relations_tags_input_ids'] = DC(osm_map_relations_tags_tokenized_tensor['input_ids'], cpu_only=False)

        example['osm_map_relations_node_member_tags_input_ids'] = DC(osm_map_relations_node_member_tags_tokenized_tensor['input_ids'], cpu_only=False)
        example['osm_map_relations_way_member_tags_input_ids'] = DC(osm_map_relations_way_member_tags_tokenized_tensor['input_ids'], cpu_only=False)
        example['osm_map_relations_relation_member_tags_input_ids'] = DC(osm_map_relations_relation_member_tags_tokenized_tensor['input_ids'], cpu_only=False)

        #============================================================================================================================================

        example['osm_map_nodes_tags_token_type_ids'] = DC(osm_map_nodes_tags_tokenized_tensor['token_type_ids'], cpu_only=False)
        example['osm_map_ways_tags_token_type_ids'] = DC(osm_map_ways_tags_tokenized_tensor['token_type_ids'], cpu_only=False)
        example['osm_map_relations_tags_token_type_ids'] = DC(osm_map_relations_tags_tokenized_tensor['token_type_ids'], cpu_only=False)
        
        example['osm_map_relations_node_member_tags_token_type_ids'] = DC(osm_map_relations_node_member_tags_tokenized_tensor['token_type_ids'], cpu_only=False)
        example['osm_map_relations_way_member_tags_token_type_ids'] = DC(osm_map_relations_way_member_tags_tokenized_tensor['token_type_ids'], cpu_only=False)
        example['osm_map_relations_relation_member_tags_token_type_ids'] = DC(osm_map_relations_relation_member_tags_tokenized_tensor['token_type_ids'], cpu_only=False)

        #============================================================================================================================================
        
        example['osm_map_nodes_tags_attention_mask'] = DC(osm_map_nodes_tags_tokenized_tensor['attention_mask'], cpu_only=False)
        example['osm_map_ways_tags_attention_mask'] = DC(osm_map_ways_tags_tokenized_tensor['attention_mask'], cpu_only=False)
        example['osm_map_relations_tags_attention_mask'] = DC(osm_map_relations_tags_tokenized_tensor['attention_mask'], cpu_only=False)
        
        example['osm_map_relations_node_member_tags_attention_mask'] = DC(osm_map_relations_node_member_tags_tokenized_tensor['attention_mask'], cpu_only=False)
        example['osm_map_relations_way_member_tags_attention_mask'] = DC(osm_map_relations_way_member_tags_tokenized_tensor['attention_mask'], cpu_only=False)
        example['osm_map_relations_relation_member_tags_attention_mask'] = DC(osm_map_relations_relation_member_tags_tokenized_tensor['attention_mask'], cpu_only=False)

        #============================================================================================================================================
        
        def filter_els(els, drop_indices):
            filtered_els = []
            for el in els:
                if el in drop_indices:
                    continue
                dropped_count = len([i for i in drop_indices if i < el])
                filtered_els.append(el - dropped_count)

            return filtered_els
        
        if use_augmentation and self.osm_augment_drop_els_rate:
            osm_map_relations_node_member_indices = [filter_els(el, drop_indices_node) for i, el in enumerate(input_dict['ann_info']['osm_map_relations_node_member_indices']) if i in keep_indices_rel]
            osm_map_relations_way_member_indices = [filter_els(el, drop_indices_way) for i, el in enumerate(input_dict['ann_info']['osm_map_relations_way_member_indices']) if i in keep_indices_rel]
            osm_map_relations_relation_member_indices = [filter_els(el, drop_indices_rel) for i, el in enumerate(input_dict['ann_info']['osm_map_relations_relation_member_indices']) if i in keep_indices_rel]
        else:
            osm_map_relations_node_member_indices = input_dict['ann_info']['osm_map_relations_node_member_indices']
            osm_map_relations_way_member_indices = input_dict['ann_info']['osm_map_relations_way_member_indices']
            osm_map_relations_relation_member_indices = input_dict['ann_info']['osm_map_relations_relation_member_indices']

        if not self.osm_highways_only:
            example['osm_map_relations_node_member_indices'] = DC([to_tensor(el) for el in osm_map_relations_node_member_indices], cpu_only=False)
            example['osm_map_relations_way_member_indices'] = DC([to_tensor(el) for el in osm_map_relations_way_member_indices], cpu_only=False)
            example['osm_map_relations_relation_member_indices'] = DC([to_tensor(el) for el in osm_map_relations_relation_member_indices], cpu_only=False)
        else: 
            example['osm_map_relations_node_member_indices'] = DC(to_tensor([]), cpu_only=False)
            example['osm_map_relations_way_member_indices'] = DC(to_tensor([]), cpu_only=False)
            example['osm_map_relations_relation_member_indices'] = DC(to_tensor([]), cpu_only=False)

        # if osm_map_relations_way_member_indices:
        #     import pdb;pdb.set_trace()
       
        if use_augmentation and self.osm_augment_drop_els_rate:
            node_pts = np.array([input_dict['ann_info']['osm_map_nodes_pts'][i] for i in keep_indices_node])
        else:
            node_pts = input_dict['ann_info']['osm_map_nodes_pts']

        if use_augmentation and self.osm_augment_random_offset_stddev:
            global_offset = np.random.normal(scale=self.osm_augment_random_offset_stddev, size=(np.array(input_dict['ann_info']['osm_map_ways_pts'][0].coords).shape[1]))[np.newaxis, ...]
            global_rot_angle = np.random.normal(scale=self.osm_augment_random_angle_stddev)
            

        if use_augmentation and self.osm_augment_random_offset_stddev and node_pts.shape[0]:

            if not self.osm_augment_global_shift:
                offsets_node = np.random.normal(scale=self.osm_augment_random_offset_stddev, size=(node_pts.shape[0], node_pts.shape[1]))
            else:
                offsets_node = np.ones((node_pts.shape[0], node_pts.shape[1])) * global_offset
            
            offsets_node = np.clip(offsets_node, a_min=-2*self.osm_augment_random_offset_stddev, a_max=2*self.osm_augment_random_offset_stddev)
            node_pts = node_pts + offsets_node
        
        # example['osm_map_nodes_pts'] = DC(to_tensor(node_pts), cpu_only=False)
        
        if not self.osm_highways_only:
            example['osm_map_nodes_pts'] = DC(to_tensor(node_pts), cpu_only=False)
        else:
            example['osm_map_nodes_pts'] = DC(to_tensor([]), cpu_only=False)
         
        if input_dict['ann_info']['osm_map_ways_pts']:
            if use_augmentation and self.osm_augment_drop_els_rate:
                ways_pts = [input_dict['ann_info']['osm_map_ways_pts'][i] for i in keep_indices_way]
            else:
                ways_pts = input_dict['ann_info']['osm_map_ways_pts']    
            resampled_pts = fixed_num_sampled_points(ways_pts, self.patch_size)
            
            if use_augmentation and self.osm_augment_random_angle_stddev:
                ways_lstrings = [LineString(pts.numpy()) for pts in resampled_pts]

                if not self.osm_augment_global_shift:
                    rotate_angles_way = np.random.normal(scale=self.osm_augment_random_angle_stddev, size=(resampled_pts.shape[0]))
                else:
                    rotate_angles_way = np.ones((resampled_pts.shape[0])) * global_rot_angle

                rotate_angles_way = np.clip(rotate_angles_way, a_min=-2*self.osm_augment_random_angle_stddev, a_max=2*self.osm_augment_random_angle_stddev)

                ways_lstrings_rot = [shapely.affinity.rotate(lstring, rot_angle, origin='centroid') for lstring, rot_angle in zip(ways_lstrings, rotate_angles_way)]
                resampled_pts = to_tensor(np.array([lstring.coords for lstring in ways_lstrings_rot]))
            
            if use_augmentation and self.osm_augment_random_offset_stddev:

                if not self.osm_augment_global_shift:
                    offsets_way = np.random.normal(scale=self.osm_augment_random_offset_stddev, size=(resampled_pts.shape[0], resampled_pts.shape[2]))
                else:
                    offsets_way = np.ones((resampled_pts.shape[0], resampled_pts.shape[2])) * global_offset

                offsets_way = np.clip(offsets_way, a_min=-2*self.osm_augment_random_offset_stddev, a_max=2*self.osm_augment_random_offset_stddev)
                resampled_pts = resampled_pts + offsets_way[:, np.newaxis, :]
            
            example['osm_map_ways_pts'] = DC(resampled_pts, cpu_only=False)
        else:
            example['osm_map_ways_pts'] = DC(to_tensor(input_dict['ann_info']['osm_map_ways_pts']), cpu_only=False)                                        



        return example
    

    def vectormap_pipeline(self, example, input_dict):
        '''
        `example` type: <class 'dict'>
            keys: 'img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img';
                  all keys type is 'DataContainer';
                  'img_metas' cpu_only=True, type is dict, others are false;
                  'gt_labels_3d' shape torch.size([num_samples]), stack=False,
                                padding_value=0, cpu_only=False
                  'gt_bboxes_3d': stack=False, cpu_only=True
        '''
        # import ipdb;ipdb.set_trace()

        anns_results = self.vector_map.gen_vectorized_samples(input_dict['annotation'] if 'annotation' in input_dict.keys() else input_dict['ann_info'],
                     example=example, feat_down_sample=self.aux_seg['feat_down_sample'])
        
        '''
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index
        '''
        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        else:
            gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
            except:
                # empty tensor, will be passed in train, 
                # but we preserve it for test
                gt_vecs_pts_loc = gt_vecs_pts_loc
        example['gt_labels_3d'] = DC(gt_vecs_label, cpu_only=False)
        example['gt_bboxes_3d'] = DC(gt_vecs_pts_loc, cpu_only=True)
        # import pdb;pdb.set_trace()
        # if self.is_vis_on_test:
        #     lidar2global_translation = to_tensor(lidar2global_translation)
        #     example['lidar2global_translation'] = DC(lidar2global_translation, cpu_only=True)
        # else:
        # example['img_metas'].data['lidar2global_translation'] = lidar2global_translation
        if anns_results['gt_semantic_mask'] is not None:
            example['gt_seg_mask'] = DC(to_tensor(anns_results['gt_semantic_mask']), cpu_only=False)
        if anns_results['gt_pv_semantic_mask'] is not None:
            example['gt_pv_seg_mask'] = DC(to_tensor(anns_results['gt_pv_semantic_mask']), cpu_only=False) 
        if anns_results['gt_map_data_idx'] is not None:
            example['gt_map_data_idx'] = DC(to_tensor(anns_results['gt_map_data_idx']), cpu_only=False) 
        if anns_results['gt_masked'] is not None:
            example['gt_masked'] = DC(to_tensor(anns_results['gt_masked']), cpu_only=False) 

        # import pdb;pdb.set_trace()

        return example

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []

        # temporal
        prev_indexs_list = list(range(index - self.prev_bev_idx_skip_interval*self.queue_length, index, self.prev_bev_idx_skip_interval))
        random.shuffle(prev_indexs_list)
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
        ##

        input_dict = self.get_data_info(index)
        if input_dict is None:
            print("Returning NONE cause no input_Dict!")
            return None
        frame_idx = input_dict['timestamp']
        scene_token = input_dict['log_id']
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        example = self.vectormap_pipeline(example,input_dict)

        #import pdb;pdb.set_trace()

        if 'osm_map_ways_pts' in input_dict['ann_info']:
            example = self.osm_map_pipeline(example,input_dict, use_augmentation=True)

        if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None
        # self.vis_gt(example)
        data_queue.insert(0, example)

        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                print("Returning NONE cause no input_Dict!")
                return None
            if input_dict['timestamp'] < frame_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                example = self.vectormap_pipeline(example,input_dict)
                if self.filter_empty_gt and \
                        (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                    print("Returning NONE!")
                    return None
                frame_idx = input_dict['timestamp']
            data_queue.insert(0, copy.deepcopy(example))

        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch_max
        input_dict = dict(
            timestamp=info['timestamp'],
            pts_filename=info['lidar_path'],
            lidar_path=info['lidar_path'],
            ego2global_translation=info['e2g_translation'],
            ego2global_rotation=info['e2g_rotation'],
            log_id=info['log_id'],
            scene_token=info['log_id'],
        )
        if self.modality['use_camera']:
            image_paths = []
            cam_intrinsics = []
            ego2img_rts = []
            ego2cam_rts = []
            cam_types = []
            cam2ego_rts = []
            input_dict["camego2global"] = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['img_fpath'])
                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = cam_info["intrinsics"]
                # input_dict["camera_intrinsics"].append(camera_intrinsics)

                # ego2img, ego = lidar
                ego2cam_rt = cam_info['extrinsics']
                cam2ego_rts.append(np.matrix(ego2cam_rt).I)

                intrinsic = cam_info['intrinsics']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                ego2img_rt = (viewpad @ ego2cam_rt)
                ego2img_rts.append(ego2img_rt)
                ego2cam_rts.append(ego2cam_rt)
                cam_intrinsics.append(viewpad)
                cam_types.append(cam_type)

                camego2global = np.eye(4, dtype=np.float32)
                camego2global[:3, :3] = cam_info['e2g_rotation']
                camego2global[:3, 3] = cam_info['e2g_translation']
                camego2global = torch.from_numpy(camego2global)

                input_dict["camego2global"].append(camego2global)
                
            lidar2ego = np.eye(4).astype(np.float32)
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=ego2img_rts,  # 认为lidar和ego是同一个坐标系
                    camera_intrinsics=cam_intrinsics,
                    ego2cam=ego2cam_rts,
                    camera2ego=cam2ego_rts,
                    cam_type=cam_types,
                    lidar2ego=lidar2ego,
                ))
        
        # if not self.test_mode:
        #     # annos = self.get_ann_info(index)
        #     input_dict['ann_info'] = dict()
        input_dict['ann_info'] = info['annotation']
        
        translation = input_dict['ego2global_translation']
        can_bus = np.ones(18)
        # can_bus.extend(translation.tolist())
        can_bus[:3] = translation
        rotation = Quaternion._from_matrix(input_dict['ego2global_rotation'])
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict['can_bus'] = can_bus
        # import pdb;pdb.set_trace()
        return input_dict

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.is_vis_on_test or self.map_completion_mode:
            example = self.vectormap_pipeline(example, input_dict)

        # import pdb;pdb.set_trace()

        if 'osm_map_ways_pts' in input_dict['ann_info']:
            example = self.osm_map_pipeline(example,input_dict, use_augmentation=True)

        return example

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
            
    def _format_gt(self):
        gt_annos = []

        # import pdb; pdb.set_trace()

        print('Start to convert gt map format...')
        assert self.map_ann_file is not None
        if (not os.path.exists(self.map_ann_file)) :
            dataset_length = len(self)
            prog_bar = mmcv.ProgressBar(dataset_length)
            mapped_class_names = self.MAPCLASSES
            for sample_id in range(dataset_length):
                sample_token = self.data_infos[sample_id]['token']
                gt_anno = {}
                gt_anno['sample_token'] = sample_token
                # gt_sample_annos = []
                gt_sample_dict = {}
                gt_sample_dict = self.vectormap_pipeline(gt_sample_dict, self.data_infos[sample_id])
                gt_labels = gt_sample_dict['gt_labels_3d'].data.numpy()
                gt_vecs = gt_sample_dict['gt_bboxes_3d'].data.instance_list

                if self.eval_only_masked_elements:
                    gt_masked = gt_sample_dict['gt_masked'].data.numpy()
                    gt_labels = gt_labels[gt_masked]
                    gt_vecs = [gt_vecs[i] for i in range(0, len(gt_vecs)) if gt_masked[i]]

                # import pdb;pdb.set_trace()
                gt_vec_list = []
                for i, (gt_label, gt_vec) in enumerate(zip(gt_labels, gt_vecs)):
                    name = mapped_class_names[gt_label]
                    anno = dict(
                        pts=np.array(list(gt_vec.coords))[:,:self.code_size],
                        pts_num=len(list(gt_vec.coords)),
                        cls_name=name,
                        type=gt_label,
                    )
                    gt_vec_list.append(anno)
                gt_anno['vectors']=gt_vec_list
                gt_annos.append(gt_anno)

                prog_bar.update()
            nusc_submissions = {
                'GTs': gt_annos
            }
            print('\n GT anns writes to', self.map_ann_file)
            mmcv.dump(nusc_submissions, self.map_ann_file)
        else:
            print(f'{self.map_ann_file} exist, not update')

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        assert self.map_ann_file is not None
        pred_annos = []
        mapped_class_names = self.MAPCLASSES

        # import pdb;pdb.set_trace()

        print('Start to convert map detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            pred_anno = {}
            vecs = output_to_vecs(det, self.eval_only_masked_elements)
            sample_token = self.data_infos[sample_id]['token']
            pred_anno['sample_token'] = sample_token
            pred_vec_list=[]
            for i, vec in enumerate(vecs):
                name = mapped_class_names[vec['label']]
                anno = dict(
                    # sample_token=sample_token,
                    pts=vec['pts'],
                    pts_num=len(vec['pts']),
                    cls_name=name,
                    type=vec['label'],
                    confidence_level=vec['score'])
                pred_vec_list.append(anno)
                # annos.append(nusc_anno)
            # nusc_annos[sample_token] = annos
            pred_anno['vectors'] = pred_vec_list
            pred_annos.append(pred_anno)


        if not os.path.exists(self.map_ann_file) or not self.formatted_gt_this_run:
            self._format_gt()
            self.formatted_gt_this_run = True
        else:
            print(f'{self.map_ann_file} exist, not update')
        # with open(self.map_ann_file,'r') as f:
        #     GT_anns = json.load(f)
        # gt_annos = GT_anns['GTs']
        nusc_submissions = {
            'meta': self.modality,
            'results': pred_annos,
            # 'GTs': gt_annos
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'av2map_results.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def to_gt_vectors(self,
                      gt_dict):
        # import pdb;pdb.set_trace()
        gt_labels = gt_dict['gt_labels_3d'].data
        gt_instances = gt_dict['gt_bboxes_3d'].data.instance_list

        gt_vectors = []

        for gt_instance, gt_label in zip(gt_instances, gt_labels):
            pts, pts_num = sample_pts_from_line(gt_instance, patch_size=self.patch_size)
            gt_vectors.append({
                'pts': pts,
                'pts_num': pts_num,
                'type': int(gt_label)
            })
        vector_num_list = {}
        for i in range(self.NUM_MAPCLASSES):
            vector_num_list[i] = []
        for vec in gt_vectors:
            if vector['pts_num'] >= 2:
                vector_num_list[vector['type']].append((LineString(vector['pts'][:vector['pts_num']]), vector.get('confidence_level', 1)))
        return gt_vectors

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='chamfer',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from projects.mmdet3d_plugin.datasets.map_utils.mean_ap import eval_map
        from projects.mmdet3d_plugin.datasets.map_utils.mean_ap import format_res_gt_by_classes
        result_path = osp.abspath(result_path)
        # import pdb;pdb.set_trace()
        detail = dict()
        
        print('Formating results & gts by classes')
        with open(result_path,'r') as f:
            pred_results = json.load(f)
        gen_results = pred_results['results']
        with open(self.map_ann_file,'r') as ann_f:
            gt_anns = json.load(ann_f)
        annotations = gt_anns['GTs']
        cls_gens, cls_gts = format_res_gt_by_classes(result_path,
                                                     gen_results,
                                                     annotations,
                                                     cls_names=self.MAPCLASSES,
                                                     num_pred_pts_per_instance=self.fixed_num,
                                                     eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,
                                                     pc_range=self.pc_range,
                                                     code_size=self.code_size)

        # import pdb;pdb.set_trace()
        
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        for metric in metrics:
            print('-*'*10+f'use metric:{metric}'+'-*'*10)

            if metric == 'chamfer':
                patch_h = self.pc_range[4]-self.pc_range[1]
                patch_w = self.pc_range[3]-self.pc_range[0]

                if patch_w > 65 and self.use_streammapnet_map:
                    thresholds = [1.0,1.5,2.0]
                else:
                    thresholds = [0.5,1.0,1.5]

                print("Thresholds: " + str(thresholds))
                # thresholds = [1.0,2.0,3.0]
            elif metric == 'iou':
                thresholds= np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds),self.NUM_MAPCLASSES))

            for i, thr in enumerate(thresholds):
                print('-*'*10+f'threshhold:{thr}'+'-*'*10)
                mAP, cls_ap = eval_map(
                                gen_results,
                                annotations,
                                cls_gens,
                                cls_gts,
                                threshold=thr,
                                cls_names=self.MAPCLASSES,
                                logger=logger,
                                num_pred_pts_per_instance=self.fixed_num,
                                pc_range=self.pc_range,
                                metric=metric,
                                code_size=self.code_size)
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']

            for i, name in enumerate(self.MAPCLASSES):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail['AV2Map_{}/{}_AP'.format(metric,name)] =  cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail['AV2Map_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()

            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer':
                        detail['AV2Map_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]
                    elif metric == 'iou':
                        if thr == 0.5 or thr == 0.75:
                            detail['AV2Map_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]

        return detail


    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name], metric=metric)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, metric=metric)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict


def output_to_vecs(detection, eval_only_masked_elements, assigned_only=False):

    if eval_only_masked_elements:
        num_unmasked = torch.count_nonzero(torch.logical_not(detection['gt_mask']))

        gt_mask = detection['gt_mask']
        assigned_pred_inds = detection['assigned_pred_inds']

        if assigned_only:
            box3d = detection['bbox_preds'][assigned_pred_inds[gt_mask]].numpy()
        else:
            box3d = detection['bbox_preds'][num_unmasked:].numpy()

        labels = torch.argmax(detection['cls_scores'], dim=1)
        labels = labels.numpy()
        scores = detection['cls_scores'][range(0, labels.shape[0]), labels].sigmoid()
        
        if assigned_only:
            scores = scores[assigned_pred_inds[gt_mask]].numpy()
            pts = detection['pts_preds'][assigned_pred_inds[gt_mask]].numpy()
            labels = labels[assigned_pred_inds[gt_mask]]
        else:
            scores = scores[num_unmasked:].numpy()
            pts = detection['pts_preds'][num_unmasked:].numpy()
            labels = labels[num_unmasked:]

    else:    
        box3d = detection['boxes_3d'].numpy()
        scores = detection['scores_3d'].numpy()
        labels = detection['labels_3d'].numpy()
        pts = detection['pts_3d'].numpy()

    # import pdb;pdb.set_trace()
    
    vec_list = []
    # import pdb;pdb.set_trace()
    for i in range(box3d.shape[0]):
        vec = dict(
            bbox = box3d[i], # xyxy
            label=labels[i],
            score=scores[i],
            pts=pts[i],
        )
        vec_list.append(vec)
    return vec_list

def sample_pts_from_line(line, 
                         fixed_num=-1,
                         sample_dist=1,
                         normalize=False,
                         patch_size=None,
                         padding=False,
                         num_samples=250,):
    if fixed_num < 0:
        distances = np.arange(0, line.length, sample_dist)
        if line.has_z:
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 3)
        else:
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
    else:
        # fixed number of points, so distance is line.length / fixed_num
        distances = np.linspace(0, line.length, fixed_num)
        if line.has_z:
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 3)
        else:
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

    if normalize:
        sampled_points[:,:2] = sampled_points[:,:2] / np.array([patch_size[1], patch_size[0]])

    num_valid = len(sampled_points)

    if not padding or fixed_num > 0:
        # fixed num sample can return now!
        return sampled_points, num_valid

    # fixed distance sampling need padding!
    num_valid = len(sampled_points)

    if fixed_num < 0:
        if num_valid < num_samples:
            padding = np.zeros((num_samples - len(sampled_points), sampled_points.shape[-1]))
            sampled_points = np.concatenate([sampled_points, padding], axis=0)
        else:
            sampled_points = sampled_points[:num_samples, :]
            num_valid = num_samples

        if normalize:
            sampled_points[:,:2] = sampled_points[:,:2] / np.array([patch_size[1], patch_size[0]])
            num_valid = len(sampled_points)

    return sampled_points[:,:2], num_valid