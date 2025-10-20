import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from sentence_transformers import SentenceTransformer

from mmcv import ConfigDict
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_positional_encoding
from mmdet3d.models import NECKS, BACKBONES
from mmdet3d.models.builder import build_backbone


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def draw_ways(image, ways, way_feats):

    # import pdb;pdb.set_trace()

    h, w, c = image.shape
    num_ways, num_way_pts, pts_dim = ways.shape

    dh = ways[:, 1:, 0] - ways[:, :-1, 0]
    dw = ways[:, 1:, 1] - ways[:, :-1, 1]

    m = torch.div(dw, dh+1e-4)
    lens = torch.sqrt(dh**2 + dw**2)

    ways_img = torch.stack([ways[:, :, 1]*h, ways[:, :, 0]*w], dim=2).to(torch.int32)
    ways_lseg_pts = torch.max(lens)*w // 7 + 1
    
    h_draw = ways_img[:, :-1, 0].flatten()
    h_draw = h_draw.unsqueeze(1) + torch.arange(0, 1.01, 1.0/ways_lseg_pts, device=dh.device).unsqueeze(0) * dh.flatten().unsqueeze(1)
    
    w_draw = m.flatten().unsqueeze(1) * (h_draw - ways_img[:, :-1, 0].flatten().unsqueeze(1)) + ways_img[:, :-1, 1].flatten().unsqueeze(1)

    pts_draw = torch.stack([h_draw.flatten(), w_draw.flatten()], dim=1)
    if way_feats.shape[0] == num_ways:
        feats_draw = torch.repeat_interleave(way_feats, int((num_way_pts-1)*(ways_lseg_pts+1)), dim=0)
    else:
        idx = torch.tensor([i for i in range(way_feats.shape[0]) if i%num_way_pts!=0], device=way_feats.device, dtype=torch.long)
        feats_draw = torch.repeat_interleave(way_feats[idx], int(ways_lseg_pts+1), dim=0)

    # if pts_draw.shape[0] != feats_draw.shape[0]:
    #     import pdb;pdb.set_trace()
    
    return draw_points(image, pts_draw.to(torch.int32), feats_draw, points_in_img_coords=True, fill7by7=True)


def draw_points(image, points, point_feats, points_in_img_coords=False, fill7by7=False):

    # import pdb;pdb.set_trace()
    
    h, w, c = image.shape
    if not points_in_img_coords:
        # import pdb;pdb.set_trace()
        points_img = torch.stack([points[:, 1]*h, points[:, 0]*w], dim=1).to(torch.int32)
    else:
        points_img = points

    # fill 3x3
    
    image[points_img[:, 0].clamp(0, h-1), points_img[:, 1].clamp(0, w-1)] = point_feats

    image[(points_img[:, 0]-1).clamp(0, h-1), points_img[:, 1].clamp(0, w-1)]     = point_feats
    image[points_img[:, 0].clamp(0, h-1), (points_img[:, 1]-1).clamp(0, w-1)]     = point_feats
    image[(points_img[:, 0]-1).clamp(0, h-1), (points_img[:, 1]-1).clamp(0, w-1)] = point_feats

    image[(points_img[:, 0]+1).clamp(0, h-1), points_img[:, 1].clamp(0, w-1)]     = point_feats
    image[points_img[:, 0].clamp(0, h-1), (points_img[:, 1]+1).clamp(0, w-1)]     = point_feats
    image[(points_img[:, 0]+1).clamp(0, h-1), (points_img[:, 1]+1).clamp(0, w-1)] = point_feats

    image[(points_img[:, 0]-1).clamp(0, h-1), (points_img[:, 1]+1).clamp(0, w-1)] = point_feats
    image[(points_img[:, 0]+1).clamp(0, h-1), (points_img[:, 1]-1).clamp(0, w-1)] = point_feats

    #===============================================================================================
    
    if fill7by7:
        # fill 5x5
        image[(points_img[:, 0]-2).clamp(0, h-1), (points_img[:, 1]+2).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]-1).clamp(0, h-1), (points_img[:, 1]+2).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]).clamp(0, h-1),   (points_img[:, 1]+2).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+1).clamp(0, h-1), (points_img[:, 1]+2).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+2).clamp(0, h-1), (points_img[:, 1]+2).clamp(0, w-1)] = point_feats

        image[(points_img[:, 0]-2).clamp(0, h-1), (points_img[:, 1]+1).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+2).clamp(0, h-1), (points_img[:, 1]+1).clamp(0, w-1)] = point_feats

        image[(points_img[:, 0]-2).clamp(0, h-1), (points_img[:, 1]).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+2).clamp(0, h-1), (points_img[:, 1]).clamp(0, w-1)] = point_feats

        image[(points_img[:, 0]-2).clamp(0, h-1), (points_img[:, 1]-1).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+2).clamp(0, h-1), (points_img[:, 1]-1).clamp(0, w-1)] = point_feats

        image[(points_img[:, 0]-2).clamp(0, h-1), (points_img[:, 1]-2).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]-1).clamp(0, h-1), (points_img[:, 1]-2).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]).clamp(0, h-1),   (points_img[:, 1]-2).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+1).clamp(0, h-1), (points_img[:, 1]-2).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+2).clamp(0, h-1), (points_img[:, 1]-2).clamp(0, w-1)] = point_feats

        # fill 7x7
        image[(points_img[:, 0]-3).clamp(0, h-1), (points_img[:, 1]+3).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]-2).clamp(0, h-1), (points_img[:, 1]+3).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]-1).clamp(0, h-1), (points_img[:, 1]+3).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]).clamp(0, h-1),   (points_img[:, 1]+3).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+1).clamp(0, h-1), (points_img[:, 1]+3).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+2).clamp(0, h-1), (points_img[:, 1]+3).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+3).clamp(0, h-1), (points_img[:, 1]+3).clamp(0, w-1)] = point_feats
        
        image[(points_img[:, 0]-3).clamp(0, h-1), (points_img[:, 1]+2).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+3).clamp(0, h-1), (points_img[:, 1]+2).clamp(0, w-1)] = point_feats
        
        image[(points_img[:, 0]-3).clamp(0, h-1), (points_img[:, 1]+1).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+3).clamp(0, h-1), (points_img[:, 1]+1).clamp(0, w-1)] = point_feats
        
        image[(points_img[:, 0]-3).clamp(0, h-1), (points_img[:, 1]).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+3).clamp(0, h-1), (points_img[:, 1]).clamp(0, w-1)] = point_feats
        
        image[(points_img[:, 0]-3).clamp(0, h-1), (points_img[:, 1]-1).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+3).clamp(0, h-1), (points_img[:, 1]-1).clamp(0, w-1)] = point_feats

        image[(points_img[:, 0]-3).clamp(0, h-1), (points_img[:, 1]-2).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+3).clamp(0, h-1), (points_img[:, 1]-2).clamp(0, w-1)] = point_feats
        
        image[(points_img[:, 0]-3).clamp(0, h-1), (points_img[:, 1]-3).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]-2).clamp(0, h-1), (points_img[:, 1]-3).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]-1).clamp(0, h-1), (points_img[:, 1]-3).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]).clamp(0, h-1),   (points_img[:, 1]-3).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+1).clamp(0, h-1), (points_img[:, 1]-3).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+2).clamp(0, h-1), (points_img[:, 1]-3).clamp(0, w-1)] = point_feats
        image[(points_img[:, 0]+3).clamp(0, h-1), (points_img[:, 1]-3).clamp(0, w-1)] = point_feats

    return image


#=====================================================================================================================================================

@torch.no_grad()
def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block, mode='reduced')
    return q.t()  # [cols, cols]


@torch.no_grad()
def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, device=None):
    """create 2D Gaussian orthogonal matrix"""
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    normalizer = final_matrix.norm(p=2, dim=1, keepdim=True)
    normalizer[normalizer == 0] = 1e-5
    final_matrix = final_matrix / normalizer

    return final_matrix


@torch.no_grad()
def orthogonal_matrix_chunk_batched(bsz, cols, device=None):
    unstructured_block = torch.randn((bsz, cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block, mode='reduced')
    return q.transpose(2, 1)  # [bsz, cols, cols]


@torch.no_grad()
def gaussian_orthogonal_random_matrix_batched(nb_samples, nb_rows, nb_columns, device=None, dtype=torch.float32):
    """create 2D Gaussian orthogonal matrix"""
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk_batched(nb_samples, nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk_batched(nb_samples, nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list, dim=1).type(dtype)
    final_matrix = F.normalize(final_matrix, p=2, dim=2)
    return final_matrix


#=====================================================================================================================================================


@POSITIONAL_ENCODING.register_module()
class SineContinuousPositionalEncoding(BaseModule):
    def __init__(self, 
                 num_feats,
                 temp=10000,
                 normalize=False,
                 range=None,
                 scale=2 * np.pi,
                 offset=0.,
                 init_cfg=None):
        super(SineContinuousPositionalEncoding, self).__init__(init_cfg)
        self.num_feats = num_feats
        self.temp = temp
        self.normalize = normalize
        self.range = torch.tensor(range) if range is not None else None
        self.offset = torch.tensor(offset) if offset is not None else None
        self.scale = scale
    
    def forward(self, x):
        """
        x: [B, N, D]

        return: [B, N, D * num_feats]
        """
        B, N, D = x.shape
        if self.normalize:
            x = (x - self.offset.to(x.device)) / self.range.to(x.device) * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temp**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x[..., None] / dim_t  # [B, N, D, num_feats]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=3).view(B, N, D * self.num_feats)
        return pos_x


#=====================================================================================================================================================


@NECKS.register_module()
class OSMMapEncoder(BaseModule):
    def __init__(self, 
                 input_dim=256, 
                 dmodel=256,
                 hidden_dim=256, 
                 orf_dim=64,
                 nlp_model_path=None,
                 nheads=8,
                 nlayers=6,
                 batch_first=True,
                 pos_encoder=None,
                 nlp_pad_token=0,
                 nlp_max_tokens=256,
                 smerf_osm_classes_mode=False,
                 smerf_num_osm_classes=8,
                 **kwargs):
        super(OSMMapEncoder, self).__init__(**kwargs)
        self.batch_dim = 0 if batch_first else 1
        self.input_dim = input_dim

        self.map_embedding = nn.Linear(input_dim, dmodel)

        self.orf_dim = orf_dim
        self.nlp_pad_token = nlp_pad_token
        self.nlp_max_tokens = nlp_max_tokens
        self.smerf_osm_classes_mode = smerf_osm_classes_mode
        self.smerf_num_osm_classes=smerf_num_osm_classes

        if nlp_model_path is not None:
            self.nlp_model = SentenceTransformer(nlp_model_path)
        else:
            raise RuntimeError("nlp_model_path needed! SMERF-like baseline still wip")
        
        if pos_encoder is not None:
            self.use_positional_encoding = True
            self.pos_encoder = build_positional_encoding(pos_encoder)
        else:
            self.use_positional_encoding = False
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dmodel, 
                                                        nhead=nheads,
                                                        dim_feedforward=hidden_dim,
                                                        batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    
    def add_orf_identifiers(self, map_features, osm_map_data, b_idx):

        if not osm_map_data['osm_map_nodes_pts'][b_idx].numel():
                num_nodes = 0
        elif len(osm_map_data['osm_map_nodes_pts'][b_idx].shape) == 1:
            osm_map_data['osm_map_nodes_pts'][b_idx] = osm_map_data['osm_map_nodes_pts'][b_idx].unsqueeze(0)
            num_nodes = len(osm_map_data['osm_map_nodes_pts'][b_idx])
        else:
            num_nodes = len(osm_map_data['osm_map_nodes_pts'][b_idx])
            
        num_ways, num_way_pts, pts_dim = osm_map_data['osm_map_ways_pts'][b_idx].shape
        num_rels = len(osm_map_data['osm_map_relations_tags_input_ids'][b_idx])
        num_rel_node_members = sum([len(l) for l in osm_map_data['osm_map_relations_node_member_indices'][b_idx]])
        num_rel_way_members = sum([len(l) for l in osm_map_data['osm_map_relations_way_member_indices'][b_idx]])
        num_rel_rel_members = sum([len(l) for l in osm_map_data['osm_map_relations_relation_member_indices'][b_idx]])
        total_numel = num_nodes + num_ways + num_rels + num_rel_node_members + num_rel_way_members + num_rel_rel_members

        n_orf_nodes = num_nodes + num_ways + num_rels
        orf_mat = gaussian_orthogonal_random_matrix(n_orf_nodes, n_orf_nodes, device=osm_map_data['osm_map_ways_pts'][b_idx].device)
        if n_orf_nodes < self.orf_dim:
            orf_node_ident = F.pad(orf_mat, (0, self.orf_dim - n_orf_nodes), 'constant', 0)
        else:
            orf_node_ident = orf_mat[:, :self.orf_dim]

        orf_idents = [torch.tile(orf_node_ident, (1, 2))]
        
        # import pdb;pdb.set_trace()self.nlp_max_tokens = nlp_max_tokens
        
        for i in range(0, num_rels):
            for member in osm_map_data['osm_map_relations_node_member_indices'][b_idx][i]:
                if member.numel():
                    orf_idents.append(torch.cat([orf_node_ident[num_nodes + num_ways + i], orf_node_ident[member.to(torch.long)]]).unsqueeze(0))

            for member in osm_map_data['osm_map_relations_way_member_indices'][b_idx][i]:
                if member.numel():
                    orf_idents.append(torch.cat([orf_node_ident[num_nodes + num_ways + i], orf_node_ident[num_nodes + member.to(torch.long)]]).unsqueeze(0))

            for member in osm_map_data['osm_map_relations_relation_member_indices'][b_idx][i]:
                if member.numel():
                    orf_idents.append(torch.cat([orf_node_ident[num_nodes + num_ways + i], orf_node_ident[num_nodes + num_ways + member.to(torch.long)]]).unsqueeze(0))
        
        orf_idents = torch.cat(orf_idents)

        # if map_features.shape[0] != orf_idents.shape[0]:
        #     import pdb;pdb.set_trace()

        map_features = torch.cat([map_features, orf_idents], dim=1)

        return map_features
    
    
    def get_nlp_model_input(self, osm_map_data, b_idx):
        nlp_model_input = dict(
            input_ids = [],
            token_type_ids = [],
            attention_mask = []
        )

        def flatten_list(list):
            return [x for xs in list for x in xs]

        nlp_model_input['input_ids'].extend(osm_map_data['osm_map_nodes_tags_input_ids'][b_idx])
        nlp_model_input['input_ids'].extend(osm_map_data['osm_map_ways_tags_input_ids'][b_idx])
        nlp_model_input['input_ids'].extend(osm_map_data['osm_map_relations_tags_input_ids'][b_idx])
        
        nlp_model_input['input_ids'].extend(flatten_list(osm_map_data['osm_map_relations_node_member_tags_input_ids'][b_idx]))
        nlp_model_input['input_ids'].extend(flatten_list(osm_map_data['osm_map_relations_way_member_tags_input_ids'][b_idx]))
        nlp_model_input['input_ids'].extend(flatten_list(osm_map_data['osm_map_relations_relation_member_tags_input_ids'][b_idx]))

        nlp_model_input['token_type_ids'].extend(osm_map_data['osm_map_nodes_tags_token_type_ids'][b_idx])
        nlp_model_input['token_type_ids'].extend(osm_map_data['osm_map_ways_tags_token_type_ids'][b_idx])
        nlp_model_input['token_type_ids'].extend(osm_map_data['osm_map_relations_tags_token_type_ids'][b_idx])

        nlp_model_input['token_type_ids'].extend(flatten_list(osm_map_data['osm_map_relations_node_member_tags_token_type_ids'][b_idx]))
        nlp_model_input['token_type_ids'].extend(flatten_list(osm_map_data['osm_map_relations_way_member_tags_token_type_ids'][b_idx]))
        nlp_model_input['token_type_ids'].extend(flatten_list(osm_map_data['osm_map_relations_relation_member_tags_token_type_ids'][b_idx]))

        nlp_model_input['attention_mask'].extend(osm_map_data['osm_map_nodes_tags_attention_mask'][b_idx])
        nlp_model_input['attention_mask'].extend(osm_map_data['osm_map_ways_tags_attention_mask'][b_idx])
        nlp_model_input['attention_mask'].extend(osm_map_data['osm_map_relations_tags_attention_mask'][b_idx])

        nlp_model_input['attention_mask'].extend(flatten_list(osm_map_data['osm_map_relations_node_member_tags_attention_mask'][b_idx]))
        nlp_model_input['attention_mask'].extend(flatten_list(osm_map_data['osm_map_relations_way_member_tags_attention_mask'][b_idx]))
        nlp_model_input['attention_mask'].extend(flatten_list(osm_map_data['osm_map_relations_relation_member_tags_attention_mask'][b_idx]))

        # import pdb;pdb.set_trace()

        device = osm_map_data['osm_map_ways_pts'][b_idx].device

        nlp_model_input['input_ids'] = [el if len(el) > 0 else torch.tensor([], device=device) for el in nlp_model_input['input_ids']]
        nlp_model_input['token_type_ids'] = [el if len(el) > 0 else torch.tensor([], device=device) for el in nlp_model_input['token_type_ids']]
        nlp_model_input['attention_mask'] = [el if len(el) > 0 else torch.tensor([], device=device) for el in nlp_model_input['attention_mask']]

        nlp_model_input['input_ids'] = nn.utils.rnn.pad_sequence(nlp_model_input['input_ids'], batch_first=True, padding_value=self.nlp_pad_token)[..., :self.nlp_max_tokens]
        nlp_model_input['token_type_ids'] = nn.utils.rnn.pad_sequence(nlp_model_input['token_type_ids'], batch_first=True, padding_value=self.nlp_pad_token)[..., :self.nlp_max_tokens]
        nlp_model_input['attention_mask'] = nn.utils.rnn.pad_sequence(nlp_model_input['attention_mask'], batch_first=True, padding_value=self.nlp_pad_token)[..., :self.nlp_max_tokens]
        
        return nlp_model_input
    
    
    def build_map_features(self, osm_map_data):
        map_features = []

        # import pdb;pdb.set_trace()

        B = len(osm_map_data['osm_map_ways_pts'])

        for b_idx in range(0, B):
            if not osm_map_data['osm_map_nodes_pts'][b_idx].numel():
                num_nodes = 0
            elif len(osm_map_data['osm_map_nodes_pts'][b_idx].shape) == 1:
                osm_map_data['osm_map_nodes_pts'][b_idx] = osm_map_data['osm_map_nodes_pts'][b_idx].unsqueeze(0)
                num_nodes = len(osm_map_data['osm_map_nodes_pts'][b_idx])
            else:
                num_nodes = len(osm_map_data['osm_map_nodes_pts'][b_idx])

            num_ways, num_way_pts, pts_dim = osm_map_data['osm_map_ways_pts'][b_idx].shape
            num_rels = len(osm_map_data['osm_map_relations_tags_input_ids'][b_idx])
            num_rel_node_members = sum([len(l) for l in osm_map_data['osm_map_relations_node_member_indices'][b_idx]])
            num_rel_way_members = sum([len(l) for l in osm_map_data['osm_map_relations_way_member_indices'][b_idx]])
            num_rel_rel_members = sum([len(l) for l in osm_map_data['osm_map_relations_relation_member_indices'][b_idx]])
            total_numel = num_nodes + num_ways + num_rels + num_rel_node_members + num_rel_way_members + num_rel_rel_members

            map_feat = torch.zeros((num_nodes+num_ways, num_way_pts, pts_dim), device=osm_map_data['osm_map_ways_pts'][b_idx].device)
            if num_nodes:
                map_feat[0:num_nodes, :, :] = torch.tile(osm_map_data['osm_map_nodes_pts'][b_idx], (1, num_way_pts)).view(num_nodes, num_way_pts, pts_dim)
            if num_ways:
                map_feat[num_nodes:, :, :] = osm_map_data['osm_map_ways_pts'][b_idx]

            if self.use_positional_encoding:
                map_feat = self.pos_encoder(map_feat)

            map_feat = map_feat.view(map_feat.shape[0], map_feat.shape[1]*map_feat.shape[2])

            # import pdb;pdb.set_trace()    
            
            if self.smerf_osm_classes_mode:
                embeddings = F.one_hot(osm_map_data['osm_map_ways_smerf_classes'][b_idx], num_classes=self.smerf_num_osm_classes)
                if num_nodes:
                  embeddings = torch.cat([torch.zeros((num_nodes, embeddings.shape[1]), device=osm_map_data['osm_map_ways_pts'][b_idx].device), embeddings])
                map_feat = torch.cat([map_feat, embeddings], dim=1)
                map_features.append(map_feat)
                continue
            
            map_feat = F.pad(map_feat, (0,0,0,total_numel - map_feat.shape[0]), 'constant', 0)
            
            nlp_model_input = self.get_nlp_model_input(osm_map_data, b_idx)
            embeddings = self.nlp_model.forward(nlp_model_input)['sentence_embedding']
            map_feat = torch.cat([map_feat, embeddings], dim=1)
            map_feat = self.add_orf_identifiers(map_feat, osm_map_data, b_idx)
            map_features.append(map_feat)
            
        return nn.utils.rnn.pad_sequence(map_features, batch_first=True, padding_value=0), None
        
    def forward(self, osm_map_data):

        map_features, bev_feats = self.build_map_features(osm_map_data)
        
        # import pdb;pdb.set_trace()
        
        map_features = self.map_embedding(map_features)  
        map_features = self.transformer_encoder(map_features) 

        return map_features, bev_feats


#=============================================================================================================================================================

class BevFeatEncoder(nn.Module):
    def __init__(self, dmodel, d_bev, d_input, d_kernel):
        super().__init__()
        self.conv1 = nn.Conv2d(d_input, d_bev, d_kernel, padding='same', padding_mode='replicate')
        self.conv2 = nn.Conv2d(d_bev, d_bev, d_kernel, padding='same', padding_mode='replicate')
        self.conv3 = nn.Conv2d(d_bev, d_bev, d_kernel, padding='same', padding_mode='replicate')
        self.emb = nn.Linear(d_bev, dmodel)

    def forward(self, x):

        # import pdb;pdb.set_trace()

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.emb(x.permute(0,2,3,1))
        return x


@NECKS.register_module()
class OSMMapEncoderPointLevel(BaseModule):
    def __init__(self, 
                 input_dim=256, 
                 dmodel=256,
                 hidden_dim=256, 
                 orf_dim=64,
                 fixed_orf_order=False,
                 use_orf_graph_ident=True,
                 nlp_model_path=None,
                 nheads=8,
                 nlayers=6,
                 batch_first=True,
                 positional_encoding_type=None,
                 pos_encoder=None,
                 nlp_pad_token=0,
                 nlp_max_tokens=256,
                 nlp_embed_dim=144,
                 render_bev_feats=False,
                 use_queries_for_bev=False,
                 bev_h=None,
                 bev_w=None,
                 pc_range=None,
                 pretrained_path=None,
                 smerf_osm_classes_mode=False,
                 pmapnet_bev_mode=False, 
                 smerf_num_osm_classes=8,
                 smerf_use_orf_graph_ident=False,
                 **kwargs):
        super(OSMMapEncoderPointLevel, self).__init__(**kwargs)

        self.batch_dim = 0 if batch_first else 1
        self.input_dim = input_dim
        self.map_embedding = nn.Linear(input_dim, dmodel)

        self.smerf_osm_classes_mode = smerf_osm_classes_mode
        self.smerf_num_osm_classes= smerf_num_osm_classes
        self.smerf_use_orf_graph_ident = smerf_use_orf_graph_ident
        self.pmapnet_bev_mode = pmapnet_bev_mode

        self.orf_dim = orf_dim
        self.fixed_orf_order = fixed_orf_order
        self.use_orf_graph_ident = use_orf_graph_ident

        self.nlp_pad_token = nlp_pad_token
        self.nlp_max_tokens = nlp_max_tokens
        self.nlp_embed_dim = nlp_embed_dim

        self.render_bev_feats = render_bev_feats
        self.use_queries_for_bev = use_queries_for_bev
        self.bev_h=bev_h
        self.bev_w=bev_w
        self.pc_range=pc_range

        if self.render_bev_feats:
            if self.use_queries_for_bev:
                self.bev_feat_enc = BevFeatEncoder(dmodel, dmodel, 3*dmodel, 7)
            elif self.smerf_osm_classes_mode:
                self.bev_feat_enc = BevFeatEncoder(dmodel, dmodel, self.smerf_num_osm_classes, 7)
            else:
                self.bev_feat_enc = BevFeatEncoder(dmodel, nlp_embed_dim, nlp_embed_dim, 7)

        if nlp_model_path is not None:
            self.nlp_model = SentenceTransformer(nlp_model_path)
        else:
            raise RuntimeError("nlp_model_path needed! SMERF-like baseline still wip")
        
        self.positional_encoding_type = positional_encoding_type
        
        if self.positional_encoding_type == 'learned':
            raise RuntimeError("not yet implemented")
        elif pos_encoder is not None:
            self.positional_encoding_type = 'custom'
            self.pos_encoder = build_positional_encoding(pos_encoder)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dmodel, 
                                                        nhead=nheads,
                                                        dim_feedforward=hidden_dim,
                                                        batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, weights_only=True))
            print("Loaded pretrained osm_map_encoder weights from " + pretrained_path)

    
    def add_orf_identifiers(self, map_features, osm_map_data, b_idx, smerf_mode=False):

        if not osm_map_data['osm_map_nodes_pts'][b_idx].numel():
                num_nodes = 0
        elif len(osm_map_data['osm_map_nodes_pts'][b_idx].shape) == 1:
            osm_map_data['osm_map_nodes_pts'][b_idx] = osm_map_data['osm_map_nodes_pts'][b_idx].unsqueeze(0)
            num_nodes = len(osm_map_data['osm_map_nodes_pts'][b_idx])
        else:
            num_nodes = len(osm_map_data['osm_map_nodes_pts'][b_idx])

        num_ways, num_way_pts, pts_dim = osm_map_data['osm_map_ways_pts'][b_idx].shape
        num_rels = len(osm_map_data['osm_map_relations_tags_input_ids'][b_idx])
        num_rel_node_members = sum([len(l) for l in osm_map_data['osm_map_relations_node_member_indices'][b_idx]])
        num_rel_way_members = sum([len(l) for l in osm_map_data['osm_map_relations_way_member_indices'][b_idx]])
        num_rel_rel_members = sum([len(l) for l in osm_map_data['osm_map_relations_relation_member_indices'][b_idx]])
        total_numel = num_nodes + num_ways + num_rels + num_rel_node_members + num_rel_way_members + num_rel_rel_members

        n_orf_nodes = num_nodes + num_ways + num_rels
        orf_mat = gaussian_orthogonal_random_matrix(n_orf_nodes, n_orf_nodes, device=osm_map_data['osm_map_ways_pts'][b_idx].device)
        if n_orf_nodes < self.orf_dim:
            orf_node_ident = F.pad(orf_mat, (0, self.orf_dim - n_orf_nodes), 'constant', 0)
        else:
            orf_node_ident = orf_mat[:, :self.orf_dim]

        orf_node_ident_tiled = torch.tile(orf_node_ident, (1, 2))
        
        if not smerf_mode:
            orf_idents = [torch.cat([orf_node_ident_tiled[:num_nodes], torch.repeat_interleave(orf_node_ident_tiled[num_nodes:num_nodes+num_ways], 
                                                                                               num_way_pts, dim=0),  orf_node_ident_tiled[num_nodes+num_ways:]], dim=0)]
        else:
            orf_idents = [torch.cat([orf_node_ident_tiled[:num_nodes], torch.repeat_interleave(orf_node_ident_tiled[num_nodes:num_nodes+num_ways], 
                                                                                   num_way_pts, dim=0)], dim=0)]
        
        #TODO: This is not correct order of rel members, FIX
        
        if self.fixed_orf_order and not smerf_mode:
            for i in range(0, num_rels):
                for member in osm_map_data['osm_map_relations_node_member_indices'][b_idx][i]:
                    if member.numel():
                        orf_idents.append(torch.cat([orf_node_ident[num_nodes + num_ways + i], orf_node_ident[member.to(torch.long)]]).unsqueeze(0))

            for i in range(0, num_rels):    
                for member in osm_map_data['osm_map_relations_way_member_indices'][b_idx][i]:
                    if member.numel():
                        orf_idents.append(torch.cat([orf_node_ident[num_nodes + num_ways + i], orf_node_ident[num_nodes + member.to(torch.long)]]).unsqueeze(0))

            for i in range(0, num_rels):
                for member in osm_map_data['osm_map_relations_relation_member_indices'][b_idx][i]:
                    if member.numel():
                        orf_idents.append(torch.cat([orf_node_ident[num_nodes + num_ways + i], orf_node_ident[num_nodes + num_ways + member.to(torch.long)]]).unsqueeze(0))
        elif not smerf_mode:
            for i in range(0, num_rels):
                for member in osm_map_data['osm_map_relations_node_member_indices'][b_idx][i]:
                    if member.numel():
                        orf_idents.append(torch.cat([orf_node_ident[num_nodes + num_ways + i], orf_node_ident[member.to(torch.long)]]).unsqueeze(0))

                for member in osm_map_data['osm_map_relations_way_member_indices'][b_idx][i]:
                    if member.numel():
                        orf_idents.append(torch.cat([orf_node_ident[num_nodes + num_ways + i], orf_node_ident[num_nodes + member.to(torch.long)]]).unsqueeze(0))

                for member in osm_map_data['osm_map_relations_relation_member_indices'][b_idx][i]:
                    if member.numel():
                        orf_idents.append(torch.cat([orf_node_ident[num_nodes + num_ways + i], orf_node_ident[num_nodes + num_ways + member.to(torch.long)]]).unsqueeze(0))
        
        orf_idents = torch.cat(orf_idents)

        # if map_features.shape[0] != orf_idents.shape[0]:
        #     import pdb;pdb.set_trace()

        map_features = torch.cat([map_features, orf_idents], dim=1)

        return map_features
    
    
    def get_nlp_model_input(self, osm_map_data, b_idx):
        nlp_model_input = dict(
            input_ids = [],
            token_type_ids = [],
            attention_mask = []
        )

        def flatten_list(list):
            return [x for xs in list for x in xs]

        nlp_model_input['input_ids'].extend(osm_map_data['osm_map_nodes_tags_input_ids'][b_idx])
        nlp_model_input['input_ids'].extend(osm_map_data['osm_map_ways_tags_input_ids'][b_idx])
        nlp_model_input['input_ids'].extend(osm_map_data['osm_map_relations_tags_input_ids'][b_idx])
        
        nlp_model_input['input_ids'].extend(flatten_list(osm_map_data['osm_map_relations_node_member_tags_input_ids'][b_idx]))
        nlp_model_input['input_ids'].extend(flatten_list(osm_map_data['osm_map_relations_way_member_tags_input_ids'][b_idx]))
        nlp_model_input['input_ids'].extend(flatten_list(osm_map_data['osm_map_relations_relation_member_tags_input_ids'][b_idx]))

        nlp_model_input['token_type_ids'].extend(osm_map_data['osm_map_nodes_tags_token_type_ids'][b_idx])
        nlp_model_input['token_type_ids'].extend(osm_map_data['osm_map_ways_tags_token_type_ids'][b_idx])
        nlp_model_input['token_type_ids'].extend(osm_map_data['osm_map_relations_tags_token_type_ids'][b_idx])

        nlp_model_input['token_type_ids'].extend(flatten_list(osm_map_data['osm_map_relations_node_member_tags_token_type_ids'][b_idx]))
        nlp_model_input['token_type_ids'].extend(flatten_list(osm_map_data['osm_map_relations_way_member_tags_token_type_ids'][b_idx]))
        nlp_model_input['token_type_ids'].extend(flatten_list(osm_map_data['osm_map_relations_relation_member_tags_token_type_ids'][b_idx]))

        nlp_model_input['attention_mask'].extend(osm_map_data['osm_map_nodes_tags_attention_mask'][b_idx])
        nlp_model_input['attention_mask'].extend(osm_map_data['osm_map_ways_tags_attention_mask'][b_idx])
        nlp_model_input['attention_mask'].extend(osm_map_data['osm_map_relations_tags_attention_mask'][b_idx])

        nlp_model_input['attention_mask'].extend(flatten_list(osm_map_data['osm_map_relations_node_member_tags_attention_mask'][b_idx]))
        nlp_model_input['attention_mask'].extend(flatten_list(osm_map_data['osm_map_relations_way_member_tags_attention_mask'][b_idx]))
        nlp_model_input['attention_mask'].extend(flatten_list(osm_map_data['osm_map_relations_relation_member_tags_attention_mask'][b_idx]))

        # import pdb;pdb.set_trace()

        device = osm_map_data['osm_map_ways_pts'][b_idx].device

        nlp_model_input['input_ids'] = [el if len(el) > 0 else torch.tensor([], device=device) for el in nlp_model_input['input_ids']]
        nlp_model_input['token_type_ids'] = [el if len(el) > 0 else torch.tensor([], device=device) for el in nlp_model_input['token_type_ids']]
        nlp_model_input['attention_mask'] = [el if len(el) > 0 else torch.tensor([], device=device) for el in nlp_model_input['attention_mask']]

        nlp_model_input['input_ids'] = nn.utils.rnn.pad_sequence(nlp_model_input['input_ids'], batch_first=True, padding_value=self.nlp_pad_token)[..., :self.nlp_max_tokens]
        nlp_model_input['token_type_ids'] = nn.utils.rnn.pad_sequence(nlp_model_input['token_type_ids'], batch_first=True, padding_value=self.nlp_pad_token)[..., :self.nlp_max_tokens]
        nlp_model_input['attention_mask'] = nn.utils.rnn.pad_sequence(nlp_model_input['attention_mask'], batch_first=True, padding_value=self.nlp_pad_token)[..., :self.nlp_max_tokens]
        
        return nlp_model_input
    
    
    def build_map_features(self, osm_map_data):
        map_features = []
        bev_feats = []
        # import pdb;pdb.set_trace()

        B = len(osm_map_data['osm_map_ways_pts'])

        for b_idx in range(0, B):

            if not osm_map_data['osm_map_nodes_pts'][b_idx].numel():
                num_nodes = 0
            elif len(osm_map_data['osm_map_nodes_pts'][b_idx].shape) == 1:
                osm_map_data['osm_map_nodes_pts'][b_idx] = osm_map_data['osm_map_nodes_pts'][b_idx].unsqueeze(0)
                num_nodes = len(osm_map_data['osm_map_nodes_pts'][b_idx])
            else:
                num_nodes = len(osm_map_data['osm_map_nodes_pts'][b_idx])

            num_ways, num_way_pts, pts_dim = osm_map_data['osm_map_ways_pts'][b_idx].shape
            num_rels = len(osm_map_data['osm_map_relations_tags_input_ids'][b_idx])
            num_rel_node_members = sum([len(l) for l in osm_map_data['osm_map_relations_node_member_indices'][b_idx]])
            num_rel_way_members = sum([len(l) for l in osm_map_data['osm_map_relations_way_member_indices'][b_idx]])
            num_rel_rel_members = sum([len(l) for l in osm_map_data['osm_map_relations_relation_member_indices'][b_idx]])
            total_numel = num_nodes + num_ways*num_way_pts + num_rels + num_rel_node_members + num_rel_way_members + num_rel_rel_members

            device = osm_map_data['osm_map_ways_pts'][b_idx].device
            dtype = osm_map_data['osm_map_ways_pts'][b_idx].dtype
            
            map_feat = torch.zeros((num_nodes+num_ways*num_way_pts, pts_dim), device=device, dtype=dtype)
            if num_nodes:
                # map_feat[0:num_nodes, :] = torch.repeat_interleave(osm_map_data['osm_map_nodes_pts'][b_idx], self.num_pts_per_vec, dim=0)
                map_feat[0:num_nodes, :] = osm_map_data['osm_map_nodes_pts'][b_idx]
            if num_ways:
                map_feat[num_nodes:, :] = osm_map_data['osm_map_ways_pts'][b_idx].view(num_ways*num_way_pts, pts_dim)

            if self.smerf_osm_classes_mode:
                map_feat = self.pos_encoder(map_feat.unsqueeze(0)).squeeze()
                if self.pmapnet_bev_mode:
                    embeddings = torch.arange(1, osm_map_data['osm_map_ways_pts'][b_idx].shape[0]+1, 
                                                       device=osm_map_data['osm_map_ways_pts'][b_idx].device).unsqueeze(1) / 256
                else:
                    embeddings = F.one_hot(osm_map_data['osm_map_ways_smerf_classes'][b_idx], num_classes=self.smerf_num_osm_classes)
                if num_nodes:
                  embeddings = torch.cat([torch.zeros((num_nodes, embeddings.shape[1]), device=osm_map_data['osm_map_ways_pts'][b_idx].device), torch.repeat_interleave(embeddings, num_way_pts, dim=0)])
                  map_feat = torch.cat([map_feat, embeddings], dim=1)
                else:
                  map_feat = torch.cat([map_feat, torch.repeat_interleave(embeddings, num_way_pts, dim=0)], dim=1)  

                if self.smerf_use_orf_graph_ident:
                    map_feat = self.add_orf_identifiers(map_feat, osm_map_data, b_idx, smerf_mode=True)

                map_features.append(map_feat)

                #print(map_feat.shape)
                if self.render_bev_feats and not self.use_queries_for_bev:
                    bev_feat = torch.zeros([self.bev_h, self.bev_w, embeddings.shape[1]], device=osm_map_data['osm_map_ways_pts'][b_idx].device)

                    if self.pmapnet_bev_mode:

                        # import pdb;pdb.set_trace()

                        if num_ways:
                            ways_normalized = normalize_2d_pts(osm_map_data['osm_map_ways_pts'][b_idx][..., 0:2], self.pc_range)
                            emb_pmapnet = torch.arange(1, osm_map_data['osm_map_ways_pts'][b_idx].shape[0]+1, 
                                                       device=osm_map_data['osm_map_ways_pts'][b_idx].device).unsqueeze(1) / 256
                            emb_pmapnet = torch.repeat_interleave(emb_pmapnet, num_way_pts, dim=0)
                            bev_feat = draw_ways(bev_feat, ways_normalized, emb_pmapnet.to(torch.float32))
                        if num_nodes:
                            nodes_normalized = normalize_2d_pts(osm_map_data['osm_map_nodes_pts'][b_idx][..., 0:2], self.pc_range)
                            emb_pmapnet_nd = torch.arange(1, osm_map_data['osm_map_nodes_pts'][b_idx].shape[0]+1, 
                                device=osm_map_data['osm_map_nodes_pts'][b_idx].device).unsqueeze(1) / 256
                            bev_feat = draw_points(bev_feat, nodes_normalized, emb_pmapnet_nd.to(torch.float32))
                        bev_feats.append(bev_feat)
                    else:
                        if num_ways:
                            ways_normalized = normalize_2d_pts(osm_map_data['osm_map_ways_pts'][b_idx][..., 0:2], self.pc_range)
                            bev_feat = draw_ways(bev_feat, ways_normalized, embeddings[num_nodes:num_nodes+num_ways].to(torch.float32))
                        if num_nodes:
                            nodes_normalized = normalize_2d_pts(osm_map_data['osm_map_nodes_pts'][b_idx][..., 0:2], self.pc_range)
                            bev_feat = draw_points(bev_feat, nodes_normalized, embeddings[:num_nodes].to(torch.float32))
                        bev_feats.append(bev_feat)

                continue
            
            nlp_model_input = self.get_nlp_model_input(osm_map_data, b_idx)
            embeddings = self.nlp_model.forward(nlp_model_input)['sentence_embedding']
            
            if not self.use_orf_graph_ident:  

                rel_pts = []
                rel_embs = []
                collected_node_members = []
                collected_way_members = []
                collected_node_member_embs = []
                collected_way_member_embs = []

                total_node_member_idx = 0
                total_way_member_idx = 0

                for i in range(0, num_rels):

                    for member in osm_map_data['osm_map_relations_node_member_indices'][b_idx][i]:
                        if member.numel():

                            collected_node_members.append(osm_map_data['osm_map_nodes_pts'][b_idx][member.to(torch.long)].unsqueeze(0))
                            rel_pts.append(osm_map_data['osm_map_nodes_pts'][b_idx][member.to(torch.long)].unsqueeze(0))

                            collected_node_member_embs.append(embeddings[num_nodes + num_ways + num_rels + total_node_member_idx].unsqueeze(0))
                            rel_embs.append(embeddings[num_nodes + num_ways + i].unsqueeze(0))

                            total_node_member_idx += 1

                    for member in osm_map_data['osm_map_relations_way_member_indices'][b_idx][i]:
                        if member.numel():

                            collected_way_members.append(osm_map_data['osm_map_ways_pts'][b_idx][member.to(torch.long)])
                            rel_pts.append(osm_map_data['osm_map_ways_pts'][b_idx][member.to(torch.long)])

                            collected_way_member_embs.append(embeddings[num_nodes + num_ways + num_rels + num_rel_node_members + total_way_member_idx].repeat(num_way_pts, 1))
                            rel_embs.append(embeddings[num_nodes + num_ways + i].repeat(num_way_pts, 1))

                            total_way_member_idx += 1

                if rel_pts:
                    rel_pts = torch.cat(rel_pts, dim=0).to(dtype)
                    rel_embs = torch.cat(rel_embs, dim=0)
                else:
                    rel_pts = torch.zeros([0, pts_dim], device=device, dtype=dtype)
                    rel_embs = torch.zeros([0, embeddings.shape[1]], device=device, dtype=dtype)
                
                if collected_node_members:
                    collected_node_members = torch.cat(collected_node_members, dim=0).to(dtype)
                    collected_node_member_embs = torch.cat(collected_node_member_embs, dim=0)
                else:
                    collected_node_members = torch.zeros([0, pts_dim], device=device, dtype=dtype)
                    collected_node_member_embs = torch.zeros([0, embeddings.shape[1]], device=device, dtype=dtype)

                if collected_way_members:
                    collected_way_members = torch.cat(collected_way_members, dim=0).to(dtype)
                    collected_way_member_embs = torch.cat(collected_way_member_embs, dim=0)
                else:
                    collected_way_members = torch.zeros([0, pts_dim], device=device, dtype=dtype)
                    collected_way_member_embs = torch.zeros([0, embeddings.shape[1]], device=device, dtype=dtype)

                map_feat = torch.cat([map_feat, rel_pts, collected_node_members, collected_way_members], dim=0)

                if self.positional_encoding_type == 'learned':
                    raise RuntimeError("not yet implemented")
                elif self.positional_encoding_type == 'custom':
                    map_feat = self.pos_encoder(map_feat.unsqueeze(0)).squeeze()

                embeddings_query = torch.cat([embeddings[:num_nodes], torch.repeat_interleave(embeddings[num_nodes:num_nodes+num_ways], num_way_pts, dim=0),  rel_embs, collected_node_member_embs, collected_way_member_embs], dim=0)
                map_feat = torch.cat([map_feat, embeddings_query], dim=1)

            else:
                if self.positional_encoding_type == 'learned':
                    raise RuntimeError("not yet implemented")
                elif self.positional_encoding_type == 'custom':
                    map_feat = self.pos_encoder(map_feat.unsqueeze(0)).squeeze()

                map_feat = F.pad(map_feat, (0,0,0,total_numel - map_feat.shape[0]), 'constant', 0)
                embeddings_query = torch.cat([embeddings[:num_nodes], torch.repeat_interleave(embeddings[num_nodes:num_nodes+num_ways], num_way_pts, dim=0),  embeddings[num_nodes+num_ways:]], dim=0)
                
                # import pdb;pdb.set_trace()

                map_feat = torch.cat([map_feat, embeddings_query], dim=1)
            
            if self.render_bev_feats and not self.use_queries_for_bev:
                bev_feat = torch.zeros([self.bev_h, self.bev_w, embeddings.shape[1]], device=osm_map_data['osm_map_ways_pts'][b_idx].device)
                if num_ways:
                    ways_normalized = normalize_2d_pts(osm_map_data['osm_map_ways_pts'][b_idx][..., 0:2], self.pc_range)
                    bev_feat = draw_ways(bev_feat, ways_normalized, embeddings[num_nodes:num_nodes+num_ways])
                if num_nodes:
                    nodes_normalized = normalize_2d_pts(osm_map_data['osm_map_nodes_pts'][b_idx][..., 0:2], self.pc_range)
                    bev_feat = draw_points(bev_feat, nodes_normalized, embeddings[:num_nodes])
                bev_feats.append(bev_feat)
            
            

            if self.use_orf_graph_ident:
                map_feat = self.add_orf_identifiers(map_feat, osm_map_data, b_idx)

            # if map_feat.dtype is torch.float64:
            #     import pdb;pdb.set_trace()
            
            map_features.append(map_feat)

        if bev_feats:
            return nn.utils.rnn.pad_sequence(map_features, batch_first=True, padding_value=0).to(dtype), torch.stack(bev_feats).to(dtype)
        else:
            return nn.utils.rnn.pad_sequence(map_features, batch_first=True, padding_value=0).to(dtype), None
    
    
    def forward(self, osm_map_data):

        map_features, bev_feats = self.build_map_features(osm_map_data)

        map_features = self.map_embedding(map_features.to(torch.float32))  
        
        map_features = self.transformer_encoder(map_features) 
        
        # import pdb;pdb.set_trace()

        if self.render_bev_feats and self.use_queries_for_bev:

            bev_feats_node = []
            bev_feats_way = []
            bev_feats_rel = []
            B = len(osm_map_data['osm_map_ways_pts'])

            for b_idx in range(0, B):
                
                if not osm_map_data['osm_map_nodes_pts'][b_idx].numel():
                    num_nodes = 0
                elif len(osm_map_data['osm_map_nodes_pts'][b_idx].shape) == 1:
                    osm_map_data['osm_map_nodes_pts'][b_idx] = osm_map_data['osm_map_nodes_pts'][b_idx].unsqueeze(0)
                    num_nodes = len(osm_map_data['osm_map_nodes_pts'][b_idx])
                else:
                    num_nodes = len(osm_map_data['osm_map_nodes_pts'][b_idx])
                
                num_ways, num_way_pts, pts_dim = osm_map_data['osm_map_ways_pts'][b_idx].shape
                num_rels = len(osm_map_data['osm_map_relations_tags_input_ids'][b_idx])

                if num_ways:
                    bev_feat = torch.zeros([self.bev_h, self.bev_w, map_features.shape[2]], device=osm_map_data['osm_map_ways_pts'][b_idx].device)
                    ways_normalized = normalize_2d_pts(osm_map_data['osm_map_ways_pts'][b_idx][..., 0:2], self.pc_range)
                    bev_feat = draw_ways(bev_feat, ways_normalized, map_features[b_idx, num_nodes:num_nodes+(num_ways*num_way_pts)])
                    bev_feats_node.append(bev_feat)
                else:
                    bev_feats_node.append(torch.zeros([self.bev_h, self.bev_w, map_features.shape[2]], device=osm_map_data['osm_map_ways_pts'][b_idx].device))
                if num_nodes:
                    bev_feat = torch.zeros([self.bev_h, self.bev_w, map_features.shape[2]], device=osm_map_data['osm_map_nodes_pts'][b_idx].device)
                    nodes_normalized = normalize_2d_pts(osm_map_data['osm_map_nodes_pts'][b_idx][..., 0:2], self.pc_range)
                    bev_feat = draw_points(bev_feat, nodes_normalized, map_features[b_idx, :num_nodes])
                    bev_feats_way.append(bev_feat)
                else:
                    bev_feats_way.append(torch.zeros([self.bev_h, self.bev_w, map_features.shape[2]], device=osm_map_data['osm_map_ways_pts'][b_idx].device))
                if num_rels:
                    rel_pts = []
                    rel_feats = []
                    for i in range(0, num_rels):
                        for member in osm_map_data['osm_map_relations_node_member_indices'][b_idx][i]:
                            if member.numel():
                                rel_feats.append(map_features[b_idx, num_nodes + num_ways + i].unsqueeze(0))
                                rel_pts.append(osm_map_data['osm_map_nodes_pts'][b_idx][member.to(torch.long)].unsqueeze(0))
                        for member in osm_map_data['osm_map_relations_way_member_indices'][b_idx][i]:
                            if member.numel():
                                rel_feats.append(map_features[b_idx, num_nodes + num_ways + i].unsqueeze(0).repeat((int(num_way_pts), 1)))
                                rel_pts.append(osm_map_data['osm_map_ways_pts'][b_idx][member.to(torch.long)])

                    bev_feat = torch.zeros([self.bev_h, self.bev_w, map_features.shape[2]], device=osm_map_data['osm_map_ways_pts'][b_idx].device)
                    
                    # import pdb;pdb.set_trace()
                    
                    rel_pts_normalized = normalize_2d_pts(torch.cat(rel_pts)[..., 0:2], self.pc_range)
                    bev_feat = draw_points(bev_feat, rel_pts_normalized, torch.cat(rel_feats))
                    bev_feats_rel.append(bev_feat)
                else:
                    bev_feats_rel.append(torch.zeros([self.bev_h, self.bev_w, map_features.shape[2]], device=osm_map_data['osm_map_ways_pts'][b_idx].device))

            
            bev_feats_node = torch.stack(bev_feats_node, dim=0)
            bev_feats_way = torch.stack(bev_feats_way, dim=0)
            bev_feats_rel = torch.stack(bev_feats_rel, dim=0)

            bev_feats = torch.cat([bev_feats_node, bev_feats_way, bev_feats_rel], dim=3)

        if bev_feats is not None:
            bev_feats = self.bev_feat_enc(bev_feats.permute(0,3,1,2)).permute(0,3,1,2)

        return map_features, bev_feats
        
