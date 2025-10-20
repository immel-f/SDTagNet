import torch
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      POSITIONAL_ENCODING,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, BaseTransformerLayer

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(MapTRDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                osm_bev_reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        intermediate_osm_bev_reference_points = []
        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            
            if osm_bev_reference_points is not None:
                osm_bev_reference_points_input = osm_bev_reference_points[..., :2].unsqueeze(
                    2)  # BS NUM_QUERY NUM_LEVEL 2
            else:
                osm_bev_reference_points_input = None
                
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                osm_bev_reference_points=osm_bev_reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                # assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points = tmp + inverse_sigmoid(reference_points)

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

                if osm_bev_reference_points is not None:
                    new_osm_bev_reference_points = torch.zeros_like(osm_bev_reference_points)
                    new_osm_bev_reference_points = tmp + inverse_sigmoid(osm_bev_reference_points)

                    new_osm_bev_reference_points = new_osm_bev_reference_points.sigmoid()

                    osm_bev_reference_points = new_osm_bev_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

                if osm_bev_reference_points is not None:
                    intermediate_osm_bev_reference_points.append(osm_bev_reference_points)

        if self.return_intermediate:
            if osm_bev_reference_points is not None:
                return torch.stack(intermediate), torch.stack(
                    intermediate_reference_points), torch.stack(intermediate_osm_bev_reference_points)
            else:
                return torch.stack(intermediate), torch.stack(intermediate_reference_points), None

        return output, reference_points, osm_bev_reference_points



@TRANSFORMER_LAYER.register_module()
class DecoupledDetrTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 num_vec=50,
                 num_pts_per_vec=20,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(DecoupledDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        # assert len(operation_order) == 10
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                osm_map_feats=None,
                osm_map_bev_feats=None,
                osm_bev_query_pos=None,
                osm_bev_reference_points=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.
        **kwargs contains some specific arguments of attentions.
        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        first_cross_attn = True
        second_cross_attn = True
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'
        # 
        num_vec = kwargs['num_vec']
        num_pts_per_vec = kwargs['num_pts_per_vec']
        for layer in self.operation_order:
            if layer == 'self_attn':
                # import ipdb;ipdb.set_trace()
                if attn_index == 0:
                    n_pts, n_batch, n_dim = query.shape
                    query = query.view(num_vec, num_pts_per_vec,n_batch,n_dim).flatten(1,2)
                    query_pos = query_pos.view(num_vec, num_pts_per_vec,n_batch,n_dim).flatten(1,2)
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=kwargs['self_attn_mask'],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                    # import ipdb;ipdb.set_trace()
                    query = query.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(0,1)
                    query_pos = query_pos.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(0,1)
                    attn_index += 1
                    identity = query
                else:
                    # import ipdb;ipdb.set_trace()
                    n_pts, n_batch, n_dim = query.shape
                    query = query.view(num_vec, num_pts_per_vec,n_batch,n_dim).permute(1,0,2,3).contiguous().flatten(1,2)
                    query_pos = query_pos.view(num_vec, num_pts_per_vec,n_batch,n_dim).permute(1,0,2,3).contiguous().flatten(1,2)
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                    # import ipdb;ipdb.set_trace()
                    query = query.view(num_pts_per_vec, num_vec, n_batch, n_dim).permute(1,0,2,3).contiguous().flatten(0,1)
                    query_pos = query_pos.view(num_pts_per_vec, num_vec, n_batch, n_dim).permute(1,0,2,3).contiguous().flatten(0,1)
                    attn_index += 1
                    identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn' and first_cross_attn and value is not None:

                # print("performing bev attention!")

                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query
                first_cross_attn = False

            # osm map cross attention
            elif layer == 'cross_attn' and osm_map_bev_feats is not None and not first_cross_attn and second_cross_attn:

                # import pdb;pdb.set_trace()
                # print("performing BEV sd map attention!")

                query = self.attentions[attn_index](
                    query,
                    key,
                    osm_map_bev_feats,
                    identity if self.pre_norm else None,
                    query_pos=osm_bev_query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    reference_points=osm_bev_reference_points,
                    spatial_shapes=kwargs['spatial_shapes'],
                    level_start_index=kwargs['level_start_index'])
                attn_index += 1
                identity = query

                second_cross_attn = False

            elif layer == 'cross_attn' and osm_map_feats is not None and (not first_cross_attn or value is None) and (not second_cross_attn or osm_map_bev_feats is None):

                # print("performing sd map attention!")
                
                query = self.attentions[attn_index](
                    query,
                    torch.permute(osm_map_feats, (1, 0, 2)),
                    torch.permute(osm_map_feats, (1, 0, 2)),
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

