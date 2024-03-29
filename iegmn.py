import logging
import math
import os
from datetime import datetime

import dgl
import torch
from torch import nn
from dgl import function as fn

from embedding import AtomEncoder, A_feature_dims
from logger import log
from model_utils import *

class IEGMN_Layer(nn.Module):
    def __init__(
            self,
            orig_invar_feats_dim_h, #orig_h_feats_dim,
            invar_feats_dim_h, #h_feats_dim,  # in dim of h
            out_feats_dim_h, #out_feats_dim,  # out dim of h
            A_input_edge_feats_dim, #lig_input_edge_feats_dim,
            B_input_edge_feats_dim, #rec_input_edge_feats_dim,
            nonlin,
            cross_msgs, #boolean
            layer_norm,
            layer_norm_coords,
            final_h_layer_norm,
            use_dist_in_layers,
            skip_weight_h,
            x_connection_init,
            leakyrelu_neg_slope,
            debug,
            device,
            dropout,
            save_trajectories=False,
            # B_square_distance_scale=1,
            standard_norm_order=True,
            normalize_coordinate_update=False,
            A_evolve = True,# lig_evolve=True,
            B_evolve = True, #rec_evolve=True,
            fine_tune=False,
            geometry_regularization=False,
            pre_crossmsg_norm_type=0,
            post_crossmsg_norm_type=0,
            norm_cross_coords_update= False,
            loss_geometry_regularization = True,
            geom_reg_steps= 1,
            geometry_reg_step_size=0.1,
            weight_sharing=True,
    ):

        super(IEGMN_Layer, self).__init__()
        
        self.fine_tune = fine_tune
        self.cross_msgs = cross_msgs
        self.normalize_coordinate_update = normalize_coordinate_update
        self.final_h_layer_norm = final_h_layer_norm
        self.use_dist_in_layers = use_dist_in_layers
        self.skip_weight_h = skip_weight_h
        self.x_connection_init = x_connection_init
        # self.B_square_distance_scale = B_square_distance_scale
        self.geometry_reg_step_size = geometry_reg_step_size
        self.norm_cross_coords_update =norm_cross_coords_update
        self.loss_geometry_regularization = loss_geometry_regularization

        self.debug = debug
        self.device = device
        self.A_evolve = A_evolve
        self.B_evolve = B_evolve
        self.invar_feats_dim_h = invar_feats_dim_h
        self.out_feats_dim_h = out_feats_dim_h
        self.standard_norm_order = standard_norm_order
        self.pre_crossmsg_norm_type = pre_crossmsg_norm_type
        self.post_crossmsg_norm_type = post_crossmsg_norm_type
        self.all_sigmas_dist = [1.5 ** x for x in range(15)]
        self.geometry_regularization = geometry_regularization
        self.geom_reg_steps = geom_reg_steps
        self.save_trajectories = save_trajectories
        self.weight_sharing = weight_sharing

        # EDGES
        A_edge_mlp_input_dim = (invar_feats_dim_h * 2) + A_input_edge_feats_dim
        if self.use_dist_in_layers and self.A_evolve: #TRUE and TRUE
            A_edge_mlp_input_dim += len(self.all_sigmas_dist)
        # if self.standard_norm_order: # TRUE
        self.A_edge_mlp = nn.Sequential(
            nn.Linear(A_edge_mlp_input_dim, self.out_feats_dim_h),
            get_layer_norm(layer_norm, self.out_feats_dim_h),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            nn.Dropout(dropout),
            nn.Linear(self.out_feats_dim_h, self.out_feats_dim_h),
            get_layer_norm(layer_norm, self.out_feats_dim_h),
        )
        if self.weight_sharing:
            self.B_edge_mlp = self.A_edge_mlp
        else:
            B_edge_mlp_input_dim = (invar_feats_dim_h * 2) + B_input_edge_feats_dim
            if self.use_dist_in_layers and self.B_evolve:
                B_edge_mlp_input_dim += len(self.all_sigmas_dist)
            # if self.standard_norm_order:
            self.B_edge_mlp = nn.Sequential(
                nn.Linear(B_edge_mlp_input_dim, self.out_feats_dim_h),
                get_layer_norm(layer_norm, self.out_feats_dim_h),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(self.out_feats_dim_h, self.out_feats_dim_h),
                get_layer_norm(layer_norm, self.out_feats_dim_h),
            )

        # NODES
        self.node_norm = nn.Identity()  # nn.LayerNorm(invar_feats_dim_h)

        if self.normalize_coordinate_update: # True
            self.A_coords_norm = CoordsNorm(scale_init=1e-2)
            if self.weight_sharing:
                self.B_coords_norm = self.A_coords_norm
            else:
                self.B_coords_norm = CoordsNorm(scale_init=1e-2)

        self.att_mlp_Q_A = nn.Sequential(
            nn.Linear(invar_feats_dim_h, invar_feats_dim_h, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K_A = nn.Sequential(
            nn.Linear(invar_feats_dim_h, invar_feats_dim_h, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V_A = nn.Sequential(
            nn.Linear(invar_feats_dim_h, invar_feats_dim_h, bias=False),
        )
        if self.weight_sharing:
            self.att_mlp_Q_B = self.att_mlp_Q_A
            self.att_mlp_K_B = self.att_mlp_K_A
            self.att_mlp_V_B = self.att_mlp_V_A
        else:
            self.att_mlp_Q_B = nn.Sequential(
                nn.Linear(invar_feats_dim_h, invar_feats_dim_h, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_K_B = nn.Sequential(
                nn.Linear(invar_feats_dim_h, invar_feats_dim_h, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_V_B = nn.Sequential(
                nn.Linear(invar_feats_dim_h, invar_feats_dim_h, bias=False),
            )
        if self.standard_norm_order:
            self.node_mlp_A = nn.Sequential(
                nn.Linear(orig_invar_feats_dim_h + 2 * invar_feats_dim_h + self.out_feats_dim_h, invar_feats_dim_h),
                get_layer_norm(layer_norm, invar_feats_dim_h),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(invar_feats_dim_h, out_feats_dim_h),
                get_layer_norm(layer_norm, out_feats_dim_h),
            )
            if self.weight_sharing:
                self.node_mlp_B = self.node_mlp_A
            else:
                self.node_mlp_B = nn.Sequential(
                    nn.Linear(orig_invar_feats_dim_h + 2 * invar_feats_dim_h + self.out_feats_dim_h, invar_feats_dim_h),
                    get_layer_norm(layer_norm, invar_feats_dim_h),
                    get_non_lin(nonlin, leakyrelu_neg_slope),
                    nn.Dropout(dropout),
                    nn.Linear(invar_feats_dim_h, out_feats_dim_h),
                    get_layer_norm(layer_norm, out_feats_dim_h),
                )

        self.final_h_layernorm_layer_A = get_norm(self.final_h_layer_norm, out_feats_dim_h)
        self.pre_crossmsg_norm_A = get_norm(self.pre_crossmsg_norm_type, invar_feats_dim_h)
        self.post_crossmsg_norm_A = get_norm(self.post_crossmsg_norm_type, invar_feats_dim_h)
        
        if self.weight_sharing:
            self.final_h_layernorm_layer_B = self.final_h_layernorm_layer_A
            self.pre_crossmsg_norm_B = self.pre_crossmsg_norm_A
            self.post_crossmsg_norm_B = self.post_crossmsg_norm_A
        else:
            self.final_h_layernorm_layer_B = get_norm(self.final_h_layer_norm, out_feats_dim_h)
            self.pre_crossmsg_norm_B = get_norm(self.pre_crossmsg_norm_type, invar_feats_dim_h)
            self.post_crossmsg_norm_B = get_norm(self.post_crossmsg_norm_type, invar_feats_dim_h)

        # if self.standard_norm_order:
        self.coords_mlp_A = nn.Sequential(
            nn.Linear(self.out_feats_dim_h, self.out_feats_dim_h),
            get_layer_norm(layer_norm_coords, self.out_feats_dim_h),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            nn.Dropout(dropout),
            nn.Linear(self.out_feats_dim_h, 1)
        )
        if self.weight_sharing:
            self.coords_mlp_B = self.coords_mlp_A
        else:
            # if self.standard_norm_order:
            self.coords_mlp_B = nn.Sequential(
                nn.Linear(self.out_feats_dim_h, self.out_feats_dim_h),
                get_layer_norm(layer_norm_coords, self.out_feats_dim_h),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(self.out_feats_dim_h, 1)
            )

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def apply_edges_A(self, edges):
        if self.use_dist_in_layers and self.A_evolve:
            x_rel_mag = edges.data['x_rel'] ** 2
            # print(x_rel_mag.device, edges.src['feat'].device, edges.dst['feat'].device, edges.data['feat'].device)
            x_rel_mag = torch.sum(x_rel_mag, dim=1, keepdim=True)
            x_rel_mag = torch.cat([torch.exp(-x_rel_mag / sigma) for sigma in self.all_sigmas_dist], dim=-1)
            return {'msg': self.A_edge_mlp(
                torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], x_rel_mag], dim=1))} # operates with edge features in it and node features
        else:
            return {
                'msg': self.A_edge_mlp(torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1))}

    def apply_edges_B(self, edges):
        if self.use_dist_in_layers and self.B_evolve:
            x_rel_mag = edges.data['x_rel'] ** 2
            x_rel_mag = torch.sum(x_rel_mag, dim=1, keepdim=True)
            x_rel_mag = torch.cat([torch.exp(-x_rel_mag / sigma) for sigma in self.all_sigmas_dist], dim=-1)
            return {'msg': self.B_edge_mlp(
                torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], x_rel_mag], dim=1))}
        else:
            return {
                'msg': self.B_edge_mlp(torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1))}

    def update_x_moment_A(self, edges):
        edge_coef_A = self.coords_mlp_A(edges.data['msg'])  # \phi^x(m_{i->j})
        x_rel = self.A_coords_norm(edges.data['x_rel']) if self.normalize_coordinate_update else edges.data['x_rel']
        return {'m': x_rel * edge_coef_A}  # (x_i - x_j) * \phi^x(m_{i->j})

    def update_x_moment_B(self, edges):
        edge_coef_B = self.coords_mlp_B(edges.data['msg'])  # \phi^x(m_{i->j})
        x_rel = self.B_coords_norm(edges.data['x_rel']) if self.normalize_coordinate_update else edges.data['x_rel']
        return {'m': x_rel * edge_coef_B}  # (x_i - x_j) * \phi^x(m_{i->j})

    def attention_coefficients(self, edges):  # for when using cross edges (but this is super slow so dont do it)
        return {'attention_coefficient': torch.sum(edges.dst['q'] * edges.src['k'], dim=1), 'values': edges.src['v']}

    def attention_aggregation(self, nodes):  # for when using cross edges (but this is super slow so dont do it)
        attention = torch.softmax(nodes.mailbox['attention_coefficient'], dim=1)
        return {'cross_attention_feat': torch.sum(attention[:, :, None] * nodes.mailbox['values'], dim=1)}

    def forward(self, A_graph, B_graph, coords_A, h_feats_A, original_A_node_features, orig_coords_A,
                coords_B, h_feats_B, original_B_node_features, orig_coords_B, mask, geometry_graph_A, geometry_graph_B):
        with A_graph.local_scope() and B_graph.local_scope():
            A_graph.ndata['x_now'] = coords_A
            B_graph.ndata['x_now'] = coords_B
            A_graph.ndata['feat'] = h_feats_A  # first time set here
            B_graph.ndata['feat'] = h_feats_B

            if self.debug:
                log(torch.max(A_graph.ndata['x_now'].abs()), 'x_now : x_i at layer entrance')
                log(torch.max(A_graph.ndata['feat'].abs()), 'data[feat] = h_i at layer entrance')

            if self.A_evolve:
                A_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'))  # x_i - x_j
                if self.debug:
                    log(torch.max(A_graph.edata['x_rel'].abs()), 'x_rel : x_i - x_j')
            if self.B_evolve:
                B_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'))

            A_graph.apply_edges(self.apply_edges_A)  ## i->j edge:  [h_i h_j] phi^e edge_mlp
            B_graph.apply_edges(self.apply_edges_B) #apply_edges_rec)
            # Equation 1 message passing to create 'msg'

            if self.debug:
                log(torch.max(A_graph.edata['msg'].abs()),
                    'data[msg] = m_{i->j} = phi^e(h_i, h_j, f_{i,j}, x_rel_mag_ligand)')

            h_feats_A_norm = apply_norm(A_graph, h_feats_A, self.final_h_layer_norm, self.final_h_layernorm_layer_A)
            h_feats_B_norm = apply_norm(B_graph, h_feats_B, self.final_h_layer_norm, self.final_h_layernorm_layer_B)
            
            cross_attention_A_feat = cross_attention(self.att_mlp_Q_A(h_feats_A_norm),
                                                       self.att_mlp_K_B(h_feats_B_norm),
                                                       self.att_mlp_V_B(h_feats_B_norm), mask, self.cross_msgs)
            cross_attention_B_feat = cross_attention(self.att_mlp_Q_B(h_feats_B_norm),
                                                       self.att_mlp_K_A(h_feats_A_norm),
                                                       self.att_mlp_V_A(h_feats_A_norm), mask.transpose(0, 1),
                                                       self.cross_msgs)
            cross_attention_A_feat = apply_norm(A_graph, cross_attention_A_feat, self.final_h_layer_norm, self.final_h_layernorm_layer_A)
            cross_attention_B_feat = apply_norm(B_graph, cross_attention_B_feat, self.final_h_layer_norm, self.final_h_layernorm_layer_B)
            # Equation 2: mu terms for all to all attention

            if self.debug:
                log(torch.max(cross_attention_A_feat.abs()), 'aggr_cross_msg(i) = sum_j a_{i,j} * h_j')

            # Equation 3: coordinate update
            if self.A_evolve:
                A_graph.update_all(self.update_x_moment_A, fn.mean('m', 'x_update')) # phi_x coord_mlp
                # Inspired by https://arxiv.org/pdf/2108.10521.pdf we use original X and not only graph.ndata['x_now']
                x_evolved_A = self.x_connection_init * orig_coords_A + (1. - self.x_connection_init) * A_graph.ndata['x_now'] + A_graph.ndata['x_update']
            else:
                x_evolved_A = coords_A

            if self.B_evolve:
                B_graph.update_all(self.update_x_moment_B, fn.mean('m', 'x_update'))
                x_evolved_B = self.x_connection_init * orig_coords_B + (1. - self.x_connection_init) * \
                                B_graph.ndata['x_now'] + B_graph.ndata['x_update']
            else:
                x_evolved_B = coords_B

            # Equation 4: Aggregate messages
            A_graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))#copy_edge
            B_graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))

            # if self.fine_tune:
            #     x_evolved_A = x_evolved_A + self.att_mlp_cross_coors_V_lig(h_feats_A) * (
            #             self.lig_cross_coords_norm(A_graph.ndata['x_now'] - cross_attention(self.att_mlp_cross_coors_Q_lig(h_feats_A),
            #                                                        self.att_mlp_cross_coors_K(h_feats_B),
            #                                                        B_graph.ndata['x_now'], mask, self.cross_msgs)))
            # if self.fine_tune:
            #     x_evolved_B = x_evolved_B + self.att_mlp_cross_coors_V(h_feats_B) * (
            #             self.rec_cross_coords_norm(B_graph.ndata['x_now'] - cross_attention(self.att_mlp_cross_coors_Q(h_feats_B),
            #                                                        self.att_mlp_cross_coors_K_lig(h_feats_A),
            #                                                        A_graph.ndata['x_now'], mask.transpose(0, 1),
            #                                                        self.cross_msgs)))
            trajectory = []
            if self.save_trajectories: trajectory.append(x_evolved_A.detach().cpu())
            if self.loss_geometry_regularization:
                src, dst = geometry_graph_A.edges()
                src = src.long()
                dst = dst.long()
                d_squared = torch.sum((x_evolved_A[src] - x_evolved_A[dst]) ** 2, dim=1)
                geom_loss = torch.sum((d_squared - geometry_graph_A.edata['feat'] ** 2) ** 2)

                src, dst = geometry_graph_B.edges()
                src = src.long()
                dst = dst.long()
                d_squared = torch.sum((x_evolved_B[src] - x_evolved_B[dst]) ** 2, dim=1)
                geom_loss += torch.sum((d_squared - geometry_graph_B.edata['feat'] ** 2) ** 2)
            else:
                geom_loss = 0
            if self.geometry_regularization:
                src, dst = geometry_graph_A.edges()
                src = src.long()
                dst = dst.long()
                for step in range(self.geom_reg_steps):
                    d_squared = torch.sum((x_evolved_A[src] - x_evolved_A[dst]) ** 2, dim=1)
                    Loss = torch.sum((d_squared - geometry_graph_A.edata['feat'] ** 2)**2) # this is the loss whose gradient we are calculating here
                    grad_d_squared = 2 * (x_evolved_A[src] - x_evolved_A[dst])
                    geometry_graph_A.edata['partial_grads'] = 2 * (d_squared - geometry_graph_A.edata['feat'] ** 2)[:,None] * grad_d_squared
                    geometry_graph_A.update_all(fn.copy_e('partial_grads', 'partial_grads_msg'),
                                              fn.sum('partial_grads_msg', 'grad_x_evolved'))
                    grad_x_evolved = geometry_graph_A.ndata['grad_x_evolved']
                    x_evolved_A = x_evolved_A + self.geometry_reg_step_size * grad_x_evolved
                    if self.save_trajectories:
                        trajectory.append(x_evolved_A.detach().cpu())

                src, dst = geometry_graph_B.edges()
                src = src.long()
                dst = dst.long()
                for step in range(self.geom_reg_steps):
                    d_squared = torch.sum((x_evolved_B[src] - x_evolved_B[dst]) ** 2, dim=1)
                    Loss = torch.sum((d_squared - geometry_graph_B.edata['feat'] ** 2)**2) # this is the loss whose gradient we are calculating here
                    grad_d_squared = 2 * (x_evolved_B[src] - x_evolved_B[dst])
                    geometry_graph_B.edata['partial_grads'] = 2 * (d_squared - geometry_graph_B.edata['feat'] ** 2)[:,None] * grad_d_squared
                    geometry_graph_B.update_all(fn.copy_e('partial_grads', 'partial_grads_msg'),fn.sum('partial_grads_msg', 'grad_x_evolved'))
                    grad_x_evolved = geometry_graph_B.ndata['grad_x_evolved']
                    x_evolved_B = x_evolved_B + self.geometry_reg_step_size * grad_x_evolved



            if self.debug:
                log(torch.max(A_graph.ndata['aggr_msg'].abs()), 'data[aggr_msg]: \sum_j m_{i->j} ')
                if self.A_evolve:
                    log(torch.max(A_graph.ndata['x_update'].abs()),
                        'data[x_update] : \sum_j (x_i - x_j) * \phi^x(m_{i->j})')
                    log(torch.max(x_evolved_A.abs()), 'x_i new = x_evolved_A : x_i + data[x_update]')

            input_node_upd_A = torch.cat((self.node_norm(A_graph.ndata['feat']),
                                               A_graph.ndata['aggr_msg'],
                                               cross_attention_A_feat,
                                               original_A_node_features), dim=-1)

            input_node_upd_B = torch.cat((self.node_norm(B_graph.ndata['feat']),
                                                 B_graph.ndata['aggr_msg'],
                                                 cross_attention_B_feat,
                                                 original_B_node_features), dim=-1)

            # Skip connections
            # Equation 5: node updates --> cross attention is mu
            if self.invar_feats_dim_h == self.out_feats_dim_h: #phi^h
                node_upd_A = self.skip_weight_h * self.node_mlp_A(input_node_upd_A) + (1. - self.skip_weight_h) * h_feats_A
                node_upd_B = self.skip_weight_h * self.node_mlp_B(input_node_upd_B) + (1. - self.skip_weight_h) * h_feats_B
            else:
                node_upd_A = self.node_mlp_A(input_node_upd_A) # phi^h
                node_upd_B = self.node_mlp_B(input_node_upd_B)

            if self.debug:
                log('node_mlp params')
                for p in self.node_mlp_B.parameters():
                    log(torch.max(p.abs()), 'max node_mlp_params')
                    log(torch.min(p.abs()), 'min of abs node_mlp_params')
                log(torch.max(input_node_upd_A.abs()), 'concat(h_i, aggr_msg, aggr_cross_msg)')
                log(torch.max(node_upd_A), 'h_i new = h_i + MLP(h_i, aggr_msg, aggr_cross_msg)')

            node_upd_A = apply_norm(A_graph, node_upd_A, self.final_h_layer_norm, self.final_h_layernorm_layer_A)
            node_upd_B = apply_norm(B_graph, node_upd_B, self.final_h_layer_norm, self.final_h_layernorm_layer_B)
            return x_evolved_A, node_upd_A, x_evolved_B, node_upd_B, trajectory, geom_loss

    def __repr__(self):
        return "IEGMN Custom Layer " + str(self.__dict__)


class IEGMN(nn.Module):
    # def __init__(self, n_lays, debug, device, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
    #              use_edge_features_in_gmn, use_mean_node_features, atom_emb_dim, iegmn_lay_hid_dim, num_att_heads,
    #              dropout, nonlin, leakyrelu_neg_slope, random_vec_dim=0, random_vec_std=1, use_scalar_features=True,
    #              num_A_feats=None, move_keypts_back=False, normalize_Z_lig_directions=False,
    #              unnormalized_kpt_weights=False, centroid_keypts_construction_rec=False,
    #              centroid_keypts_construction_lig=False, B_no_softmax=False, A_no_softmax=False,
    #              normalize_Z_rec_directions=False,
    #              centroid_keypts_construction=False, evolve_only=True, separate_lig=False, save_trajectories=False, **kwargs):
    def __init__(self, n_lays, debug, device, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
                 use_edge_features_in_gmn, use_mean_node_features, atom_emb_dim, iegmn_lay_hid_dim,
                 dropout, nonlin, leakyrelu_neg_slope, random_vec_dim=0, random_vec_std=1, use_scalar_features=True,
                 num_A_feats=None, B_no_softmax=False, A_no_softmax=False,
                 evolve_only=True, separate_lig=False, save_trajectories=False,
                 weight_sharing = True, conditional_mask=False, **kwargs):
        super(IEGMN, self).__init__()
        self.debug = debug
        self.cross_msgs = cross_msgs
        self.device = device
        self.save_trajectories = save_trajectories
        # self.unnormalized_kpt_weights = unnormalized_kpt_weights
        self.separate_lig =separate_lig
        self.noise_decay_rate = noise_decay_rate
        self.noise_initial = noise_initial
        self.use_edge_features_in_gmn = use_edge_features_in_gmn
        self.use_mean_node_features = use_mean_node_features
        self.random_vec_dim = random_vec_dim
        self.random_vec_std = random_vec_std
        # self.move_keypts_back = move_keypts_back
        # self.normalize_Z_lig_directions = normalize_Z_lig_directions
        # self.centroid_keypts_construction = centroid_keypts_construction
        # self.centroid_keypts_construction_rec = centroid_keypts_construction_rec
        # self.centroid_keypts_construction_lig = centroid_keypts_construction_lig
        # self.normalize_Z_rec_directions = normalize_Z_rec_directions
        self.B_no_softmax = B_no_softmax
        self.A_no_softmax = A_no_softmax
        self.evolve_only = evolve_only
        self.weight_sharing = weight_sharing
        self.conditional_mask = conditional_mask

        self.atom_embedder = AtomEncoder(emb_dim=atom_emb_dim - self.random_vec_dim,
                                             feature_dims=A_feature_dims, use_scalar_feat=use_scalar_features,
                                             n_feats_to_use=num_A_feats)

        input_node_feats_dim = atom_emb_dim #64
        if self.use_mean_node_features:
            input_node_feats_dim += 5  ### Additional features from mu_r_norm
        self.iegmn_layers = nn.ModuleList()
        # Create First Layer
        self.iegmn_layers.append(
            IEGMN_Layer(orig_invar_feats_dim_h=input_node_feats_dim,
                        invar_feats_dim_h=input_node_feats_dim,
                        out_feats_dim_h=iegmn_lay_hid_dim,
                        nonlin=nonlin,
                        cross_msgs=self.cross_msgs,
                        leakyrelu_neg_slope=leakyrelu_neg_slope,
                        debug=debug,
                        device=device,
                        dropout=dropout,
                        save_trajectories=save_trajectories,
                        weight_sharing = self.weight_sharing, **kwargs))

        # Create N - 1 layers
        if shared_layers:
            interm_lay = IEGMN_Layer(orig_invar_feats_dim_h=input_node_feats_dim,
                                     invar_feats_dim_h=iegmn_lay_hid_dim,
                                     out_feats_dim_h=iegmn_lay_hid_dim,
                                     cross_msgs=self.cross_msgs,
                                     nonlin=nonlin,
                                     leakyrelu_neg_slope=leakyrelu_neg_slope,
                                     debug=debug,
                                     device=device,
                                     dropout=dropout,
                                     save_trajectories=save_trajectories,
                                     weight_sharing = self.weight_sharing, **kwargs)
            for layer_idx in range(1, n_lays):
                self.iegmn_layers.append(interm_lay)
        else:
            for layer_idx in range(1, n_lays):
                debug_this_layer = debug if n_lays - 1 == layer_idx else False
                self.iegmn_layers.append(
                    IEGMN_Layer(orig_invar_feats_dim_h=input_node_feats_dim,
                                invar_feats_dim_h=iegmn_lay_hid_dim,
                                out_feats_dim_h=iegmn_lay_hid_dim,
                                cross_msgs=self.cross_msgs,
                                nonlin=nonlin,
                                leakyrelu_neg_slope=leakyrelu_neg_slope,
                                debug=debug_this_layer,
                                device=device,
                                dropout=dropout,
                                save_trajectories=save_trajectories,
                                weight_sharing = self.weight_sharing, **kwargs))
        # if self.separate_lig:
        #     self.iegmn_layers_separate = nn.ModuleList()
        #     self.iegmn_layers_separate.append(
        #         IEGMN_Layer(orig_invar_feats_dim_h=input_node_feats_dim,
        #                     invar_feats_dim_h=input_node_feats_dim,
        #                     out_feats_dim_h=iegmn_lay_hid_dim,
        #                     nonlin=nonlin,
        #                     cross_msgs=self.cross_msgs,
        #                     leakyrelu_neg_slope=leakyrelu_neg_slope,
        #                     debug=debug,
        #                     device=device,
        #                     dropout=dropout,
        #                     save_trajectories=save_trajectories,**kwargs))

            # if shared_layers:
            #     interm_lay = IEGMN_Layer(orig_invar_feats_dim_h=input_node_feats_dim,
            #                              invar_feats_dim_h=iegmn_lay_hid_dim,
            #                              out_feats_dim_h=iegmn_lay_hid_dim,
            #                              cross_msgs=self.cross_msgs,
            #                              nonlin=nonlin,
            #                              leakyrelu_neg_slope=leakyrelu_neg_slope,
            #                              debug=debug,
            #                              device=device,
            #                              dropout=dropout,
            #                              save_trajectories=save_trajectories,**kwargs)
            #     for layer_idx in range(1, n_lays):
            #         self.iegmn_layers_separate.append(interm_lay)
            # else:
            #     for layer_idx in range(1, n_lays):
            #         debug_this_layer = debug if n_lays - 1 == layer_idx else False
            #         self.iegmn_layers_separate.append(
            #             IEGMN_Layer(orig_invar_feats_dim_h=input_node_feats_dim,
            #                         invar_feats_dim_h=iegmn_lay_hid_dim,
            #                         out_feats_dim_h=iegmn_lay_hid_dim,
            #                         cross_msgs=self.cross_msgs,
            #                         nonlin=nonlin,
            #                         leakyrelu_neg_slope=leakyrelu_neg_slope,
            #                         debug=debug_this_layer,
            #                         device=device,
            #                         dropout=dropout,
            #                         save_trajectories=save_trajectories,**kwargs))
        # Attention layers
        # self.num_att_heads = num_att_heads
        # self.out_feats_dim_h = iegmn_lay_hid_dim
        # self.keypts_attention_lig = nn.Sequential(
        #     nn.Linear(self.out_feats_dim_h, self.num_att_heads * self.out_feats_dim_h, bias=False))
        # self.keypts_queries_lig = nn.Sequential(
        #     nn.Linear(self.out_feats_dim_h, self.num_att_heads * self.out_feats_dim_h, bias=False))
        # self.keypts_attention_rec = nn.Sequential(
        #     nn.Linear(self.out_feats_dim_h, self.num_att_heads * self.out_feats_dim_h, bias=False))
        # self.keypts_queries_rec = nn.Sequential(
        #     nn.Linear(self.out_feats_dim_h, self.num_att_heads * self.out_feats_dim_h, bias=False))

        # self.h_mean_A = nn.Sequential(
        #     nn.Linear(self.out_feats_dim_h, self.out_feats_dim_h),
        #     nn.Dropout(dropout),
        #     get_non_lin(nonlin, leakyrelu_neg_slope),
        # )
        # self.h_mean_B = nn.Sequential(
        #     nn.Linear(self.out_feats_dim_h, self.out_feats_dim_h),
        #     nn.Dropout(dropout),
        #     get_non_lin(nonlin, leakyrelu_neg_slope),
        # )

        # if self.unnormalized_kpt_weights:
        #     self.scale_lig = nn.Linear(self.out_feats_dim_h, 1)
        #     self.scale_rec = nn.Linear(self.out_feats_dim_h, 1)
        # self.reset_parameters()

        # if self.normalize_Z_lig_directions:
        #     self.Z_lig_dir_norm = CoordsNorm()
        # if self.normalize_Z_rec_directions:
        #     self.Z_rec_dir_norm = CoordsNorm()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, A_graph, B_graph, geometry_graph_A,geometry_graph_B, epoch, complex_names = "test"):
        orig_coords_A = A_graph.ndata['x'] #.to(self.device)
        orig_coords_B = B_graph.ndata['x'] #.to(self.device)

        coords_A = A_graph.ndata['x']
        coords_B = B_graph.ndata['x']

        h_feats_A = self.atom_embedder(A_graph.ndata['feat'])
        h_feats_B = self.atom_embedder(B_graph.ndata['feat'])

        rand_dist = torch.distributions.normal.Normal(loc=0, scale=self.random_vec_std)
        rand_h_A = rand_dist.sample([h_feats_A.size(0), self.random_vec_dim]).to(self.device)
        rand_h_B = rand_dist.sample([h_feats_B.size(0), self.random_vec_dim]).to(self.device)
        h_feats_A = torch.cat([h_feats_A, rand_h_A], dim=1)
        h_feats_B = torch.cat([h_feats_B, rand_h_B], dim=1)

        if self.debug:
            log(torch.max(h_feats_A.abs()), 'max h_feats_A before layers and noise ')
            log(torch.max(h_feats_B.abs()), 'max h_feats_B before layers and noise ')

        # random noise:
        if self.noise_initial > 0:
            noise_level = self.noise_initial * self.noise_decay_rate ** (epoch + 1)
            h_feats_A = h_feats_A + noise_level * torch.randn_like(h_feats_A)
            h_feats_B = h_feats_B + noise_level * torch.randn_like(h_feats_B)
            coords_A = coords_A + noise_level * torch.randn_like(coords_A)
            coords_B = coords_B + noise_level * torch.randn_like(coords_B)

        if self.debug:
            log(torch.max(h_feats_A.abs()), 'h_feats_A before layers but after noise ')
            log(torch.max(h_feats_B.abs()), 'h_feats_B before layers but after noise ')

        if self.use_mean_node_features:
            h_feats_A = torch.cat([h_feats_A, torch.log(A_graph.ndata['mu_r_norm'])],
                                    dim=1)
            h_feats_B = torch.cat(
                [h_feats_B, torch.log(B_graph.ndata['mu_r_norm'])], dim=1)

        if self.debug:
            log(torch.max(h_feats_A.abs()), torch.norm(h_feats_A),
                'max and norm of h_feats_A before layers but after noise and mu_r_norm')
            log(torch.max(h_feats_B.abs()), torch.norm(h_feats_A),
                'max and norm of h_feats_B before layers but after noise and mu_r_norm')

        original_A_node_features = h_feats_A
        original_B_node_features = h_feats_B
        A_graph.edata['feat'] *= self.use_edge_features_in_gmn
        B_graph.edata['feat'] *= self.use_edge_features_in_gmn

        mask = None
        if self.cross_msgs:
            if self.conditional_mask:
                mask = get_conditional_mask()# TODO for decoder
            else:
                mask = get_mask(A_graph.batch_num_nodes(), B_graph.batch_num_nodes(), self.device)
        if self.separate_lig:
            coords_lig_separate =coords_A
            h_feats_lig_separate =h_feats_A
            coords_rec_separate =coords_B
            h_feats_rec_separate =h_feats_B
        full_trajectory = [coords_A.detach().cpu()]
        geom_losses = 0
        for i, layer in enumerate(self.iegmn_layers):
            if self.debug: log('layer ', i)
            coords_A, \
            h_feats_A, \
            coords_B, \
            h_feats_B, trajectory, geom_loss = layer(A_graph=A_graph,
                                B_graph=B_graph,
                                coords_A=coords_A,
                                h_feats_A=h_feats_A,
                                original_A_node_features=original_A_node_features,
                                orig_coords_A=orig_coords_A,
                                coords_B=coords_B,
                                h_feats_B=h_feats_B,
                                original_B_node_features=original_B_node_features,
                                orig_coords_B=orig_coords_B,
                                mask=mask,
                                geometry_graph_A=geometry_graph_A,
                                geometry_graph_B=geometry_graph_B,
                                )
            if not self.separate_lig:
                geom_losses = geom_losses + geom_loss
                full_trajectory.extend(trajectory)
        return coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory

    def __repr__(self):
        return "IEGMN " + str(self.__dict__)

# =================================================================================================================
