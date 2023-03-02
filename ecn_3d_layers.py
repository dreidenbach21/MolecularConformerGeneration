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

class Pooling_3D_Layer(nn.Module):
    def __init__(
            self,
            # orig_invar_feats_dim_h, #orig_h_feats_dim,
            invar_feats_dim_h, #h_feats_dim,  # in dim of h
            out_feats_dim_h, #out_feats_dim,  # out dim of h
            A_input_edge_feats_dim, #lig_input_edge_feats_dim,
            B_input_edge_feats_dim, #rec_input_edge_feats_dim,
            nonlin,
            cross_msgs, #TODO boolean need for consistency until we switch to hydra
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
            # standard_norm_order=True,
            normalize_coordinate_update=False,
            # A_evolve = True,# lig_evolve=True,
            # B_evolve = True, #rec_evolve=True,
            # fine_tune=False,
            geometry_regularization=False,
            pre_crossmsg_norm_type=0,
            post_crossmsg_norm_type=0,
            norm_cross_coords_update= False,
            loss_geometry_regularization = True,
            geom_reg_steps= 1,
            geometry_reg_step_size=0.1,
            weight_sharing=True,
            # A_input_edge_feats_dim = None,
            # B_input_edge_feats_dim = None
    ):

        super(Pooling_3D_Layer, self).__init__()
        self.normalize_coordinate_update = normalize_coordinate_update
        self.final_h_layer_norm = final_h_layer_norm
        self.use_dist_in_layers = use_dist_in_layers
        self.skip_weight_h = skip_weight_h
        self.x_connection_init = x_connection_init
        self.norm_cross_coords_update =norm_cross_coords_update
        self.debug = debug
        self.device = device
        self.invar_feats_dim_h = invar_feats_dim_h
        self.out_feats_dim_h = out_feats_dim_h
        # self.standard_norm_order = standard_norm_order
        self.pre_crossmsg_norm_type = pre_crossmsg_norm_type
        self.post_crossmsg_norm_type = post_crossmsg_norm_type
        self.all_sigmas_dist = [1.5 ** x for x in range(15)]
        self.save_trajectories = save_trajectories
        self.weight_sharing = weight_sharing

        # EDGES
        A_edge_mlp_input_dim = (invar_feats_dim_h * 2) + A_input_edge_feats_dim
        if self.use_dist_in_layers: 
            A_edge_mlp_input_dim += len(self.all_sigmas_dist)

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
            if self.use_dist_in_layers: #and self.B_evolve
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
        # if self.standard_norm_order:
        self.node_mlp_A = nn.Sequential(
            nn.Linear(invar_feats_dim_h + self.out_feats_dim_h, invar_feats_dim_h), #orig_invar_feats_dim_h + 2*
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
                nn.Linear(invar_feats_dim_h + self.out_feats_dim_h, invar_feats_dim_h), #orig_invar_feats_dim_h + 
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
        if self.use_dist_in_layers:# and self.A_evolve:
            x_rel_mag = edges.data['x_rel_m'] ** 2
            # print(x_rel_mag.device, edges.src['feat'].device, edges.dst['feat'].device, edges.data['feat'].device)
            x_rel_mag = torch.sum(x_rel_mag, dim=1, keepdim=True)
            x_rel_mag = torch.cat([torch.exp(-x_rel_mag / sigma) for sigma in self.all_sigmas_dist], dim=-1)
            return {'msg': self.A_edge_mlp(
                torch.cat([edges.src['feat_fine'], edges.dst['feat_coarse'], edges.data['feat'], x_rel_mag], dim=1))} # operates with edge features in it and node features
        else:
            return {
                'msg': self.A_edge_mlp(torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1))}

    def apply_edges_B(self, edges):
        if self.use_dist_in_layers: # and self.B_evolve:
            x_rel_mag = edges.data['x_rel_m'] ** 2
            x_rel_mag = torch.sum(x_rel_mag, dim=1, keepdim=True)
            x_rel_mag = torch.cat([torch.exp(-x_rel_mag / sigma) for sigma in self.all_sigmas_dist], dim=-1)
            return {'msg': self.B_edge_mlp(
                torch.cat([edges.src['feat_fine'], edges.dst['feat_coarse'], edges.data['feat'], x_rel_mag], dim=1))}
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

    def forward(self, A_pool, B_pool, fine_h_A, fine_h_B, coarse_h_A, coarse_h_B, fine_x_A, fine_x_B,
                pool_h_A, pool_h_B, pool_x_A, pool_x_B, og_pool_x_A, og_pool_x_B):
        with A_pool.local_scope() and B_pool.local_scope():
            A_pool.ndata['x_now'] = pool_x_A
            B_pool.ndata['x_now'] = pool_x_B
            A_pool.ndata['feat'] = pool_h_A 
            B_pool.ndata['feat'] = pool_h_B
            # print("pool_x_A", pool_x_A.shape)
            # print("pool_h_A", pool_h_A.shape)
            # print("fine_x_A", fine_x_A.shape)
            # print("fine_h_A", fine_h_A.shape)
            # print("coarse_x_A", coarse_x_A.shape)
            # print("coarse_h_A", coarse_h_A.shape)
# torch.cat((A.ndata['x'], torch.zeros((10,3)).to('cuda:0')), dim = 0)
            N = coarse_h_A.shape[0]
            n = fine_x_A.shape[0]
            D = fine_h_A.shape[1]
            # pooling graph has n + N nodes
            A_pool.ndata['x_fine'] = torch.cat((fine_x_A, torch.zeros((N,3)).to(self.device)), dim = 0)
            B_pool.ndata['x_fine'] = torch.cat((fine_x_B, torch.zeros((N,3)).to(self.device)), dim = 0)
            coarse_x_A = pool_x_A[-N,:] #! due to painn we use the current coordinates for the message passing
            coarse_x_B = pool_x_B[-N,:]
            A_pool.ndata['x_coarse'] = torch.cat((torch.zeros((n,3)).to(self.device), coarse_x_A), dim = 0)
            B_pool.ndata['x_coarse'] = torch.cat((torch.zeros((n,3)).to(self.device), coarse_x_B), dim = 0)

            A_pool.ndata['feat_fine'] = torch.cat((fine_h_A, torch.zeros((N,D)).to(self.device)), dim = 0)
            B_pool.ndata['feat_fine'] = torch.cat((fine_h_B, torch.zeros((N,D)).to(self.device)), dim = 0)
            A_pool.ndata['feat_coarse'] = torch.cat((torch.zeros((n,D)).to(self.device), coarse_h_A), dim = 0)
            B_pool.ndata['feat_coarse'] = torch.cat((torch.zeros((n,D)).to(self.device), coarse_h_B), dim = 0)

            # if self.debug:
            #     log(torch.max(A_graph.ndata['x_now'].abs()), 'x_now : x_i at layer entrance')
            #     log(torch.max(A_graph.ndata['feat'].abs()), 'data[feat] = h_i at layer entrance')

            # A_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'))
            A_pool.apply_edges(fn.u_sub_v('x_fine', 'x_coarse','x_rel_m'))  # x_I - x_j
            A_pool.apply_edges(fn.u_sub_v('x_now', 'x_now','x_rel'))
            if self.debug:
                log(torch.max(A_pool.edata['x_rel'].abs()), 'x_rel : x_I - x_j')
            
            # B_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'))
            B_pool.apply_edges(fn.u_sub_v('x_fine', 'x_coarse','x_rel_m'))
            B_pool.apply_edges(fn.u_sub_v('x_now', 'x_now','x_rel'))

            A_pool.apply_edges(self.apply_edges_A)  ## i->j edge:  [h_i h_j] phi^e edge_mlp
            B_pool.apply_edges(self.apply_edges_B) #apply_edges_rec)
            # Equation 1 message passing to create 'msg'
            # if self.debug:
            #     log(torch.max(A_graph.edata['msg'].abs()),
            #         'data[msg] = m_{i->j} = phi^e(h_i, h_j, f_{i,j}, x_rel_mag_ligand)')

            # h_feats_A_norm = apply_norm(A_pool, h_feats_A, self.final_h_layer_norm, self.final_h_layernorm_layer_A) #used in attention
            # h_feats_B_norm = apply_norm(B_pool, h_feats_B, self.final_h_layer_norm, self.final_h_layernorm_layer_B)

            # Equation 3: coordinate update
            # if self.A_evolve:
            A_pool.update_all(self.update_x_moment_A, fn.mean('m', 'x_update')) # phi_x coord_mlp
            # Inspired by https://arxiv.org/pdf/2108.10521.pdf we use original X and not only graph.ndata['x_now']
            x_evolved_A = self.x_connection_init * og_pool_x_A + (1. - self.x_connection_init) * A_pool.ndata['x_now'] + A_pool.ndata['x_update']
            B_pool.update_all(self.update_x_moment_B, fn.mean('m', 'x_update'))
            x_evolved_B = self.x_connection_init * og_pool_x_B + (1. - self.x_connection_init) * B_pool.ndata['x_now'] + B_pool.ndata['x_update']

            # Equation 4: Aggregate messages
            A_pool.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))#copy_edge
            B_pool.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))

            trajectory = []
            # TODO: Set up the following regularization for CG only
            assert(1==0)
            # if self.save_trajectories: trajectory.append(x_evolved_A.detach().cpu())
            # if self.loss_geometry_regularization:
            #     src, dst = geometry_graph_A.edges()
            #     src = src.long()
            #     dst = dst.long()
            #     d_squared = torch.sum((x_evolved_A[src] - x_evolved_A[dst]) ** 2, dim=1)
            #     geom_loss = torch.sum((d_squared - geometry_graph_A.edata['feat'] ** 2) ** 2)

            #     src, dst = geometry_graph_B.edges()
            #     src = src.long()
            #     dst = dst.long()
            #     d_squared += torch.sum((x_evolved_B[src] - x_evolved_B[dst]) ** 2, dim=1)
            #     geom_loss += torch.sum((d_squared - geometry_graph_B.edata['feat'] ** 2) ** 2)
            # else:
            #     geom_loss = 0
            # if self.geometry_regularization:
            #     src, dst = geometry_graph_A.edges()
            #     src = src.long()
            #     dst = dst.long()
            #     for step in range(self.geom_reg_steps):
            #         d_squared = torch.sum((x_evolved_A[src] - x_evolved_A[dst]) ** 2, dim=1)
            #         Loss = torch.sum((d_squared - geometry_graph_A.edata['feat'] ** 2)**2) # this is the loss whose gradient we are calculating here
            #         grad_d_squared = 2 * (x_evolved_A[src] - x_evolved_A[dst])
            #         geometry_graph_A.edata['partial_grads'] = 2 * (d_squared - geometry_graph_A.edata['feat'] ** 2)[:,None] * grad_d_squared
            #         geometry_graph_A.update_all(fn.copy_e('partial_grads', 'partial_grads_msg'),
            #                                   fn.sum('partial_grads_msg', 'grad_x_evolved'))
            #         grad_x_evolved = geometry_graph_A.ndata['grad_x_evolved']
            #         x_evolved_A = x_evolved_A + self.geometry_reg_step_size * grad_x_evolved
            #         if self.save_trajectories:
            #             trajectory.append(x_evolved_A.detach().cpu())

            #     src, dst = geometry_graph_B.edges()
            #     src = src.long()
            #     dst = dst.long()
            #     for step in range(self.geom_reg_steps):
            #         d_squared = torch.sum((x_evolved_B[src] - x_evolved_B[dst]) ** 2, dim=1)
            #         Loss = torch.sum((d_squared - geometry_graph_B.edata['feat'] ** 2)**2) # this is the loss whose gradient we are calculating here
            #         grad_d_squared = 2 * (x_evolved_B[src] - x_evolved_B[dst])
            #         geometry_graph_B.edata['partial_grads'] = 2 * (d_squared - geometry_graph_B.edata['feat'] ** 2)[:,None] * grad_d_squared
            #         geometry_graph_B.update_all(fn.copy_e('partial_grads', 'partial_grads_msg'),fn.sum('partial_grads_msg', 'grad_x_evolved'))
            #         grad_x_evolved = geometry_graph_B.ndata['grad_x_evolved']
            #         x_evolved_B = x_evolved_B + self.geometry_reg_step_size * grad_x_evolved

            input_node_upd_A = torch.cat((self.node_norm(A_pool.ndata['feat']), A_pool.ndata['aggr_msg']), dim=-1)
                                            #    cross_attention_A_feat,
                                            #    original_A_node_features), dim=-1)
            input_node_upd_B = torch.cat((self.node_norm(B_pool.ndata['feat']), B_pool.ndata['aggr_msg']), dim=-1)
                                                #  cross_attention_B_feat,
                                                #  original_B_node_features), dim=-1)
            # Skip connections
            # Equation 5: node updates --> cross attention is mu
            if self.invar_feats_dim_h == self.out_feats_dim_h: #phi^h
                node_upd_A = self.skip_weight_h * self.node_mlp_A(input_node_upd_A) + (1. - self.skip_weight_h) * pool_h_A
                node_upd_B = self.skip_weight_h * self.node_mlp_B(input_node_upd_B) + (1. - self.skip_weight_h) * pool_h_B
            else:
                assert(1 == 0)
                node_upd_A = self.node_mlp_A(input_node_upd_A) # phi^h
                node_upd_B = self.node_mlp_B(input_node_upd_B)

            node_upd_A = apply_norm(A_pool, node_upd_A, self.final_h_layer_norm, self.final_h_layernorm_layer_A)
            node_upd_B = apply_norm(B_pool, node_upd_B, self.final_h_layer_norm, self.final_h_layernorm_layer_B)
            return x_evolved_A, node_upd_A, x_evolved_B, node_upd_B 

    def __repr__(self):
        return "Pooling 3D Layer " + str(self.__dict__)

class Coarse_Grain_3DLayer(nn.Module):
    def __init__(
            self,
            # orig_invar_feats_dim_h, #orig_h_feats_dim,
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
            # standard_norm_order=True,
            normalize_coordinate_update=False,
            # A_evolve = True,# lig_evolve=True,
            # B_evolve = True, #rec_evolve=True,
            # fine_tune=False,
            geometry_regularization=False,
            pre_crossmsg_norm_type=0,
            post_crossmsg_norm_type=0,
            norm_cross_coords_update= False,
            loss_geometry_regularization = True,
            geom_reg_steps= 1,
            geometry_reg_step_size=0.1,
            weight_sharing=True,
    ):

        super(Coarse_Grain_Layer, self).__init__()
        
        # self.fine_tune = fine_tune
        # TODO clean class and initialize everything
        self.eps = 1e-7
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
        # self.A_evolve = A_evolve
        # self.B_evolve = B_evolve
        self.invar_feats_dim_h = invar_feats_dim_h
        self.out_feats_dim_h = out_feats_dim_h
        # self.standard_norm_order = standard_norm_order
        self.pre_crossmsg_norm_type = pre_crossmsg_norm_type
        self.post_crossmsg_norm_type = post_crossmsg_norm_type
        self.all_sigmas_dist = [1.5 ** x for x in range(15)]
        self.geometry_regularization = geometry_regularization
        self.geom_reg_steps = geom_reg_steps
        self.save_trajectories = save_trajectories
        self.weight_sharing = weight_sharing

        # EDGES
        A_edge_mlp_input_dim = (invar_feats_dim_h * 2) + A_input_edge_feats_dim
        if self.use_dist_in_layers: #TRUE and TRUE
            A_edge_mlp_input_dim += len(self.all_sigmas_dist)

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
            if self.use_dist_in_layers:
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
        # if self.weight_sharing:
        #     self.att_mlp_Q_B = self.att_mlp_Q_A
        #     self.att_mlp_K_B = self.att_mlp_K_A
        #     self.att_mlp_V_B = self.att_mlp_V_A
        # else:
        #     self.att_mlp_Q_B = nn.Sequential(
        #         nn.Linear(invar_feats_dim_h, invar_feats_dim_h, bias=False),
        #         get_non_lin(nonlin, leakyrelu_neg_slope),
        #     )
        #     self.att_mlp_K_B = nn.Sequential(
        #         nn.Linear(invar_feats_dim_h, invar_feats_dim_h, bias=False),
        #         get_non_lin(nonlin, leakyrelu_neg_slope),
        #     )
        #     self.att_mlp_V_B = nn.Sequential(
        #         nn.Linear(invar_feats_dim_h, invar_feats_dim_h, bias=False),
        #     )
        # if self.standard_norm_order:
        self.node_mlp_A = nn.Sequential(
            nn.Linear(2 * invar_feats_dim_h + self.out_feats_dim_h, invar_feats_dim_h), #orig_invar_feats_dim_h + 
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
                nn.Linear(2 * invar_feats_dim_h + self.out_feats_dim_h, invar_feats_dim_h), #orig_invar_feats_dim_h + 
                get_layer_norm(layer_norm, invar_feats_dim_h),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(invar_feats_dim_h, out_feats_dim_h),
                get_layer_norm(layer_norm, out_feats_dim_h),
            )

        self.final_h_layernorm_layer_A = get_norm(self.final_h_layer_norm, out_feats_dim_h)
        # self.pre_crossmsg_norm_A = get_norm(self.pre_crossmsg_norm_type, invar_feats_dim_h)
        # self.post_crossmsg_norm_A = get_norm(self.post_crossmsg_norm_type, invar_feats_dim_h)
        
        if self.weight_sharing:
            self.final_h_layernorm_layer_B = self.final_h_layernorm_layer_A
            # self.pre_crossmsg_norm_B = self.pre_crossmsg_norm_A
            # self.post_crossmsg_norm_B = self.post_crossmsg_norm_A
        else:
            self.final_h_layernorm_layer_B = get_norm(self.final_h_layer_norm, out_feats_dim_h)
            # self.pre_crossmsg_norm_B = get_norm(self.pre_crossmsg_norm_type, invar_feats_dim_h)
            # self.post_crossmsg_norm_B = get_norm(self.post_crossmsg_norm_type, invar_feats_dim_h)

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

    # def apply_rbf_A(self, edges):
    #     # self.params['max_correlation_length'] = 5, self.params['num_rbf'] = 12
    #     # self.all_sigmas_dist = [1.5 ** x for x in range(15)]
    #     # ? Do we want egnn style rbf or linker style
    #     x_rel_mag = edges.data['x_rel_m'] ** 2
    #     x_rel_mag = torch.sum(x_rel_mag, dim=1, keepdim=True)
    #     rbf = torch.cat([torch.exp(-x_rel_mag / sigma ) for sigma in self.all_sigmas_dist], dim=-1)
    #     # ? Do we need the 1x1 random parameter and the centering?
    #     rbf = torch.linspace(0, self.max_correlation_length, self.num_rbf, device=self.device)
    #     dist = torch.linalg.norm(edges.data['x_rel_m'], dim=1)
    #     rbf = torch.square(torch.tile(dist, [1, self.params['num_rbf']]) - rbf) * self.weights['A_rbf_single'] 
    #     # w_rbfs = []
    #     # for i in range(3): # W: 12=num_rbf x (32(h) | 12(v) | 12(v)) FOR 1 2 3
    #     #     w_rbf = torch.einsum('nh, ijn->ijh', self.weights['A_rbf_'+ str(i)], rbf) + self.weights['A_biases_rbf_'+str(i)]
    #     #     w_rbfs.append(w_rbf) #torch.Size([8, 28, 28, 32])
    #     #     # should be the same as a linear layer on this
    #     w_rbfs = [self.rbf_A_1(rbf), self.rbf_A_2(rbf), self.rbf_A_3(rbf)] #[ n_rbf X D, n_rbf X F, n_rbf X F]
    #      return {
    #         "rbf_h": w_rbfs[0],
    #         "rbf_v_1": w_rbfs[1],
    #         "rbf_v_2":w_rbfs[2]
    #      }

    def generate_mixed_features_A(self, nodes):
        h = nodes.data['feat_pool']
        v = nodes.data['v_now']
        v_norm = nodes.data['v_norm']

        combo = torch.cat([h, v_norm], dim =1)
        hp = self.scalar_neuron_h_A(combo)
        hpp_0 = self.scalar_neuron_v_0_A(h) #? should this be combo
        hpp_1 = self.scalar_neuron_v_1_A(h) #? should this be combo
        vp = self.vector_neuron_A(v)

        return {
            "hp":hp,
            "hpp_0": hpp_0,
            "hpp_1": hpp_1,
            "vp": vp
        }

    def apply_edges_A(self, edges):
        if self.use_dist_in_layers:# and self.A_evolve:
            x_rel_mag = edges.data['x_rel_m'] ** 2
            # print(x_rel_mag.device, edges.src['feat'].device, edges.dst['feat'].device, edges.data['feat'].device)
            x_rel_mag = torch.sum(x_rel_mag, dim=1, keepdim=True)
            x_rel_mag = torch.cat([torch.exp(-x_rel_mag / sigma) for sigma in self.all_sigmas_dist], dim=-1)
            return {'msg': self.A_edge_mlp(
                torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], x_rel_mag], dim=1))} #

    # def point_convolution_A(self, edges):
    #     hp = edges.dst['hp']
    #     rbf_h = edges.dst['rbf_h']
    #     hpp_0 = edges.dst['hpp_0']
    #     hpp_1 = edges.dst['hpp_1']
    #     rbf_v_1 = edges.dst['rbf_v_1']
    #     vp = edges.dst['vp']
    #     rbf_v_2 = edges.dst['rbf_v_2']
    #     x_delta = edges.dst['x_delta']

    #     m_h = torch.einsum('jh, ijh->ijh', hp, rbf_h)
    #     mv = torch.einsum('jvk, jv, ijv->ijvk', phi_v_v, hpp_0, rbf_v_1)
    #     mv = mv + torch.einsum('ijk, jv, ijv->ijvk', x_delta, hpp_1, rbf_v_2)

    #     return {
    #         "msg_h": m_h,
    #         "msg_v": m_v
    #     }

    def point_convolution_A(self, edges):
        rbf = torch.linspace(0, self.max_correlation_length, self.num_rbf, device=self.device)
        r_ij = edges.data['r_ij'] # edges x 3
        r_ij_norm = torch.linalg.norm(r_ij, dim=1).unsqueeze(1) # edges x 1
        rbf = torch.exp(-torch.square(torch.tile(r_ij_norm, [1, self.params['num_rbf']]) - rbf) * self.weights['A_rbf_k']) # edges x num_rbf
        rbf_h = self.rbf_A(rbf, 1) # edges x D
        rbf_v_1 = self.rbf_A(rbf, 2) # edges x F
        rbf_v_2 = self.rbf_A(rbf, 3) # edges x F
        
        hp = edges.dst['hp'] # edges x D
        hpp_0 = edges.dst['hpp_0'] # E x F
        hpp_1 = edges.dst['hpp_1'] # E x F
        vp = edges.dst['vp'] # E x F x 3

        msg_h = rbf_h * hp

        msg_v_0 = rbf_v_1.unsqueeze(2)*vp# E x F * E X F x 3
        msg_v_1 = ((r_ij.T)@(hpp1_*rbf_v_2)).T # ((E x 3)T@(ExF * ExF))T = E x 3
        msg_v = msg_v_0 + msg_v_1.unsqueeze(1)

        return {
            "msg_h": msg_h,
            "msg_v": msg_v
        }

    def message_passing(self, g):
        hp = g.ndata['hp']
        rbf_h = g.ndata['rbf_h']
        hpp_0 = g.ndata['hpp_0']
        hpp_1 = g.ndata['hpp_1']
        rbf_v_1 = g.ndata['rbf_v_1']
        vp = g.ndata['vp']
        rbf_v_2 = g.ndata['rbf_v_2']
        x_delta = g.ndata['x_delta']

        m = torch.einsum('jh, ijh->ijh', hp, rbf_h)
        m_v = torch.einsum('jvk, jv, ijv->ijvk', vp, hpp_0, rbf_v_1)
        m_v = m_v + torch.einsum('ijk, jv, ijv->ijvk', x_delta, hpp_1, rbf_v_2)
        # m = torch.einsum('bjh, bijh, bij->bijh', phi_h_s, w_rbfs[0], adj[:, edge_type])
        # m_v = torch.einsum('bjvk, bjv, bijv, bij->bijvk', phi_v_v, phi_h_v0, w_rbfs[1], adj[:, edge_type]) 
        # m_v += torch.einsum('bijk, bjv, bijv, bij->bijvk', x_delta, phi_h_v1, w_rbfs[2], adj[:, edge_type])
        # aggregate messages
        m = torch.sum(m, dim=1) # same shape as h
        m_v = torch.sum(m_v, dim=1)# same shape as v
        return m, m_v

    def rbf_A(self, data, idx):
        if idx == 1: #TODO init 3 different linear layers
            return self.rbf_A_1(data)
        elif idx == 2:
            return self.rbf_A_2(data)
        else:
            return self.rbf_A_3(data)

    def forward(self, A_graph, B_graph, v_A, v_B, coords_A, h_feats_A, orig_coords_A,
                coords_B, h_feats_B, orig_coords_B, mask, geometry_graph_A, geometry_graph_B,
                pool_coords_A, pool_coords_B, pool_feats_A, pool_feats_B): #original_B_node_features, original_A_node_features
        with A_graph.local_scope() and B_graph.local_scope():
            # A_graph.ndata['x_now'] = coords_A # no longer need
            # B_graph.ndata['x_now'] = coords_B
            A_graph.ndata['feat'] = h_feats_A  # first time set here
            B_graph.ndata['feat'] = h_feats_B
            N = coords_A.shape[0]
            A_graph.ndata['x_pool'] = pool_coords_A[-N:, :]
            B_graph.ndata['x_pool'] = pool_coords_B[-N:, :]
            A_graph.ndata['feat_pool'] = pool_feats_A[-N:, :]
            B_graph.ndata['feat_pool'] = pool_feats_B[-N:, :]

            A_graph.ndata['v_now'] = v_A
            N = V_A.shape[0]
            B_graph.ndata['v_now'] = v_B

            A_graph.apply_edges(fn.u_sub_v('x_pool', 'x_pool', 'r_ij'))  # x_i - x_j
            # B_graph.apply_edges(fn.u_sub_v('x_pool', 'x_pool', 'x_rel_m')) #! skip for debug
            # ? Do we need this at a node level. Most likely yes to be able to set yp the messages. We can't have distance edge feautres
            # x = A_graph.ndata['x_pool']
            # left = torch.unsqueeze(x, dim=1).repeat(1, N, 1)
            # right = torch.unsqueeze(x, dim=0).repeat(N, 1, 1)
            # A_graph.ndata['x_delta'] = left - right # N x N x 3
            # A_graph.ndata['dist'] = torch.linalg.norm(A_graph.ndata['x_delta'], dim=2)
            # x = B_graph.ndata['x_pool'] #! skip for debug
            # left = torch.unsqueeze(x, dim=1).repeat(1, N, 1)
            # right = torch.unsqueeze(x, dim=0).repeat(N, 1, 1)
            # B_graph.ndata['x_rel_m'] = left - right # N x N x 3
            # B_graph.ndata['dist'] = torch.linalg.norm(B_graph.ndata['x_delta'], dim=2)


            # A_graph.apply_edges(self.apply_rbf_A)
            # rbf = torch.linspace(0, self.max_correlation_length, self.num_rbf, device=self.device)
            # rbf_A = torch.square(torch.tile(A_graph.ndata['dist'], [1, self.num_rbf]) - rbf) * self.weights['A_rbf_single'] 
            # A_graph.ndata["rbf_h"] = self.rbf_A(rbf_A, 1)
            # A_graph.ndata["rbf_v_1"] = self.rbf_A(rbf_A, 2)
            # A_graph.ndata["rbf_v_2"] = self.rbf_A(rbf_A, 3)
            # B_graph.apply_edges(self.apply_rbf_B) #! skip for debug
            # rbf_B = torch.square(torch.tile(B_graph.ndata['dist'], [1, self.num_rbf]) - rbf) * self.weights['B_rbf_single'] 
            # A_graph.ndata["rbf_h"] = self.rbf_B(rbf_A, 1)
            # A_graph.ndata["rbf_v_1"] = self.rbf_B(rbf_A, 2)
            # A_graph.ndata["rbf_v_2"] = self.rbf_B(rbf_A, 3)

            A_graph.ndata['v_norm'] = torch.norm(torch.einsum('nci, cd->ndi', v_A, self.weights['A_v_norm_preprocess']), dim=2) + self.eps
            # v_norm_B = torch.norm(torch.einsum('nci, cd->ndi', v, self.weights['B_v_norm_preprocess']), dim=2) + self.eps#! skip for debug
            # TODO: Implement Attention 
            #?: Should we do attention over the mixed features?
            h_feats_A_norm = apply_norm(A_graph, h_feats_A, self.final_h_layer_norm, self.final_h_layernorm_layer_A)
            # h_feats_B_norm = apply_norm(B_graph, h_feats_B, self.final_h_layer_norm, self.final_h_layernorm_layer_B)#! skip for debug
            
            cross_attention_A_feat = cross_attention(self.att_mlp_Q_A(h_feats_A_norm),
                                                       self.att_mlp_K_B(h_feats_B_norm),
                                                       self.att_mlp_V_B(h_feats_B_norm), mask, self.cross_msgs)
            # cross_attention_B_feat = 0*cross_attention_A_feat
            cross_attention_A_feat = apply_norm(A_graph, cross_attention_A_feat, self.final_h_layer_norm, self.final_h_layernorm_layer_A)

            # TODO feature mixing: apply nodes does all nodes in graph so its ok
            A_graph.apply_nodes(self.generate_mixed_features_A) # use the pooling feats
            # B_graph.apply_nodes(self.generate_mixed_features_B)#! skip for debug
            
            # TODO implement message passing
            A_graph.apply_edges(self.point_convolution_A)
            # B_graph.apply_edges(self.point_convolution_B)#! skip for debug
            A_graph.update_all(fn.copy_e('msg_h', 'm_h'), fn.mean('m_h', 'aggr_msg_h'))#copy_edge
            # B_graph.update_all(fn.copy_e('msg_h', 'm_h'), fn.mean('m_h', 'aggr_msg_h'))#! skip for debug
            A_graph.update_all(fn.copy_e('msg_v', 'm_v'), fn.mean('m_v', 'aggr_msg_v'))#copy_edge
            # B_graph.update_all(fn.copy_e('msg_v', 'm_v'), fn.mean('m_v', 'aggr_msg_v'))#! skip for debug
            # ? OR
            # A_graph.ndata['aggr_msg'], A_graph.ndata['aggr_msg_v'] = self.message_passing(A_graph)
            # B_graph.ndata['aggr_msg'], B_graph.ndata['aggr_msg_v'] = self.message_passing(B_graph)#! skip for debug

            input_node_upd_A = torch.cat((self.node_norm(A_graph.ndata['feat']),
                                            #    A_graph.ndata['aggr_msg_h'],
                                               cross_attention_A_feat), dim=-1)
            # input_node_upd_B = torch.cat((self.node_norm(B_graph.ndata['feat']),#! skip for debug
            #                                 #    A_graph.ndata['aggr_msg'],
            #                                    cross_attention_B_feat), dim=-1)
            # TODO do we want to have the residual
            node_upd_A = A_graph.ndata['feat'] + self.gru(A_graph.ndata['aggr_msg'], input_node_upd_A).reshape(h_feat_A.shape)
            # node_upd_B = self.gru(B_graph.ndata['aggr_msg'], input_node_upd_B).reshape(h_feat_A.shape)#! skip for debug

            v_evolved_A_input = torch.cat([v_A, A_graph.ndata['aggr_msg_v']], dim = -2)
            # v_evolved_B_input = torch.cat([v_B, B_graph.ndata['aggr_msg_v']], dim = -2)#! skip for debug

            # TODO do we want to have the residual
            v_evolved_A = v_A + self.fully_connected_vec_A(v_evolved_A_input)
            # v_evolved_B = self.fully_connected_vec_B(v_evolved_B_input)#! skip for debug
            return v_evolved_A, node_upd_A, #v_evolved_A, node_upd_A, #! skip for debug


    #         # Equation 3: coordinate update
    #         A_graph.update_all(self.update_x_moment_A, fn.mean('m', 'x_update')) # phi_x coord_mlp
    #         # Inspired by https://arxiv.org/pdf/2108.10521.pdf we use original X and not only graph.ndata['x_now']
    #         x_evolved_A = self.x_connection_init * orig_coords_A + (1. - self.x_connection_init) * A_graph.ndata['x_now'] + A_graph.ndata['x_update']

    #         B_graph.update_all(self.update_x_moment_B, fn.mean('m', 'x_update'))
    #         x_evolved_B = self.x_connection_init * orig_coords_B + (1. - self.x_connection_init) * \
    #                         B_graph.ndata['x_now'] + B_graph.ndata['x_update']
    #         # Equation 4: Aggregate messages
    #         A_graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))#copy_edge
    #         B_graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))

    #         trajectory = []
    #         # TODO: move the geom regularization to the Pooling layer since that is where we get the coordinates
    #         # TODO: have to create new pooling layer anyways since the coordinates are different
    #         if self.save_trajectories: trajectory.append(x_evolved_A.detach().cpu())
    #         if self.loss_geometry_regularization:
    #             src, dst = geometry_graph_A.edges()
    #             src = src.long()
    #             dst = dst.long()
    #             d_squared = torch.sum((x_evolved_A[src] - x_evolved_A[dst]) ** 2, dim=1)
    #             geom_loss = torch.sum((d_squared - geometry_graph_A.edata['feat'] ** 2) ** 2)

    #             src, dst = geometry_graph_B.edges()
    #             src = src.long()
    #             dst = dst.long()
    #             d_squared += torch.sum((x_evolved_B[src] - x_evolved_B[dst]) ** 2, dim=1)
    #             geom_loss += torch.sum((d_squared - geometry_graph_B.edata['feat'] ** 2) ** 2)
    #         else:
    #             geom_loss = 0
    #         if self.geometry_regularization:
    #             src, dst = geometry_graph_A.edges()
    #             src = src.long()
    #             dst = dst.long()
    #             for step in range(self.geom_reg_steps):
    #                 d_squared = torch.sum((x_evolved_A[src] - x_evolved_A[dst]) ** 2, dim=1)
    #                 Loss = torch.sum((d_squared - geometry_graph_A.edata['feat'] ** 2)**2) # this is the loss whose gradient we are calculating here
    #                 grad_d_squared = 2 * (x_evolved_A[src] - x_evolved_A[dst])
    #                 geometry_graph_A.edata['partial_grads'] = 2 * (d_squared - geometry_graph_A.edata['feat'] ** 2)[:,None] * grad_d_squared
    #                 geometry_graph_A.update_all(fn.copy_e('partial_grads', 'partial_grads_msg'),
    #                                           fn.sum('partial_grads_msg', 'grad_x_evolved'))
    #                 grad_x_evolved = geometry_graph_A.ndata['grad_x_evolved']
    #                 x_evolved_A = x_evolved_A + self.geometry_reg_step_size * grad_x_evolved
    #                 if self.save_trajectories:
    #                     trajectory.append(x_evolved_A.detach().cpu())

    #             src, dst = geometry_graph_B.edges()
    #             src = src.long()
    #             dst = dst.long()
    #             for step in range(self.geom_reg_steps):
    #                 d_squared = torch.sum((x_evolved_B[src] - x_evolved_B[dst]) ** 2, dim=1)
    #                 Loss = torch.sum((d_squared - geometry_graph_B.edata['feat'] ** 2)**2) # this is the loss whose gradient we are calculating here
    #                 grad_d_squared = 2 * (x_evolved_B[src] - x_evolved_B[dst])
    #                 geometry_graph_B.edata['partial_grads'] = 2 * (d_squared - geometry_graph_B.edata['feat'] ** 2)[:,None] * grad_d_squared
    #                 geometry_graph_B.update_all(fn.copy_e('partial_grads', 'partial_grads_msg'),fn.sum('partial_grads_msg', 'grad_x_evolved'))
    #                 grad_x_evolved = geometry_graph_B.ndata['grad_x_evolved']
    #                 x_evolved_B = x_evolved_B + self.geometry_reg_step_size * grad_x_evolved


    #         input_node_upd_A = torch.cat((self.node_norm(A_graph.ndata['feat']),
    #                                            A_graph.ndata['aggr_msg'],
    #                                            cross_attention_A_feat), dim=-1)
    #                                         #    original_A_node_features), dim=-1)
    #         # we are zero initializing the og node features as there is no mebedding

    #         input_node_upd_B = torch.cat((self.node_norm(B_graph.ndata['feat']),
    #                                              B_graph.ndata['aggr_msg'],
    #                                              cross_attention_B_feat), dim=-1)
    #                                             #  original_B_node_features), dim=-1)

    #         # Skip connections
    #         # Equation 5: node updates --> cross attention is mu
    #         if self.invar_feats_dim_h == self.out_feats_dim_h: #phi^h
    #             node_upd_A = self.skip_weight_h * self.node_mlp_A(input_node_upd_A) + (1. - self.skip_weight_h) * h_feats_A
    #             node_upd_B = self.skip_weight_h * self.node_mlp_B(input_node_upd_B) + (1. - self.skip_weight_h) * h_feats_B
    #         else:
    #             node_upd_A = self.node_mlp_A(input_node_upd_A) # phi^h
    #             node_upd_B = self.node_mlp_B(input_node_upd_B)

    #         node_upd_A = apply_norm(A_graph, node_upd_A, self.final_h_layer_norm, self.final_h_layernorm_layer_A)
    #         node_upd_B = apply_norm(B_graph, node_upd_B, self.final_h_layer_norm, self.final_h_layernorm_layer_B)
    #         return x_evolved_A, node_upd_A, x_evolved_B, node_upd_B, trajectory, geom_loss

    def __repr__(self):
        return "Coarse Grain 3D Layer " + str(self.__dict__)

        # phi_h_s = scalar_neuron(torch.cat([h, v_norm], dim=2),
        #             self.weights['edge_s_output_weights'+weights_suffix][edge_type],
        #             self.weights['edge_s_biases'+weights_suffix][edge_type]) #torch.Size([8, 28, 32])

        # phi_h_v0 = scalar_neuron(h,
        #             self.weights['edge_v0_output_weights'+weights_suffix][edge_type],
        #             self.weights['edge_v0_biases'+weights_suffix][edge_type]) #torch.Size([8, 28, 12])

        # phi_h_v1 = scalar_neuron(h,
        #             self.weights['edge_v1_output_weights' + weights_suffix][edge_type],
        #             self.weights['edge_v1_biases' + weights_suffix][edge_type]) #torch.Size([8, 28, 12])

        # phi_v_v = vector_neuron(v, 
        #             self.weights['edge_v_nonlinear_Q' + weights_suffix][edge_type],
        #             self.weights['edge_v_nonlinear_K' + weights_suffix][edge_type]) #torch.Size([8, 28, 12, 3])
            
        # m += torch.einsum('bjh, bijh, bij->bijh', phi_h_s, w_rbfs[0], adj[:, edge_type]) # torch.Size([8, 28, 28, 32])

        # m_v += torch.einsum('bjvk, bjv, bijv, bij->bijvk', phi_v_v, phi_h_v0, w_rbfs[1], adj[:, edge_type]) + \
        #         torch.einsum('bijk, bjv, bijv, bij->bijvk', x_delta, phi_h_v1, w_rbfs[2], adj[:, edge_type]) #

        # m = torch.sum(m, dim=2) #torch.Size([8, 28, 32]) # same shape as h
        # m_v = torch.sum(m_v, dim=2) # torch.Size([8, 28, 12, 3]) # same shape as v
 