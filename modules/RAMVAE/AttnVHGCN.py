import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.utils import softmax as scatter_softmax
import math
from logging import getLogger
from typing import Tuple

class AttnVHGCN(nn.Module):
    """
    Heterogeneous Graph Convolutional Network
    """
    def __init__(
        self,
        channel,
        n_hops,
        n_users,
        n_relations,
        qk_shared=False,
        node_dropout_rate=0.5,
        mess_dropout_rate=0.1):
        super(AttnVHGCN, self).__init__()

        self.logger = getLogger()

        self.no_attn_convs = nn.ModuleList()
        self.n_hops = n_hops
        self.n_users = n_users
        self.n_relations = n_relations
        self.qk_shaerd = qk_shared
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        initializer = nn.init.xavier_uniform_
        relation_emb = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.relation_emb = nn.Parameter(relation_emb)  # [n_relations - 1, in_channel]

        self.W_K = nn.Parameter(torch.Tensor(channel, channel))
        self.W_Q = self.W_K if qk_shared else nn.Parameter(torch.Tensor(channel, channel))
        self.W_K_mu = nn.Parameter(torch.Tensor(channel, channel))
        self.W_Q_mu = self.W_K_mu if qk_shared else nn.Parameter(torch.Tensor(channel, channel))
        self.W_K_logvar = nn.Parameter(torch.Tensor(channel, channel))
        self.W_Q_logvar = self.W_K_logvar if qk_shared else nn.Parameter(torch.Tensor(channel, channel))

        self.n_heads = 2
        self.d_k = channel // self.n_heads

        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K_mu)
        nn.init.xavier_uniform_(self.W_Q_mu)
        nn.init.xavier_uniform_(self.W_K_logvar)
        nn.init.xavier_uniform_(self.W_Q_logvar)

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout
        return
    
    def non_attn_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = relation_emb[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """user aggregate"""
        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg
        
    def shared_layer_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_K).view(-1, self.n_heads, self.d_k)

        key = key * relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)

        relation_emb = relation_emb[edge_type - 1] # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)

        edge_attn_score = scatter_softmax(edge_attn_score, head)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads*self.d_k)
        # attn weight makes mean to sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]] #interaction weight * item embedding
        # w_attn = self.ui_weighting(user_emb, entity_emb, inter_edge)
        # item_agg += w_attn.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg

    def to_mu(self, entity_emb, edge_index, edge_type, relation_emb):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q_mu).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_K_mu).view(-1, self.n_heads, self.d_k)

        key = key * relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)

        relation_emb = relation_emb[edge_type - 1] # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)

        edge_attn_score = scatter_softmax(edge_attn_score, head)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads*self.d_k)
        # attn weight makes mean to sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)
        return entity_agg
    
    def to_logvar(self, entity_emb, edge_index, edge_type, relation_emb):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q_logvar).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_K_logvar).view(-1, self.n_heads, self.d_k)

        key = key * relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)

        relation_emb = relation_emb[edge_type - 1] # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)

        edge_attn_score = scatter_softmax(edge_attn_score, head)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads*self.d_k)
        # attn weight makes mean to sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)
        return entity_agg

    # @TimeCounter.count_time(warmup_interval=4)
    def forward_deterministic(
        self, user_emb, entity_emb, edge_index, edge_type,
        inter_edge, inter_edge_w, mess_dropout=True, item_attn=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        #3.2計算
        if item_attn is not None:
            item_attn = item_attn[inter_edge[1, :]]
            item_attn = scatter_softmax(item_attn, inter_edge[0, :])
            norm = scatter_sum(torch.ones_like(inter_edge[0, :]), inter_edge[0, :], dim=0, dim_size=user_emb.shape[0])
            norm = torch.index_select(norm, 0, inter_edge[0, :])
            item_attn = item_attn * norm
            inter_edge_w = inter_edge_w * item_attn

        entity_emb_h = entity_emb  # [n_entity, channel]
        user_emb_h = user_emb  # [n_users, channel]
        for i in range(self.n_hops):
            entity_emb, user_emb = self.shared_layer_agg(user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, self.relation_emb)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_emb_h = torch.add(entity_emb_h, entity_emb)
            user_emb_h = torch.add(user_emb_h, user_emb)
        return entity_emb_h, user_emb_h

    # @TimeCounter.count_time(warmup_interval=4)
    def forward(
            self, user_emb, entity_emb, edge_index, edge_type,
            inter_edge, inter_edge_w, stochastic, mess_dropout=True, item_attn=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        entity_emb_h, user_emb_h = self.forward_deterministic(
            user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, mess_dropout, item_attn)
        
        if not stochastic:
            return entity_emb_h, user_emb_h, None, None, None, None
        
        entity_emb_mu = self.to_mu(entity_emb_h, edge_index, edge_type, self.relation_emb)
        entity_emb_logvar = self.to_logvar(entity_emb_h, edge_index, edge_type, self.relation_emb)

        # reparameterise
        epsilon = torch.randn_like(entity_emb_mu)
        entity_emb_sample = entity_emb_mu + epsilon * torch.exp(entity_emb_logvar / 2)

        # user
        item_agg_sample = inter_edge_w.unsqueeze(-1) * entity_emb_sample[inter_edge[1, :]]
        user_agg_sample = scatter_sum(src=item_agg_sample, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)

        item_agg_mu = inter_edge_w.unsqueeze(-1) * entity_emb_mu[inter_edge[1, :]]
        user_agg_mu = scatter_sum(src=item_agg_mu, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        
        if mess_dropout:
            entity_emb_mu = self.dropout(entity_emb_mu)
            user_agg_mu = self.dropout(user_agg_mu)
            entity_emb_sample = self.dropout(entity_emb_sample)
            user_agg_sample = self.dropout(user_agg_sample)

        entity_emb_result = torch.add(entity_emb_h, F.normalize(entity_emb_mu))
        user_emb_result = torch.add(user_emb_h, F.normalize(user_agg_mu))
        entity_emb_result_sample = torch.add(entity_emb_h, F.normalize(entity_emb_sample))
        user_emb_result_sample = torch.add(user_emb_h, F.normalize(user_agg_sample))
        return entity_emb_result, user_emb_result, entity_emb_result_sample, user_emb_result_sample, entity_emb_mu, entity_emb_logvar
    
    def forward_ui(self, user_emb, item_emb, inter_edge, inter_edge_w, mess_dropout=True):
        item_res_emb = item_emb  # [n_entity, channel]
        for i in range(self.n_hops):
            user_emb, item_emb = self.ui_agg(user_emb, item_emb, inter_edge, inter_edge_w)
            """message dropout"""
            if mess_dropout:
                item_emb = self.dropout(item_emb)
                user_emb = self.dropout(user_emb)
            item_emb = F.normalize(item_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            item_res_emb = torch.add(item_res_emb, item_emb) #全hop的item embedding相加
        return item_res_emb
    
    def forward_kg(self, entity_emb, edge_index, edge_type, mess_dropout=True):
        entity_res_emb = entity_emb
        for i in range(self.n_hops):
            entity_emb = self.kg_agg(entity_emb, edge_index, edge_type)
            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb) #全hop的item embedding相加
        return entity_res_emb

    def ui_agg(self, user_emb, item_emb, inter_edge, inter_edge_w): #使用scatter_sum
        num_items = item_emb.shape[0]
        item_emb = inter_edge_w.unsqueeze(-1) * item_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_emb, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        user_emb = inter_edge_w.unsqueeze(-1) * user_emb[inter_edge[0, :]]
        item_agg = scatter_sum(src=user_emb, index=inter_edge[1, :], dim_size=num_items, dim=0)
        return user_agg, item_agg

    def kg_agg(self, entity_emb, edge_index, edge_type): #使用scatter_mean
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        edge_relation_emb = self.relation_emb[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        return entity_agg

    @torch.no_grad()
    def norm_attn_computer(self, entity_emb, edge_index, edge_type=None, print=False, return_logits=False):
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        if edge_type is not None:
            key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_logits = edge_attn.mean(-1).detach()
        # softmax by head_node
        edge_attn_score = scatter_softmax(edge_attn_logits, head) #公式2
        # normalization by head_node degree
        norm = scatter_sum(torch.ones_like(head), head, dim=0, dim_size=entity_emb.shape[0]) #out-degree
        norm = torch.index_select(norm, 0, head)
        edge_attn_score = edge_attn_score * norm #公式6
        # print attn score
        if print:
            self.logger.info("edge_attn_score std: {}".format(edge_attn_score.std()))
        if return_logits:
            return edge_attn_score, edge_attn_logits
        return edge_attn_score