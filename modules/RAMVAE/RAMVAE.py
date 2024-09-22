
import numpy as np
import torch
import torch.nn as nn
from .AttnVHGCN import AttnVHGCN
from ..contrast import Contrast
from logging import getLogger
import torch.nn.functional as F
from torch_scatter import scatter_mean


def _adaptive_kg_drop_cl(edge_index, edge_type, edge_attn_score, keep_rate):
    _, least_attn_edge_id = torch.topk(-edge_attn_score,
                                       int((1-keep_rate) * edge_attn_score.shape[0]), sorted=False) #如果keep_rate=0.7,則會drop ateention score 最低0.3
    cl_kg_mask = torch.ones_like(edge_attn_score).bool()
    cl_kg_mask[least_attn_edge_id] = False
    cl_kg_edge = edge_index[:, cl_kg_mask]
    cl_kg_type = edge_type[cl_kg_mask]
    return cl_kg_edge, cl_kg_type

def _adaptive_ui_drop_cl(item_attn_mean, inter_edge, inter_edge_w, keep_rate=0.7, samp_func = "torch"):
    inter_attn_prob = item_attn_mean[inter_edge[1]]
    # add gumbel noise
    noise = -torch.log(-torch.log(torch.rand_like(inter_attn_prob)))
    """ prob based drop """
    inter_attn_prob = inter_attn_prob + noise
    inter_attn_prob = F.softmax(inter_attn_prob, dim=0)

    if samp_func == "np":
        # we observed abnormal behavior of torch.multinomial on mind
        sampled_edge_idx = np.random.choice(np.arange(inter_edge_w.shape[0]), size=int(keep_rate * inter_edge_w.shape[0]), replace=False, p=inter_attn_prob.cpu().numpy())
    else:
        sampled_edge_idx = torch.multinomial(inter_attn_prob, int(keep_rate * inter_edge_w.shape[0]), replacement=False)

    return inter_edge[:, sampled_edge_idx], inter_edge_w[sampled_edge_idx]/keep_rate


def _relation_aware_edge_sampling(edge_index, edge_type, n_relations, samp_rate=0.5):
    # exclude interaction
    for i in range(n_relations - 1):
        edge_index_i, edge_type_i = _edge_sampling(
            edge_index[:, edge_type == i], edge_type[edge_type == i], samp_rate)
        if i == 0:
            edge_index_sampled = edge_index_i
            edge_type_sampled = edge_type_i
        else:
            edge_index_sampled = torch.cat(
                [edge_index_sampled, edge_index_i], dim=1)
            edge_type_sampled = torch.cat(
                [edge_type_sampled, edge_type_i], dim=0)
    return edge_index_sampled, edge_type_sampled


def _mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_egde_id):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    topk_egde_id = topk_egde_id.cpu().numpy()
    topk_mask = np.zeros(n_edges, dtype=bool)
    topk_mask[topk_egde_id] = True
    # add another group of random mask
    random_indices = np.random.choice(
        n_edges, size=topk_egde_id.shape[0], replace=False)
    random_mask = np.zeros(n_edges, dtype=bool)
    random_mask[random_indices] = True
    # combine two masks(論文沒有這一步)
    mask = topk_mask | random_mask

    remain_edge_index = edge_index[:, ~mask]
    remain_edge_type = edge_type[~mask]
    masked_edge_index = edge_index[:, mask]
    masked_edge_type = edge_type[mask]

    return remain_edge_index, remain_edge_type, masked_edge_index, masked_edge_type, mask

def _edge_sampling(edge_index, edge_type, samp_rate=0.5):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(
        n_edges, size=int(n_edges * samp_rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices]


def _sparse_dropout(i, v, keep_rate=0.5):
    noise_shape = i.shape[1]

    random_tensor = keep_rate
    # the drop rate is 1 - keep_rate
    random_tensor += torch.rand(noise_shape).to(i.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)

    i = i[:, dropout_mask]
    v = v[dropout_mask] / keep_rate

    return i, v


class RAMVAE(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat, hp_dict=None):
        super(RAMVAE, self).__init__()
        self.args_config = args_config
        self.logger = getLogger()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda else torch.device("cpu")
        
        self.vi = args_config.vi
        self.qk_shared = args_config.qk_shared
        self.ablation = args_config.ab
        self.mae_msize = args_config.mae_msize
        self.tau = args_config.cl_tau
        self.cl_drop = args_config.cl_drop_ratio

        self.samp_func = "torch"

        if args_config.dataset == 'last-fm':
            self.mae_msize = 256
            self.tau = 1.0
            self.cl_drop = 0.5
        elif args_config.dataset == 'alibaba-fashion':
            self.mae_msize = 256
            self.tau = 0.2
            self.cl_drop = 0.5
        elif args_config.dataset == 'movie-lens':
            self.mae_msize = 256
            self.tau = 1.0
            self.cl_drop = 0.5

        
        # update hps
        if hp_dict is not None:
            for k,v in hp_dict.items():
                setattr(self, k, v) #設置屬性k值為v

        self.inter_edge, self.inter_edge_w = self._convert_sp_mat_to_tensor(
            adj_mat)

        self.edge_index, self.edge_type = self._get_edges(graph)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

        self.gcn = AttnVHGCN(
            channel=self.emb_size,
            n_hops=self.context_hops,
            n_users=self.n_users,
            n_relations=self.n_relations,
            qk_shared=self.qk_shared,
            node_dropout_rate=self.node_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate)
        self.contrast_fn = Contrast(self.emb_size, tau=self.tau)
        # self.print_shapes()
        return

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))

    def _convert_sp_mat_to_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return i.to(self.device), v.to(self.device)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        epoch_start = batch['batch_start'] == 0

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        """node dropout"""
        # 1. graph sprasification;
        edge_index, edge_type = _relation_aware_edge_sampling( #沒考慮interaction
            self.edge_index, self.edge_type, self.n_relations, self.node_dropout_rate)
        # 2. compute rationale scores;
        edge_attn_score, edge_attn_logits = self.gcn.norm_attn_computer(
            item_emb, edge_index, edge_type, print=epoch_start, return_logits=True)
        # for adaptive UI MAE
        item_attn_mean_1 = scatter_mean(edge_attn_score, edge_index[0], dim=0, dim_size=self.n_entities) #公式12
        item_attn_mean_1[item_attn_mean_1 == 0.] = 1.
        item_attn_mean_2 = scatter_mean(edge_attn_score, edge_index[1], dim=0, dim_size=self.n_entities)
        item_attn_mean_2[item_attn_mean_2 == 0.] = 1.
        item_attn_mean = (0.5 * item_attn_mean_1 + 0.5 * item_attn_mean_2)[:self.n_items]
        # for adaptive MAE training
        std = torch.std(edge_attn_score).detach()
        noise = -torch.log(-torch.log(torch.rand_like(edge_attn_score)))
        edge_attn_score = edge_attn_score + noise #加上gumbel noise
        topk_v, topk_attn_edge_id = torch.topk(
            edge_attn_score, self.mae_msize, sorted=False)
        top_attn_edge_type = edge_type[topk_attn_edge_id]

        enc_edge_index, enc_edge_type, masked_edge_index, masked_edge_type, mask_bool = _mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_attn_edge_id)

        inter_edge, inter_edge_w = _sparse_dropout(
            self.inter_edge, self.inter_edge_w, self.node_dropout_rate)

        # rec task
        entity_gcn_emb, user_gcn_emb, entity_gcn_emb_sample, user_gcn_emb_sample, entity_gcn_emb_mu, entity_gcn_emb_logvar = self.gcn(
            user_emb, item_emb,
            enc_edge_index, enc_edge_type,
            inter_edge, inter_edge_w,
            stochastic=self.vi, mess_dropout=self.mess_dropout)

        # BPR
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        loss, rec_loss, reg_loss = self.create_bpr_loss(u_e, pos_e, neg_e)

        mae_loss = None
        kl_loss = None
        if self.vi:
            node_pair_emb_sample = entity_gcn_emb_sample[masked_edge_index.t()]
            masked_edge_emb = self.gcn.relation_emb[masked_edge_type-1]
            mae_loss = self.create_mae_loss(node_pair_emb_sample, masked_edge_emb)
            kl_loss = self.create_kl_loss(entity_gcn_emb_mu, entity_gcn_emb_logvar)
        else:
            node_pair_emb = entity_gcn_emb[masked_edge_index.t()]
            masked_edge_emb = self.gcn.relation_emb[masked_edge_type-1]
            mae_loss = self.create_mae_loss(node_pair_emb, masked_edge_emb)

        # CL task
        """adaptive sampling"""
        cl_kg_edge, cl_kg_type = _adaptive_kg_drop_cl(
            edge_index, edge_type, edge_attn_score, keep_rate=1-self.cl_drop)
        cl_ui_edge, cl_ui_w = _adaptive_ui_drop_cl(
            item_attn_mean, inter_edge, inter_edge_w, 1-self.cl_drop, samp_func=self.samp_func)

        item_agg_ui = self.gcn.forward_ui(
            user_emb, item_emb[:self.n_items], cl_ui_edge, cl_ui_w)
        item_agg_kg = self.gcn.forward_kg(
            item_emb, cl_kg_edge, cl_kg_type)[:self.n_items]
        cl_loss = self.contrast_fn(item_agg_ui, item_agg_kg)
        return loss, mae_loss, kl_loss, cl_loss

    def calc_topk_attn_edge(self, entity_emb, edge_index, edge_type, k): #沒用到
        edge_attn_score = self.gcn.norm_attn_computer(
            entity_emb, edge_index, edge_type, return_logits=True)
        positive_mask = edge_attn_score > 0
        edge_attn_score = edge_attn_score[positive_mask]
        edge_index = edge_index[:, positive_mask]
        edge_type = edge_type[positive_mask]
        topk_values, topk_indices = torch.topk(
            edge_attn_score, k, sorted=False)
        return edge_index[:, topk_indices], edge_type[topk_indices]

    def generate(self): 
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(user_emb,
                        item_emb,
                        self.edge_index,
                        self.edge_type,
                        self.inter_edge,
                        self.inter_edge_w,
                        stochastic=self.vi,
                        mess_dropout=False)[:2]

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    # @TimeCounter.count_time(warmup_interval=4)
    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        if torch.isnan(mf_loss):
            raise ValueError("nan mf_loss")

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def create_mae_loss(self, node_pair_emb, masked_edge_emb=None): 
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        if masked_edge_emb is not None:
            pos1 = tail_embs * masked_edge_emb
        else:
            pos1 = tail_embs
        scores = - torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        return scores
    
    def create_kl_loss(self, entity_gcn_emb_mu, entity_gcn_emb_logvar):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + entity_gcn_emb_logvar - entity_gcn_emb_mu.pow(2) - entity_gcn_emb_logvar.exp(),dim=1),dim=0)
        return kl_loss

    def print_shapes(self):
        self.logger.info("########## Ablation ##########")
        self.logger.info("ablation: {}".format(self.ablation))
        self.logger.info("########## Model HPs ##########")
        self.logger.info("tau: {}".format(self.contrast_fn.tau))
        self.logger.info("cL_drop: {}".format(self.cl_drop))
        self.logger.info("mae_msize: {}".format(self.mae_msize))
        self.logger.info("########## Model Parameters ##########")
        self.logger.info("context_hops: %d", self.context_hops)
        self.logger.info("node_dropout: %d", self.node_dropout)
        self.logger.info("node_dropout_rate: %.1f", self.node_dropout_rate)
        self.logger.info("mess_dropout: %d", self.mess_dropout)
        self.logger.info("mess_dropout_rate: %.1f", self.mess_dropout_rate)
        self.logger.info('all_embed: {}'.format(self.all_embed.shape))
        self.logger.info('interact_mat: {}'.format(self.inter_edge.shape))
        self.logger.info('edge_index: {}'.format(self.edge_index.shape))
        self.logger.info('edge_type: {}'.format(self.edge_type.shape))

    def generate_kg_drop(self): #沒用到
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        edge_index, edge_type = _edge_sampling(
        self.edge_index, self.edge_type, self.kg_drop_test_keep_rate)
        return self.gcn(user_emb,
                        item_emb,
                        edge_index,
                        edge_type,
                        self.inter_edge,
                        self.inter_edge_w,
                        stochastic=self.vi,
                        mess_dropout=False)[:2]
    
    def generate_global_attn_score(self): #沒用到
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        edge_attn_score = self.gcn.norm_attn_computer(
            item_emb, self.edge_index, self.edge_type)

        return edge_attn_score, self.edge_index, self.edge_type