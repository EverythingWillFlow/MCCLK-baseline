import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum

class Aggregator(nn.Module):
    def __init__(self, n_users):
        super(Aggregator, self).__init__()
        #print("into aggregator init ")
        self.n_users = n_users

    def forward(self, entity_emb, user_emb,
                edge_index, edge_type, interact_mat,
                weight):
        #print("into aggregator forward ")
        torch.cuda.empty_cache()
            # Add safety checks
        head, tail = edge_index
        head = torch.clamp(head, 0, entity_emb.shape[0] - 1)
        tail = torch.clamp(tail, 0, entity_emb.shape[0] - 1)
        edge_index = torch.stack([head, tail])
        
        # Ensure edge_type indices are valid
        edge_type = torch.clamp(edge_type - 1, 0, weight.shape[0] - 1)

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]

        # # ------------calculate attention weights ---------------
        # neigh_relation_emb_weight = self.calculate_sim_hrt(entity_emb[head], entity_emb[tail], weight[edge_type - 1])
        # neigh_relation_emb_weight = neigh_relation_emb_weight.expand(neigh_relation_emb.shape[0],
        #                                                              neigh_relation_emb.shape[1])
        # neigh_relation_emb = neigh_relation_emb * neigh_relation_emb_weight.unsqueeze(-1)

        neigh_relation_emb_weight = self.calculate_sim_hrt_blocked(entity_emb[head], entity_emb[tail], weight[edge_type - 1])
        # 应该是 [N] 或 [N, 1]
        if len(neigh_relation_emb_weight.shape) == 2:
            neigh_relation_emb_weight = neigh_relation_emb_weight.squeeze(-1)

        neigh_relation_emb = neigh_relation_emb * neigh_relation_emb_weight.unsqueeze(-1)



        # neigh_relation_emb_tmp = torch.matmul(neigh_relation_emb_weight, neigh_relation_emb)
        # neigh_relation_emb_weight = scatter_softmax(neigh_relation_emb_weight, index=head, dim=0)
        # neigh_relation_emb = torch.mul(neigh_relation_emb_weight, neigh_relation_emb)

        # 注意：再次加权也应 unsqueeze
        soft_weights = scatter_softmax(neigh_relation_emb_weight, index=head, dim=0)  # [N]
        neigh_relation_emb = neigh_relation_emb * soft_weights.unsqueeze(-1)          # [N, D]

        entity_agg = scatter_sum(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        user_agg = torch.sparse.mm(interact_mat, entity_emb)
        # user_agg = user_agg + user_emb * user_agg
        score = torch.mm(user_emb, weight.t())
        score = torch.softmax(score, dim=-1)
        user_agg = user_agg + (torch.mm(score, weight)) * user_agg

        return entity_agg, user_agg

    def calculate_sim_hrt_blocked(self, head_emb, tail_emb, rel_emb, block_size=500000):
        weights = []
        total = head_emb.shape[0]
        for i in range(0, total, block_size):
            h = head_emb[i:i+block_size]
            t = tail_emb[i:i+block_size]
            r = rel_emb[i:i+block_size]
            w = self.calculate_sim_hrt(h, t, r).detach()  # or keep grad if needed
            weights.append(w)
        return torch.cat(weights, dim=0)

    def calculate_sim_hrt(self, entity_emb_head, entity_emb_tail, relation_emb):

        tail_relation_emb = entity_emb_tail * relation_emb
        tail_relation_emb = tail_relation_emb.norm(dim=1, p=2, keepdim=True)
        head_relation_emb = entity_emb_head * relation_emb
        head_relation_emb = head_relation_emb.norm(dim=1, p=2, keepdim=True)
        att_weights = torch.matmul(head_relation_emb.unsqueeze(dim=1), tail_relation_emb.unsqueeze(dim=2)).squeeze(dim=-1)
        att_weights = att_weights ** 2
        return att_weights

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                  n_relations, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        #print("into GRAPh Conv~~")
        super(GraphConv, self).__init__()
        
        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
                # 必须使用interact_mat的真实维度
        self.n_entities = interact_mat.shape[1]  # 必须=136499
        self.n_users = n_users
        self.n_items = self.n_entities - self.n_users  # 新增属性
        
        # 调试输出
        # print(f"图卷积维度终极验证:")
        # print(f"交互矩阵列数: {self.n_entities}")
        # print(f"接收用户数: {self.n_users}")
        # print(f"计算项目数: {self.n_items}")

        self.n_entities = interact_mat.shape[1]  # 不减去用户数
        self.n_users = n_users
        
        # print(f"图卷积维度终极验证:")
        # print(f"接收的实体总数: {self.n_entities}")
        # print(f"接收的用户数: {self.n_users}")
       
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.topk = 10
        self.lambda_coeff = 0.5
        self.temperature = 0.2
        self.device = torch.device("cuda:" + str(0))
        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]
        #print("into n_hops~")
        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout
 


        
   


    def _edge_sampling(self, edge_index, edge_type, rate=0.2):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                interact_mat, mess_dropout=True, node_dropout=False):

              # Print dimensions for debugging
        # print(f"entity_emb shape: {entity_emb.shape}")
        # print(f"expected shape: ({self.n_entities}, {entity_emb.shape[1]})")
        #  # Debug dimension information
        # print(f"Forward dimensions check:")
        # print(f"entity_emb shape: {entity_emb.shape}")
        # print(f"expected entity shape: ({self.n_entities - self.n_users}, {entity_emb.shape[1]})")
        # print(f"user_emb shape: {user_emb.shape}")
        # print(f"expected user shape: ({self.n_users}, {user_emb.shape[1]})")
        
        # Ensure entity_emb has correct dimensions


        
        # Debug information
        # print(f"Forward pass dimensions:")
        # print(f"user_emb: {user_emb.shape}")
        # print(f"entity_emb: {entity_emb.shape}")
        # print(f"expected entity_emb: ({self.n_items}, {entity_emb.shape[1]})")
        # Move tensors to correct device
        entity_emb = entity_emb.to(self.device)
        user_emb = user_emb.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        interact_mat = interact_mat.to(self.device)
   

        # Rest of the forward method remains the same
        # if node_dropout:
        #     edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
        #     interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)
        # # 修正后断言：应该匹配总实体数
 
        edge_type = torch.clamp(edge_type, 0, self.n_relations-1)
   #     print(f"预期实体维度: ({self.n_entities}, {entity_emb.shape[1]})")  # 使用n_entities而非n_items

        # 修正后断言（使用总实体数）：
        # print(f"预期实体维度: ({self.n_entities}, {entity_emb.shape[1]})")
        assert entity_emb.shape[0] == self.n_entities, \
            f"实体维度错误：预期{self.n_entities}，实际{entity_emb.shape[0]}"

        # print("减小构建维度~~~~")
        item_emb = entity_emb[self.n_users:]  # 只取 item
        origin_item_adj = self.build_adj(item_emb, self.topk)
        # print("减小构建维度执行完毕~~~~")

        
        # ----------------build item-item graph-------------------
        #origin_item_adj = self.build_adj(entity_emb, self.topk)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        # print("进入conv~ for循环")
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight)
            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
        # print("退出conv~ for循环")
        # update item-item graph
        # entity_res_emb: [n_entities, D] = users + items
# 只构建项目部分的相似图
        item_emb = entity_res_emb[self.n_users:]               # [90580, D]
        new_item_adj = self.build_adj(item_emb, self.topk)     # [90580, 90580]
        item_adj = (1 - self.lambda_coeff) * new_item_adj + self.lambda_coeff * origin_item_adj

        # item_adj = (1 - self.lambda_coeff) * self.build_adj(entity_res_emb,
        #            self.topk) + self.lambda_coeff * origin_item_adj
        # item_emb = entity_res_emb[self.n_users:]  # 提取项目嵌入
        # item_adj = (1 - self.lambda_coeff) * self.build_adj(item_emb, self.topk) + \
        #    self.lambda_coeff * origin_item_adj
        # print(f"[Adj Update Done]")

        return entity_res_emb, user_res_emb, item_adj

    # def build_adj(self, context, topk):
    #     # construct similarity adj matrix
    #     n_entity = context.shape[0]
    #     context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True)).cpu()
    #     sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    #     knn_val, knn_ind = torch.topk(sim, topk, dim=-1)
    #     # adj_matrix = (torch.zeros_like(sim)).scatter_(-1, knn_ind, knn_val)
    #     knn_val, knn_ind = knn_val.to(self.device), knn_ind.to(self.device)

    #     y = knn_ind.reshape(-1)
    #     x = torch.arange(0, n_entity).unsqueeze(dim=-1).to(self.device)
    #     x = x.expand(n_entity, topk).reshape(-1)
    #     indice = torch.cat((x.unsqueeze(dim=0), y.unsqueeze(dim=0)), dim=0)
    #     value = knn_val.reshape(-1)
    #     adj_sparsity = torch.sparse.FloatTensor(indice.data, value.data, torch.Size([n_entity, n_entity])).to(self.device)

    #     # normalized laplacian adj
    #     rowsum = torch.sparse.sum(adj_sparsity, dim=1)
    #     d_inv_sqrt = torch.pow(rowsum, -0.5)
    #     d_mat_inv_sqrt_value = d_inv_sqrt._values()
    #     x = torch.arange(0, n_entity).unsqueeze(dim=0).to(self.device)
    #     x = x.expand(2, n_entity)
    #     d_mat_inv_sqrt_indice = x
    #     d_mat_inv_sqrt = torch.sparse.FloatTensor(d_mat_inv_sqrt_indice, d_mat_inv_sqrt_value, torch.Size([n_entity, n_entity]))
    #     L_norm = torch.sparse.mm(torch.sparse.mm(d_mat_inv_sqrt, adj_sparsity), d_mat_inv_sqrt)
    #     return L_norm
    def build_adj(self, context, topk):
        """
        构建 item-item TopK 稀疏相似度图，避免全矩阵爆炸。
        """
        context = F.normalize(context, p=2, dim=1)  # [N, D]
        n_item = context.shape[0]
        device = context.device

        # 分批构建 TopK 邻接（避免一次性 OOM）
        knn_ind_list = []
        knn_val_list = []
        batch_size = 1024

        for i in range(0, n_item, batch_size):
            end = min(i + batch_size, n_item)
            batch = context[i:end]  # [B, D]
            sim = torch.matmul(batch, context.T)  # [B, N]
            knn_val, knn_ind = torch.topk(sim, topk)  # [B, topk]
            knn_ind_list.append(knn_ind)
            knn_val_list.append(knn_val)

        knn_ind = torch.cat(knn_ind_list, dim=0)  # [N, topk]
        knn_val = torch.cat(knn_val_list, dim=0)  # [N, topk]

        # 构建稀疏邻接矩阵
        x = torch.arange(n_item).unsqueeze(1).expand(-1, topk).reshape(-1).to(device)
        y = knn_ind.reshape(-1)
        values = knn_val.reshape(-1)
        indices = torch.stack([x, y])

        adj = torch.sparse_coo_tensor(indices, values, (n_item, n_item), device=device)
        return adj.coalesce()




class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        #print("into init~~~")
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.emb_size = args_config.dim
      #assert self.n_nodes == self.n_entities, f"节点数({self.n_nodes})应等于总实体数({self.n_entities})"

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity

        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                  else torch.device("cpu")
        self.adj_mat = adj_mat
        self.n_entities = adj_mat.shape[1]  # 必须=136499
        self.n_users = data_config['n_users']
        
        # 强制覆盖所有配置参数
        self.n_items = self.n_entities - self.n_users  # 136499-45919=90580
        self.n_nodes = self.n_entities
        
                # 重建正确数据配置
        data_config['n_entities'] = self.n_entities
        data_config['n_items'] = self.n_items
        data_config['n_nodes'] = self.n_entities  

        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.gcn = self._init_model()
        self.lightgcn_layer = 2
        self.n_item_layer = 1
        self.alpha = 0.2
        self.user_embedding = nn.Embedding(self.n_users, self.emb_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.emb_size)
        self.fc1 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        self.fc2 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        self.fc3 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        self.entity_emb = nn.Parameter(torch.empty(self.n_entities, self.emb_size))

                # 关键修改：从交互矩阵获取真实实体总数
# 从邻接矩阵获取真实维度

         # 验证核心参数
        # print(f"参数终极验证:")
        # print(f"adj_mat列数: {self.adj_mat.shape[1]}")
        # print(f"用户数: {self.n_users}")
        # print(f"真实项目数: {self.n_items}")
        # print(f"总实体数: {self.n_entities}")



    def _init_weight(self):
        #print("into weight~~~")
        #initializer = nn.init.xavier_uniform_
        #self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        #self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        #print("into weight~~~")
        initializer = nn.init.xavier_uniform_
        # 使用修正后的维度初始化
        self.all_embed = initializer(torch.empty(self.n_entities, self.emb_size))
        
        # print(f"嵌入矩阵终极验证:")
        # print(f"实际维度: {self.all_embed.shape}")
        # print(f"预期维度: ({self.n_entities}, {self.emb_size})")
        assert self.all_embed.shape[0] == self.n_entities, "嵌入矩阵初始化错误"


    
    # 确保交互矩阵维度正确
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        #print("into model~~~")
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(
        self,
        batch=None,
                ):
        #print("into forward~~~")
        
        user = batch['users']
        item = batch['items']
        labels = batch['labels']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        
        # 实体嵌入（完整传递）
        entity_emb = self.all_embed  # 不进行任何切片
        
        # print(f"运行时维度验证:")
        # print(f"user_emb: {user_emb.shape} (应={self.n_users})")
        # print(f"entity_emb: {entity_emb.shape} (应={self.n_entities})")
        
        # 传递给图卷积
        entity_gcn_emb, user_gcn_emb, item_adj = self.gcn(
            user_emb,
            entity_emb,  # 传递完整嵌入
            self.edge_index,
            self.edge_type,
            self.interact_mat
        )
        # print(f"entity_gcn_emb.shape: {entity_gcn_emb.shape}, expected: ({self.n_entities}, {self.emb_size})")
        
        u_e = user_gcn_emb[user]
        i_e = entity_gcn_emb[item]
        
        i_h = item_emb
        for i in range(self.n_item_layer):
            i_h = torch.sparse.mm(item_adj, i_h)
        i_h = F.normalize(i_h, p=2, dim=1)
        i_e_1 = i_h[item]

        interact_mat_new = self.interact_mat
        indice_old = interact_mat_new._indices()
        value_old = interact_mat_new._values()
        x = indice_old[0, :]
        y = indice_old[1, :]
        x_A = x
        y_A = y + self.n_users
        x_A_T = y + self.n_users
        y_A_T = x
        x_new = torch.cat((x_A, x_A_T), dim=-1)
        y_new = torch.cat((y_A, y_A_T), dim=-1)
        indice_new = torch.cat((x_new.unsqueeze(dim=0), y_new.unsqueeze(dim=0)), dim=0)
        value_new = torch.cat((value_old, value_old), dim=-1)
        interact_graph = torch.sparse.FloatTensor(indice_new, value_new, torch.Size([self.n_users + self.n_entities, self.n_users + self.n_entities]))
        

        user_emb = self.user_embedding.weight
        item_emb = self.entity_embedding.weight
        #entity_emb = torch.cat([user_emb, item_emb], dim=0)

        # graph convolution
        user_gcn_emb, item_gcn_emb = self.light_gcn(user_emb, item_emb, interact_graph)

# # 取出 batch 的嵌入来计算 loss
# batch_user_emb = user_gcn_emb[batch_user_ids]  # shape = (batch_size, dim)
# batch_item_emb = item_gcn_emb[batch_item_ids]

       # user_lightgcn_emb, item_lightgcn_emb = self.light_gcn(user_emb, item_emb, interact_graph)
        u_e_2 = user_gcn_emb[user]
        i_e_2 = item_gcn_emb[item]
        # batch_user_emb = user_lightgcn_emb[batch['users']]       # shape: (batch_size, dim)
        # batch_item_emb = item_lightgcn_emb[batch['pos_items']]   # or neg_items 等

       
        # # loss_contrast = 0
        # loss_contrast = self.alpha * self.calculate_loss(i_e_1, i_e_2)
        # # i_e_1 = i_e_1 + i_e_2
        # loss_contrast = loss_contrast + ((1-self.alpha)/2)*self.calculate_loss_1(i_e_2, i_e)
        # loss_contrast = loss_contrast + ((1-self.alpha)/2)*self.calculate_loss_2(u_e_2, u_e)
        #
        # u_e = torch.cat((u_e, u_e), dim=-1)
        # i_e = torch.cat((i_e, i_e_1), dim=-1)
        # i_e_1 = i_e_1 + i_e_2
        item_1 = item_emb[item]
        user_1 = user_emb[user]
        loss_contrast = self.calculate_loss(i_e_1, i_e_2)
        loss_contrast = loss_contrast + self.calculate_loss_1(item_1, i_e_2)
        loss_contrast = loss_contrast + self.calculate_loss_2(user_1, u_e_2)

        # u_e = torch.cat((u_e, u_e_2, u_e_2), dim=-1)
        # i_e = torch.cat((i_e, i_e_1, i_e_2), dim=-1)
        u_e = (u_e + u_e_2 * 2) / 3
        i_e = (i_e + i_e_1 + i_e_2) / 3

        return self.create_bpr_loss(u_e, i_e, labels, loss_contrast)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def calculate_loss(self, A_embedding, B_embedding):
        # first calculate the sim rec
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc1(A_embedding)
        B_embedding = self.fc1(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        # refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        # between_sim_1 = f(self.sim(B_embedding, A_embedding))
        # loss_2 = -torch.log(
        #     between_sim_1.diag()
        #     / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        # ret = (loss_1 + loss_2) * 0.5
        ret = loss_1
        ret = ret.mean()
        return ret

    def calculate_loss_1(self, A_embedding, B_embedding):
        # first calculate the sim rec
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc2(A_embedding)
        B_embedding = self.fc2(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret

    def calculate_loss_2(self, A_embedding, B_embedding):
        # first calculate the sim rec
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc3(A_embedding)
        B_embedding = self.fc3(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret

    def light_gcn(self, user_embedding, item_embedding, adj):
        # print("into light gcn~~~")
        ego_embeddings = torch.cat((user_embedding, item_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        # print(adj.shape)  # 确保 adj 的形状是 (num_nodes, num_nodes)
        # print(ego_embeddings.shape)  # 确保 ego_embeddings 的形状是 (num_nodes, embedding_dim)



        for i in range(self.lightgcn_layer):
            assert adj.shape[0] == ego_embeddings.shape[0], "邻接矩阵和嵌入矩阵的节点数不匹配"
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_entities], dim=0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, items, labels, loss_contrast):
        batch_size = users.shape[0]
        scores = (items * users).sum(dim=1)
        scores = torch.sigmoid(scores)
        criteria = nn.BCELoss()
        bce_loss = criteria(scores, labels.float())
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # cor_loss = self.sim_decay * cor
        return bce_loss + emb_loss + 0.001*loss_contrast, scores, bce_loss, emb_loss

    # def generate(self):
    #     user_emb = self.all_embed[:self.n_users, :]
    #     item_emb = self.all_embed[self.n_users:, :]
    #     entity_gcn_emb, user_gcn_emb, item_adj = self.gcn(user_emb,
    #                                                       item_emb,
    #                                                       self.edge_index,
    #                                                       self.edge_type,
    #                                                       self.interact_mat,
    #                                                       mess_dropout=self.mess_dropout,
    #                                                       node_dropout=self.node_dropout)
    #
    #     interact_mat_new = torch.sparse.mm(self.interact_mat, item_adj)
    #     indice_old = interact_mat_new._indices()
    #     value_old = interact_mat_new._values()
    #     x = indice_old[0, :]
    #     y = indice_old[1, :]
    #     x_A = x
    #     y_A = y + self.n_users
    #     x_A_T = y + self.n_users
    #     y_A_T = x
    #     x_new = torch.cat((x_A, x_A_T), dim=-1)
    #     y_new = torch.cat((y_A, y_A_T), dim=-1)
    #     indice_new = torch.cat((x_new.unsqueeze(dim=0), y_new.unsqueeze(dim=0)), dim=0)
    #     value_new = torch.cat((value_old, value_old), dim=-1)
    #     interact_graph = torch.sparse.FloatTensor(indice_new, value_new, torch.Size(
    #         [self.n_users + self.n_entities, self.n_users + self.n_entities]))
    #     user_lightgcn_emb, item_lightgcn_emb = self.light_gcn(user_emb, item_emb, interact_graph)
    #     u_e = torch.cat((user_gcn_emb, user_lightgcn_emb), dim=-1)
    #     i_e = torch.cat((entity_gcn_emb, item_lightgcn_emb), dim=-1)
    #     return i_e, u_e
