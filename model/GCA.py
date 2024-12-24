import torch
import torch.nn as nn
import torch.nn.functional as F  


class graphic_cls_aggregation(nn.Module):
    def __init__(self, dim_in=768, dim_hidden=768, dim_out=768, topk=6, agg_type='bi-interaction', dropout=0., pool='attn'):
        super().__init__()

        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.LeakyReLU())
        self.message_dropout = nn.Dropout(dropout)

        
        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)

        self.scale = dim_hidden ** -0.5
        self.topk = topk
        self.agg_type = agg_type

        self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)

        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_out)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_out)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_out)
            self.linear2 = nn.Linear(dim_hidden, dim_out)
        else:
            raise NotImplementedError
        
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim_out)

    def forward(self, cls_tokens,feats, tok_ratio=0.5):
    # def forward(self, x):
        topk = int(tok_ratio * cls_tokens.shape[1])
        x = torch.concat([cls_tokens,feats.clone().detach()],dim=1)

        e_h = self.W_head(cls_tokens)
        e_t = self.W_tail(x)

        # construct neighbour
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # 1
        topk_weight, topk_index = torch.topk(attn_logit, k=topk, dim=-1)

        # add an extra dimension to the index tensor, making it available for advanced indexing, aligned with the dimensions of e_t
        topk_index = topk_index.to(torch.long)

        # expand topk_index dimensions to match e_t
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # shape: [1, 10000, 4]

        # create a RANGE tensor to help indexing
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # shape: [1, 1, 1]

        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # shape: [1, 10000, 4, 512]

        # use SoftMax to obtain probability
        topk_prob = F.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))  # 1 pixel wise   2 matmul

        # gated knowledge attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1((e_h + e_Nh) * 0.1 + cls_tokens))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh * 0.1 + cls_tokens))
            embedding = sum_embedding + bi_embedding
        else:
            embedding = e_Nh

        h = self.message_dropout(embedding)
        h = self.norm(embedding)
    
        return h
            
if __name__ == "__main__":
    data = torch.randn((4, 784, 768)).cuda()
    token = torch.randn((4, 20, 768)).cuda()
    model = graphic_cls_aggregation(dim_in=768, dim_hidden=768, dim_out=768, topk=6, agg_type='bi-interaction', dropout=0.3, pool='attn').cuda()
    output = model(token, data)
    print(output.shape)
