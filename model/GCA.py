import torch
import torch.nn as nn
import torch.nn.functional as F  


class graphic_cls_aggregation(nn.Module):
    def __init__(self, in_dim=768, mid_dim=768, out_dim=768, dropout=0.,):
        super().__init__()

        self._fc1 = nn.Sequential(nn.Linear(in_dim, mid_dim), nn.LeakyReLU())

        self.head_proj = nn.Linear(mid_dim, mid_dim)
        self.tail_proj = nn.Linear(mid_dim, mid_dim)
        self.scale = mid_dim ** -0.5

        self.linear1 = nn.Linear(mid_dim, out_dim)
        self.linear2 = nn.Linear(mid_dim, out_dim)
        
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, cls_tokens,feats, tok_ratio=0.5):
        topk = int(tok_ratio * cls_tokens.shape[1])
        x = torch.concat([cls_tokens,feats.clone().detach()],dim=1)

        # build head and tail
        e_h = self.head_proj(cls_tokens)
        e_t = self.tail_proj(x)

        # construct neighbour
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # 1
        weight_topk, index_topk = torch.topk(attn_logit, k=topk, dim=-1)
        index_topk = index_topk.to(torch.long)
        index_topk_expanded = index_topk.expand(e_t.size(0), -1, -1)
        
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(index_topk.device) 
        Nb_h = e_t[batch_indices, index_topk_expanded, :] 

        # SoftMax to generate probability
        topk_prob = F.softmax(weight_topk, dim=2)
        eh_r = torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2)) + torch.mul(topk_prob.unsqueeze(-1), Nb_h)

        # gated knowledge attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        ### bidirectional aggregate
        sum_embedding = self.activation(self.linear1((e_h + e_Nh) * 0.1 + cls_tokens))
        bi_embedding = self.activation(self.linear2(e_h * e_Nh * 0.1 + cls_tokens))
        embedding = sum_embedding + bi_embedding

        mctokens_enhanced = self.norm(self.dropout(embedding))
    
        return mctokens_enhanced
