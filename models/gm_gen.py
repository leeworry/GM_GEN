import sys
import ipdb
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import nn
from torch.nn import functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q.unsqueeze(2) / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v).sum(2)

        return output, attn

class BiLinearAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, n_head, k_dim, temperature=1.0, attn_dropout=0.):
        super().__init__()
        self.bilinear = nn.Bilinear(k_dim,k_dim,n_head)
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        B, N, Q, K, h_size = k.size(0), k.size(1), q.size(2), k.size(2), k.size(3),
        q = q.view(B,N,1,h_size).expand(B,N,K,h_size).reshape(-1,h_size)
        k = k.view(-1,h_size)
        attn = self.bilinear(q,k).view(B,N,-1,K)
        # attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) # 4, 5 , 1, 5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v).sum(2)

        return output, attn

class BiAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, k_dim, n_hid, temperature=1.0, dropout=0.1):
        super().__init__()
        self.bilinear = nn.Bilinear(k_dim, k_dim, n_hid)
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, mask=None):
        attn = self.bilinear(inp, inp).mean(-1)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1).unsqueeze(-2)
        output = torch.matmul(attn, inp).sum(-2)
        return output, attn
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, d_k, bias=False)
        self.w_ks = nn.Linear(d_model, d_k, bias=False)
        # self.w_vs = nn.Linear(d_model, d_v, bias=False)
        self.attention_bi = BiLinearAttention(n_head, d_k, temperature=1.0, attn_dropout=dropout)#d_k ** 0.5
        self.attention_dot = ScaledDotProductAttention(temperature=1.0)# d_k ** 0.5

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, attn='dot', mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_kk, len_v = q.size(0), q.size(1), k.size(1), k.size(2), v.size(1)

        q_1 = self.w_qs(q).view(sz_b, len_q, d_k)
        q_2 = q.view(sz_b, len_q, -1)

        k = self.w_ks(k).view(sz_b, len_k, len_kk, d_k) # 1*5*5*768  1, 5, 5, 512
        v = v.view(sz_b, len_k, len_kk, -1) # 1*5*5*768  1, 5, 5, 512

        if mask is not None:
            mask = mask.unsqueeze(1)
        if attn=='bi':
            v, attn = self.attention_bi(q_1, k, v, mask=mask)
        else:
            v, attn = self.attention_dot(q_1, k, v, mask=mask)
        return v, q_2


class Gm_Gen(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, dot=False, relation_encoder=None, N=5, Q=1, head=2, hid=1536, d_k=1536, struct_hid=32, attn_hid=32, dropout=0.):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.fc = nn.Linear(d_k, d_k)
        self.struct = BiAttention(hid,struct_hid)
        self.drop = nn.Dropout(dropout)
        self.dot = dot
        self.attn = BiAttention(hid, attn_hid)
        
        self.relation_encoder = relation_encoder
        self.hidden_size = hid

    def __pred__(self, w, x, dim):
        if self.dot:
            return (w * x).sum(dim)
        else:
            return -(torch.pow(w - x, 2)).sum(dim)

    def pred(self, S, Q):
        return self.__pred__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, rel_txt, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''

        rel_gol, rel_loc = self.sentence_encoder(rel_txt, cat=False)
        rel_loc = torch.mean(rel_loc[:,1:,:], 1) #[B*N, D]
        rel_rep = torch.cat((rel_gol, rel_loc), -1).view(-1, N,1, rel_gol.shape[1]*2)

        support_h, support_t,  s_loc = self.sentence_encoder(support)
        query_h, query_t,  q_loc = self.sentence_encoder(query)

        support_emb = torch.cat((support_h, support_t), -1)
        query_emb = torch.cat((query_h, query_t), -1)
        support = self.drop(support_emb)
        query = self.drop(query_emb)

        support = support.view(-1, N, K, self.hidden_size)
        query = query.view(-1, total_Q, 1, self.hidden_size)
        B = support.size(0)

        if K==1:
            support = torch.cat([support, rel_rep], dim=-2)
            support1 = support.mean(2)
            query1 = query.view(B,total_Q,-1)
            inp = self.fc(support).view(B * N, K+1, -1)
            support2, _ = self.struct(inp)
            support2 = support2.view(B,N,-1)
            query2 = self.fc(query).view(B,total_Q,-1)
        else:
            self.dot=False
            rel_rep = rel_rep.view(B, N, 1, -1)
            support = torch.cat([support, rel_rep], dim=-2)
            support1 = support.mean(2)

            query1 = query.view(B, total_Q, -1)
            inp = self.fc(support).view(B * N, K + 1, -1)
            support2, _ = self.attn(inp)
            support2 = support2.view(B, total_Q, -1)
            query2 = self.fc(query).view(B, N, -1)

        final_support = torch.cat([support1,support2],dim=-1)
        final_query = torch.cat([query1,query2],dim=-1)

        logits = self.pred(final_support, final_query)
        # logits = self.pred(support1, query1)
        # logits = self.pred(support2, query2)

        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        return logits, pred

    def generate_weight(self, support, query, rel_txt, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''

        rel_gol, rel_loc = self.sentence_encoder(rel_txt, cat=False)
        rel_loc = torch.mean(rel_loc[:,1:,:], 1) #[B*N, D]
        rel_rep = torch.cat((rel_gol, rel_loc), -1).view(-1, N,1, rel_gol.shape[1]*2)

        support_h, support_t,  s_loc = self.sentence_encoder(support)
        query_h, query_t,  q_loc = self.sentence_encoder(query)

        support_emb = torch.cat((support_h, support_t), -1)
        query_emb = torch.cat((query_h, query_t), -1)
        support = self.drop(support_emb)
        query = self.drop(query_emb)

        support = support.view(-1, N, K, self.hidden_size)
        query = query.view(-1, total_Q, 1, self.hidden_size)
        B = support.size(0)
        temp_support1 = torch.cat([support_h, support_t], -1).view(B,N*K,-1)
        temp_support2 = self.fc(temp_support1).view(B,N*K,-1)
        temp_support = torch.cat([temp_support1,temp_support2],dim=-1)
        if K==1:
            support = torch.cat([support, rel_rep], dim=-2)
            support1 = support.mean(2)
            query1 = query.view(B,total_Q,-1)
            inp = self.fc(support).view(B * N, K+1, -1)
            support2, _ = self.struct(inp)
            support2 = support2.view(B,N,-1)
            query2 = self.fc(query).view(B,total_Q,-1)
        else:
            self.dot = False
            rel_rep = rel_rep.view(B, N, 1, -1)
            support = torch.cat([support, rel_rep], dim=-2)
            support1 = support.mean(2)

            query1 = query.view(B, total_Q, -1)
            inp = self.fc(support).view(B * N, K + 1, -1)
            support2, _ = self.attn(inp)
            support2 = support2.view(B, N, -1)
            query2 = self.fc(query).view(B, total_Q, -1)

        final_support = torch.cat([support1,support2],dim=-1)
        final_query = torch.cat([query1,query2],dim=-1)
        # logits = self.pred(final_support, final_query)
        # logits = self.pred(support2, query2)

        # minn, _ = logits.min(-1)
        # logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)
        # _, pred = torch.max(logits.view(-1, N + 1), 1)
        return final_support.detach(), temp_support.detach(), final_query.detach()

    # def generate_input(self, support, query, rel_txt, N, K, total_Q):
    #     '''
    #     support: Inputs of the support set.
    #     query: Inputs of the query set.
    #     N: Num of classes
    #     K: Num of instances for each class in the support set
    #     Q: Num of instances in the query set
    #     '''
    #     query_h, query_t, q_loc = self.sentence_encoder(query)  # (B * total_Q, D)
    #
    #     query = torch.cat((query_h, query_t), -1)
    #     query = query.view(-1, total_Q, self.hidden_size)  # (B, total_Q, D)
    #     B = query.shape[0]
    #
    #     query1 = query.view(B, total_Q, -1)
    #     query2 = self.fc(query).view(B, total_Q, -1)
    #     query = torch.cat([query1,query2],-1)
    #     return query