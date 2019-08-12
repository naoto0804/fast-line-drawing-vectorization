from typing import List

import numpy as np
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: List[int], d_k: int, d_v: int,
                 dropout: float = 0.1):
        super().__init__()

        d_model = np.prod(d_model)

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    # def forward(self, q, k, v, mask=None):
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q,
                                                    d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k,
                                                    d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v,
                                                    d_v)  # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        # output, attn = self.attention(q, k, v, mask=mask)
        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q,
                                                              -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class MultiHeadAttentionConv(nn.Module):
    def __init__(self, n_head: int, d_model: List[int], d_k: int, d_v: int,
                 dropout: float = 0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        C, H, W = d_model
        out_ch_k = int((n_head * d_k) / (H * W))
        out_ch_v = int((n_head * d_v) / (H * W))

        self.w_qs = nn.Conv2d(C, out_ch_k, 1, 1, 0)
        self.w_ks = nn.Conv2d(C, out_ch_k, 1, 1, 0)
        self.w_vs = nn.Conv2d(C, out_ch_v, 1, 1, 0)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (C + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (C + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (C + d_v)))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.InstanceNorm2d(C)

        self.fc = nn.Conv2d(out_ch_v, C, 1, 1, 0)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    # def forward(self, q, k, v, mask=None):
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        len_q, _, _, _ = q.size()
        len_k, _, _, _ = k.size()
        len_v, _, _, _ = v.size()

        residual = q

        q = self.w_qs(q).view(1, len_q, n_head, d_k)
        k = self.w_ks(k).view(1, len_k, n_head, d_k)
        v = self.w_vs(v).view(1, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q,
                                                    d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k,
                                                    d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v,
                                                    d_v)  # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        # output, attn = self.attention(q, k, v, mask=mask)
        output, attn = self.attention(q, k, v)

        output = output.view(n_head, 1, len_q, d_v)
        _, H, W = self.d_model
        output = output.permute(1, 2, 0, 3).contiguous().view(
            len_q, -1, H, W)  # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class RelationModule(torch.nn.Module):
    def __init__(self, ch_in, atype='dot_product'):
        super().__init__()
        self.atype = atype
        self.ch_in = ch_in
        self.ch_out = ch_in // 8

        self.phi = nn.Linear(self.ch_in, self.ch_out)
        self.psi = nn.Linear(self.ch_in, self.ch_out)
        self.linear = nn.Linear(self.ch_in, self.ch_in)

    def __call__(self, x):
        N_roi, N_feat = x.shape
        phi_x = self.phi(x)
        psi_x = self.psi(x)
        if self.atype == 'dot_product':
            h = torch.matmul(psi_x, torch.transpose(phi_x, 0, 1))
        else:
            raise NotImplementedError
        h = torch.matmul(h, x)
        h = self.linear(h) / len(h)
        return h + x

# if __name__ == '__main__':
#     N_roi = 10
#     N_feat = 256
#     feat = torch.zeros((N_roi, N_feat))
#
#     module = RelationModule(N_feat)
#     module(feat)