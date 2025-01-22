import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Multihead Attention Block
# in practice dim_Q = dim_K
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2) # Mulitihead(Q_,K_,V_)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2) # Q_+ Mulitihead(Q_,K_,V_)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)  # H=LayerNorm(Q_+ Mulitihead(Q_,K_,V_))
        O = O + F.relu(self.fc_o(O))  # H + rFF(H)
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)  # LayerNorm(H + rFF(H))
        return O

# Set Attention Blocks
# in paper SAB(X):=MAB(X,X)
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

# Induced Set Attention Block
class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out)) # inducing points I.shape:=tensor([num_inds, dim_out])
        nn.init.xavier_uniform_(self.I)                           # initilize inducing points
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)          # H:=MAB(I,X) H.shape:=(num_inds,dim_out)
        return self.mab1(X, H)                                    # ISABm(X):=MAB(X,H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

def main():
    example = ISAB(dim_in=3, dim_out=8, num_heads=4, num_inds=2, ln=False)
    print("example.I.shape:")
    print(example.I.shape) #(1,2,8)
    h1 = example(np.random.randn(3,2,8))
    print(h1.shape)

if __name__ == '__main__':
    main()

