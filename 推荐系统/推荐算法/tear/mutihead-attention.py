import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MultiHeadSelfAttention(nn.Module):
    def _init_(self,embed_dim,num_heads):
        super(MultiHeadSelfAttention,self).__init__()
        self.num_heads=num_heads
        self.head_dim=embed_dim/num_heads

        self.query=nn.linear(embed_dim,embed_dim)
        self.key=nn.linear(embed_dim,embed_dim)
        self.value=nn.linear(embed_dim,embed_dim)
        self.fc=nn.linear(embed_dim,embed_dim)
    
    def forward(self,x):
        batch_size,seq_len,embed_dim=x.size()

        q=self.query(x).view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k=self.key(x).view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v=self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights=torch.matmul(q,k.tanspose(-2,-1))/torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        x = self.fc(attended_values) + x

        return x

