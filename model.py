import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import math
import copy



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("*********************************")
print(f"DEVICE = {device}")
print("*********************************")


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.input_embedding = nn.Embedding(vocab_size, d_model) # Matrix of size (vocab_size, embed_dim)

    def forward(self, x):
        return self.input_embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size, d_model, seq_length=512):
        super(PositionalEncoding, self).__init__()
        
        # self.positional_encoding = torch.zeros()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_length = seq_length
        self.positional_encoding = torch.zeros(self.seq_length*self.d_model).reshape(self.seq_length, self.d_model)

    
    def forward(self, x):
        
        batch_size, seq_length = x.shape
        n = 10000

        for pos in range(self.seq_length):
            # for each dimension
            for i in range(self.d_model//2):
                # calculate the internal value for sin and cos
                theta = pos / (n ** ((2*i)/self.d_model))

                # even dims: sin   
                self.positional_encoding[pos, 2*i] = math.sin(theta)

                # odd dims: cos               
                self.positional_encoding[pos, 2*i+1] = math.cos(theta)
        
        return torch.unsqueeze(self.positional_encoding, 0).repeat((batch_size, 1, 1))

        # Output shape Batch size, Seq length, d_model


# test = torch.randint(high=2048, size=(16, 128))
# pe = PositionalEncoding(2048, 512)
# out = pe(test)
# print(test.shape)
# print(out.shape)




# Encoder Attention
class SelfAttention(nn.Module):
    def __init__(self, dims, d_model):
        super(SelfAttention, self).__init__()

        self.Wq = nn.Linear(in_features=d_model, out_features=dims)
        self.Wk = nn.Linear(in_features=d_model, out_features=dims)
        self.Wv = nn.Linear(in_features=d_model, out_features=dims)

        self.dims = dims
        self.d_model = d_model


    def forward(self, x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        normalization = (self.dims)**(-0.5)
        attention_scores = ((q @ torch.transpose(k, 1, 2)) * normalization) @ v
        return attention_scores



class MultiHeadAttention(nn.Module):
    # Input is Input Embeddings + Positional Encoding
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "Embedding dimensions should be divisible by num heads"
        self.dims = d_model // num_heads
        self.self_attention_blocks = nn.ModuleList([SelfAttention(self.dims, d_model) for _ in range(num_heads)])


    def forward(self, x):
        for idx, block in enumerate(self.self_attention_blocks):
            # if idx == 0
            if not idx:
                res = block(x)
            
            # if idx != 0
            else:
                res = torch.cat((res, block(x)), 2)
        return res

# test = torch.rand((16, 128, 512))
# mha = MultiHeadAttention(512, 8)
# out = mha(test)
# print(out.shape)


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_dim, non_linearity=F.relu):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, inner_dim)
        self.linear2 = nn.Linear(inner_dim, d_model)
        self.non_linearity = non_linearity
        
    def forward(self, x):
        return self.linear2(self.non_linearity(self.linear1(x)))



class AttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, inner_dim):
        super(AttentionBlock, self).__init__()

        """
            · Multi-Head Attention
            · Feed Forward
        """

        self.multihead_attention    = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward           = FeedForward(d_model, inner_dim)
        

    def forward(self, x):
        x += self.multihead_attention(x)
        bs, N, dims = x.shape
        x = F.layer_norm(x, (N, dims))


        x += self.feed_forward(x)
        x = F.layer_norm(x, (N, dims))
        return x


# batch_size, seq_length, d_model = 16, 128, 512
# test = torch.rand((batch_size, seq_length, d_model)).to(device)
# mha = AttentionBlock(d_model=512, num_heads=8, inner_dim=2048).to(device)
# out = mha(test)
# print(test.shape)
# print(out.shape)


# Size of vocab_size is random
class Encoder(nn.Module):
    def __init__(self, d_model=512, N=6, num_heads=8, inner_dim=2048, vocab_size=2048):
        super(Encoder, self).__init__()

        self.input_embedding = Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEncoding(vocab_size, d_model, seq_length=128)
        self.attention_block = nn.ModuleList([AttentionBlock(d_model=d_model, num_heads=num_heads, inner_dim=inner_dim) 
                                              for _ in range(N)])

    def forward(self, x):
        x = self.input_embedding(x) + self.positional_embedding(x)


        for block in self.attention_block:
            x = block(x)
        
        return x


vocab_size = 2048
batch_size, seq_length, d_model = 16, 128, 512
test = torch.randint(high=vocab_size, size=(batch_size, seq_length))
enc = Encoder(d_model=512, N=6, num_heads=8, inner_dim=2048, vocab_size=vocab_size)
out = enc(test)
print(test.shape)
print(out.shape)





# class Decoder(nn.Module):
#     def __init__(self, ):
#         super(Decoder, self).__init__()
#         pass



#     def forward(self, ):
#         pass
