import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import math
import copy

VOCAB_SIZE  = 2048
BATCH_SIZE  = 16
SEQ_LENGTH  = 128
D_MODEL     = 256
NUM_HEADS   = 8
N           = 6
INNER_DIM   = 1028


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("*********************************")
# print(f"DEVICE = {device}")
# print("*********************************")


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
        batch_size, _ = x.shape
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

# -------------------- TEST CODE -----------------------------------
# ------------------------------------------------------------------
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
        attention_scores = F.softmax((torch.matmul(q, torch.transpose(k, 1, 2)) * normalization), dim=-1)
        out = torch.matmul(attention_scores, v)
        return out



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

# -------------------- TEST CODE -----------------------------------
# ------------------------------------------------------------------
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


# -------------------- TEST CODE -----------------------------------
# ------------------------------------------------------------------
# batch_size, seq_length, d_model = 16, 128, 512
# test = torch.rand((BATCH_SIZE, SEQ_LENGTH, D_MODEL)).to(device)
# mha = AttentionBlock(d_model=512, num_heads=8, inner_dim=2048).to(device)
# out = mha(test)
# print(test.shape)
# print(out.shape)


# Size of vocab_size is random
class Encoder(nn.Module):
    def __init__(self, d_model=512, N=6, num_heads=8, inner_dim=2048, vocab_size=2048, seq_length=128):
        super(Encoder, self).__init__()

        # self.input_embedding = Embedding(vocab_size, d_model)
        # self.positional_embedding = PositionalEncoding(vocab_size, d_model, seq_length=seq_length)
        self.attention_block = nn.ModuleList([AttentionBlock(d_model=d_model, num_heads=num_heads, inner_dim=inner_dim) 
                                              for _ in range(N)])

    def forward(self, x):
        # x = self.input_embedding(x) + self.positional_embedding(x)
        for block in self.attention_block:
            x = block(x)
        return x


# -------------------- TEST CODE -----------------------------------
# ------------------------------------------------------------------
# test = torch.randint(high=VOCAB_SIZE, size=(batch_size, seq_length))
# enc = Encoder(d_model=512, N=6, num_heads=8, inner_dim=2048, vocab_size=VOCAB_SIZE)
# out = enc(test)
# print(test.shape)
# print(out.shape)





class Decoder(nn.Module):
    def __init__(self, d_model=512, N=6, num_heads=8, inner_dim=2048, vocab_size=2048, seq_length=128):
        super(Decoder, self).__init__()

        # self.input_embedding = Embedding(vocab_size, d_model)
        # self.positional_embedding = PositionalEncoding(vocab_size, d_model, seq_length=seq_length)
        self.attention_block = nn.ModuleList([DecoderAttentionBlock(d_model=d_model, num_heads=num_heads, inner_dim=inner_dim) 
                                              for _ in range(N)])


    def forward(self, x, encoder_output):
        # x = self.input_embedding(x) + self.positional_embedding(x)
        for block in self.attention_block:
            x = block(x, encoder_output)
        return x



class DecoderAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, inner_dim):
        super(DecoderAttentionBlock, self).__init__()

        """
            · Masked Multi-Head Attention
            · Multi-Head Cross Attention
            · Feed Forward
        """

        self.multihead_attention            = MaskedMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.multihead_crossattention       = DecoderMultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward                   = FeedForward(d_model, inner_dim)


    def forward(self, x, encoder_output):
        _, N, dims = x.shape
        
        x += self.multihead_attention(x)
        x = F.layer_norm(x, (N, dims))
        x += self.multihead_crossattention(encoder_output, x)
        x = F.layer_norm(x, (N, dims))
        x += self.feed_forward(x)
        x = F.layer_norm(x, (N, dims))
        return x




class MaskedAttention(nn.Module):
    def __init__(self, d_model, dims):
        super(MaskedAttention, self).__init__()

        self.Wq = nn.Linear(in_features=d_model, out_features=dims)
        self.Wk = nn.Linear(in_features=d_model, out_features=dims)
        self.Wv = nn.Linear(in_features=d_model, out_features=dims)
        self.mask = torch.triu(torch.ones((SEQ_LENGTH, SEQ_LENGTH))*float('-inf'), diagonal=1)

        self.dims = dims
        self.d_model = d_model


    def forward(self, x):
        B, T, C = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        
        normalization = (self.dims)**(-0.5)
        # (( Q * Kt + mask ) / norm ) * V
        attention_scores = F.softmax((torch.matmul(q, torch.transpose(k, 1, 2)) * normalization), dim=-1)
        out = torch.matmul(attention_scores, v)

        return out


# -------------------- TEST CODE -----------------------------------
# ------------------------------------------------------------------
# test = torch.rand((BATCH_SIZE, SEQ_LENGTH, D_MODEL))
# mha = MaskedAttention(d_model=d_model, dims = d_model//8)
# out = mha(test)
# print(test.shape)
# print(out.shape)
# print(out)

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MaskedMultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "Embedding dimensions should be divisible by num heads"
        self.dims = d_model // num_heads
        self.self_attention_blocks = nn.ModuleList([MaskedAttention(d_model=d_model, dims=self.dims) for _ in range(num_heads)])


    def forward(self, x):
        for idx, block in enumerate(self.self_attention_blocks):
            # if idx == 0
            if not idx:
                res = block(x)
            
            # if idx != 0
            else:
                res = torch.cat((res, block(x)), 2)
        return res


# -------------------- TEST CODE -----------------------------------
# ------------------------------------------------------------------
# test = torch.rand((BATCH_SIZE, SEQ_LENGTH, D_MODEL))
# mha = MaskedMultiHeadAttention(d_model=d_model, num_heads=8)
# out = mha(test)
# print(test.shape)
# print(out.shape)
# print(out)


class CrossAttention(nn.Module):
    def __init__(self, dims, d_model):
        super(CrossAttention, self).__init__()

        self.Wq = nn.Linear(in_features=d_model, out_features=dims)
        self.Wk = nn.Linear(in_features=d_model, out_features=dims)
        self.Wv = nn.Linear(in_features=d_model, out_features=dims)

        self.dims = dims
        self.d_model = d_model


    def forward(self, encoder_output, decoder):
        # From the decoder
        q = self.Wq(decoder)
        
        # From the encoder
        k = self.Wk(encoder_output)
        v = self.Wv(encoder_output)

        normalization = (self.dims)**(-0.5)
        attention_scores = F.softmax((torch.matmul(q, torch.transpose(k, 1, 2)) * normalization), dim=-1)
        out = torch.matmul(attention_scores, v)
        return out



class DecoderMultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderMultiHeadCrossAttention, self).__init__()
        
        assert d_model % num_heads == 0, "Embedding dimensions should be divisible by num heads"
        self.dims = d_model // num_heads
        self.self_attention_blocks = nn.ModuleList([CrossAttention(self.dims, d_model) for _ in range(num_heads)])


    def forward(self, encoder_output, decoder_input):
        # print(f"DecoderMultiHeadCrossAttention/ decoder_input shape:{decoder_input.shape}")
        for idx, block in enumerate(self.self_attention_blocks):
            # if idx == 0
            if not idx:
                res = block(encoder_output, decoder_input)
            
            # if idx != 0
            else:
                res = torch.cat((res, block(encoder_output, decoder_input)), 2)
        # print(f"DecoderMultiHeadCrossAttention/ output shape:{res.shape}")
        return res
    

# -------------------- TEST CODE -----------------------------------
# ------------------------------------------------------------------


# test = torch.randint(high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LENGTH))
# test2 = torch.randint(high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LENGTH))

# enc = Encoder(d_model=D_MODEL, seq_length=SEQ_LENGTH, N=N, num_heads=NUM_HEADS, inner_dim=INNER_DIM, vocab_size=VOCAB_SIZE)
# dec = Decoder(d_model=D_MODEL, seq_length=SEQ_LENGTH, N=N, num_heads=NUM_HEADS, inner_dim=INNER_DIM, vocab_size=VOCAB_SIZE)

# emb = Embedding(VOCAB_SIZE, D_MODEL)
# pos = PositionalEncoding(VOCAB_SIZE, D_MODEL, seq_length=SEQ_LENGTH)
# masked_attn = MaskedMultiHeadAttention(d_model=D_MODEL, num_heads=8)

# enc_out = enc(test2)
# out = dec(test, enc_out)
# dec_out = emb(test) + pos(test)
# dec_out = masked_attn(dec_out)

# mha = DecoderMultiHeadCrossAttention(d_model=D_MODEL, num_heads=8)
# out = mha(enc_out, dec_out)
# print(test.shape)
# print(out.shape)
# print(out)




class Transformer(nn.Module):
    def __init__(self, d_model=512, N=6, num_heads=8, inner_dim=2048, vocab_size=2048, seq_length=128, batch_size=32):
        super(Transformer, self).__init__()

        self.input_embedding        = Embedding(vocab_size, d_model)
        self.positional_embedding   = PositionalEncoding(vocab_size, d_model, seq_length=seq_length)
        self.encoder                = Encoder(d_model=d_model, seq_length=seq_length, N=N, num_heads=num_heads, inner_dim=inner_dim, vocab_size=vocab_size)
        self.decoder                = Decoder(d_model=d_model, seq_length=seq_length, N=N, num_heads=num_heads, inner_dim=inner_dim, vocab_size=vocab_size)
        self.d_model = d_model


    def forward(self, x, y):
        x = self.input_embedding(x)*(self.d_model)**0.5 + self.positional_embedding(x) # multiply embeddings by sqrt(d_model)
        y = self.input_embedding(y)*(self.d_model)**0.5 + self.positional_embedding(y) # multiply embeddings by sqrt(d_model)
        enc_out = self.encoder(x)
        dec_out = self.decoder(y, enc_out)
        pre_softmax = torch.matmul(self.input_embedding.input_embedding.weight, torch.transpose(dec_out, 1, 2))
        return pre_softmax
        # out = F.softmax(pre_softmax, dim=1)
        # out = torch.argmax(out, dim=1)

# -------------------- TEST CODE -----------------------------------
# ------------------------------------------------------------------
# x = torch.randint(high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LENGTH))
# y = torch.randint(high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LENGTH))
# transformer = Transformer(d_model=D_MODEL, seq_length=SEQ_LENGTH, N=N, num_heads=NUM_HEADS, inner_dim=INNER_DIM, vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE)


# transformer = Transformer(batch_size=32, vocab_size=enc.n_vocab, d_model=256, N=4, num_heads=4, inner_dim=1024, 
#                           seq_length=128)



# out = transformer(x, y)
# print(x.shape)
# print(y.shape)
# print(out.shape)