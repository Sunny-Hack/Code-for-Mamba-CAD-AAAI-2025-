from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask
import random
import numpy as np
import torch
from transformers import AutoTokenizer
from einops import rearrange, repeat, einsum

class CADEmbedding(nn.Module):
    """Embedding: positional embed + command embed + parameter embed + group embed (optional)"""
    def __init__(self, cfg, seq_len, use_group=False, group_len=None):
        super().__init__()

        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)
        args_dim = cfg.args_dim + 1
        self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)

        # use_group: additional embedding for each sketch-extrusion pair
        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len + 2, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len+2)

    def forward(self, commands, args, groups=None):
        S, N = commands.shape

        src = self.command_embed(commands.long()) + \
              self.embed_fcn(self.arg_embed((args + 1).long()).view(S, N, -1))  # shift due to -1 PAD_VAL

        if self.use_group:
            src = src + self.group_embed(groups.long())

        #src = self.pos_encoding(src)
        return src


class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg, seq_len):
        super().__init__()

        self.d_model = cfg.d_model
        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=seq_len)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src

        
###design Mamba
##### Mamba Core 
class Mamba(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        seq_len = cfg.max_total_len
        self.embedding = CADEmbedding(cfg, seq_len)
        self.layers = nn.ModuleList([ResidualBlock(cfg) for _ in range(cfg.m_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.vocab_size, cfg.d_model, bias=False)
        #self.latent_layer = nn.Linear(seq_len * cfg.vocab_size, cfg.vocab_size, bias = False)
        #self.lm_head.weight = self.command_embed.weight
    def forward(self, commands, args):
        x = self.embedding(commands, args)
        x = _make_batch_first(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits
###### Residual module
class ResidualBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mixer = MambaBlock(cfg)
        self.norm = RMSNorm(cfg.d_model)
    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output
####### Mamba Block
class MambaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_proj = nn.Linear(cfg.d_model, cfg.d_inner * 2, bias=cfg.bias)

        self.conv1d = nn.Conv1d(
            in_channels=cfg.d_inner,
            out_channels=cfg.d_inner,
            bias=cfg.conv_bias,
            kernel_size=cfg.d_conv,
            groups=cfg.d_inner,
            padding=cfg.d_conv - 1,
        )
        self.x_proj = nn.Linear(cfg.d_inner, cfg.dt_rank + cfg.d_state * 2, bias = False)

        self.dt_proj = nn.Linear(cfg.dt_rank, cfg.d_inner, bias = True)
        A = repeat(torch.arange(1, cfg.d_state + 1), 'n -> d n', d=cfg.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(cfg.d_inner))
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=cfg.bias)
        self.d_inner = cfg.d_inner
        self.dt_rank = cfg.dt_rank
    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)
        return output
        
    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        return y
    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y
####### RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
###design Mamba 
class ConvModel(nn.Module): 
    def __init__(self, cfg):
        super(ConvModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(num_features=64),
            nn.Tanh()
            #nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            #nn.BatchNorm1d(num_features=32),
            #nn.ReLU(),
            #nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1),
            #nn.BatchNorm1d(num_features=1),
            #nn.ReLU()
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return x
class CT_ConvModel(nn.Module):
    def __init__(self, cfg):
        super(CT_ConvModel, self).__init__()
        self.conv = nn.Sequential(
            #nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size=1),
            #nn.BatchNorm1d(num_features=32),
            #nn.ReLU(),
            #nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=1),
            #nn.BatchNorm1d(num_features=64),
            #nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x 
class UpConvModel(nn.Module): 
    def __init__(self, cfg):
        super(UpConvModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x        
class FCN(nn.Module):
    def __init__(self, d_model, n_commands, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim

        self.command_fcn = nn.Linear(d_model, n_commands)
        self.args_fcn = nn.Linear(d_model, n_args * args_dim)

    def forward(self, out):
        S, N, _ = out.shape

        command_logits = self.command_fcn(out)  # Shape [S, N, n_commands]

        args_logits = self.args_fcn(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return command_logits, args_logits


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        seq_len = cfg.max_total_len
        
        args_dim = cfg.args_dim + 1
        self.norm = nn.LayerNorm(cfg.d_model)
        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim)
        self.upconv = CT_ConvModel(cfg)
    def forward(self, z):
        z = self.upconv(z)
        z = _make_seq_first(z)
        z = self.norm(z)
        command_logits, args_logits = self.fcn(z)
        out_logits = (command_logits, args_logits)
        return out_logits


class CADTransformer(nn.Module):
    def __init__(self, cfg):
        super(CADTransformer, self).__init__()

        self.args_dim = cfg.args_dim + 1
        self.mamba = Mamba(cfg)
        self.conv = ConvModel(cfg)
        #self.upconv = UpConvModel(cfg)
        #self.upconv = CT_ConvModel(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, commands_enc, args_enc,
                z=None, return_tgt=True, encode_mode=False):
        commands_enc_, args_enc_ = _make_seq_first(commands_enc, args_enc)  # Possibly None, None

        if z is None:
            z = self.mamba(commands_enc_, args_enc_)
            z = self.conv(z)
        #else:
            #z = _make_seq_first(z)
        if encode_mode: return z
        #z = self.upconv(z)
        #z = _make_seq_first(z)
        out_logits = self.decoder(z)
        out_logits = _make_batch_first(*out_logits)


        res = {
            "command_logits": out_logits[0],
            "args_logits": out_logits[1]
        }

        if return_tgt:
            res["tgt_commands"] = commands_enc
            res["tgt_args"] = args_enc

        return res
