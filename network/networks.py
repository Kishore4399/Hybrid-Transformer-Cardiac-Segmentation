# coding=utf-8
# codes from https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
from xml.etree.ElementTree import QName
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm, MaxPool3d
from torch.nn.modules.utils import _pair, _triple
from .configs import *
from torch.distributions.normal import Normal


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention_block_map(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block_map,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int),
            #nn.InstanceNorm3d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int),
            #nn.InstanceNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(1),
            #nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return psi

class TransformerMoudle(nn.Module):
    def __init__(self, num_layers, feature_size, patch_size, in_channels, hidden_size):
        super(TransformerMoudle, self).__init__()
        self.patch_size = patch_size
        self.embeddings = EmbeddingsM(feature_size, patch_size, in_channels, hidden_size)
        self.encoder = EncoderM(num_layers, hidden_size)
        if max(self.patch_size) != 1:
            self.up = nn.Upsample(scale_factor=self.patch_size, mode='trilinear', align_corners=False)
        self.l, self.h, self.w = int(feature_size[0] // patch_size[0]), int(feature_size[1] // patch_size[1]), int(feature_size[2] // patch_size[2])

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        encoded = encoded.permute(0, 2, 1) # (B, hidden, n_patch)
        B, hidden, n_patches = encoded.size()
        encoded = encoded.contiguous().view(B, hidden, self.l, self.h, self.w)
        if max(self.patch_size) != 1:
            encoded = self.up(encoded)
        return encoded

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm3d(input_dim),
            nn.ReLU(),
            nn.Conv3d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm3d(output_dim),
            nn.ReLU(),
            nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class SimpleUnet(nn.Module):
    def consecutive_conv_f(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, padding=0),
        )
    def convs(self, in_channels, out_channels, mid_channels=None):
        if not mid_channels:
            mid_channels = out_channels
        return nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def __init__(self, num_channels=64, num_inputs=4, num_outputs=4):
        super(SimpleUnet, self).__init__()

        self.conv_initial = self.convs(num_inputs, num_channels)

        self.conv_rest1 = self.convs(num_channels * 1, num_channels * 2)
        self.conv_rest2 = self.convs(num_channels * 2, num_channels * 4)
        self.conv_rest3 = self.convs(num_channels * 4, num_channels * 8)
        self.conv_rest4 = self.convs(num_channels * 8, num_channels * 16)  
        
        self.conv_up1 = self.convs(num_channels * 16, num_channels * 8)
        self.conv_up2 = self.convs(num_channels * 8, num_channels * 4)
        self.conv_up3 = self.convs(num_channels * 4, num_channels * 2 )
        self.conv_up4 = self.convs(num_channels * 2, num_channels)

        self.conv_final = self.consecutive_conv_f(num_channels * 1, num_outputs)
        
        self.maxpool = nn.MaxPool3d(2)

        self.upsample1 = nn.ConvTranspose3d(num_channels * 16, num_channels * 8, 2, 2)
        self.upsample2 = nn.ConvTranspose3d(num_channels * 8, num_channels * 4, 2, 2)
        self.upsample3 = nn.ConvTranspose3d(num_channels * 4, num_channels * 2, 2, 2)
        self.upsample4 = nn.ConvTranspose3d(num_channels * 2, num_channels * 1, 2, 2)

    def forward(self, x):
        x1 = self.conv_initial(x)
        x2 = self.maxpool(x1)
        x2 = self.conv_rest1(x2)
        x3 = self.maxpool(x2)
        x3 = self.conv_rest2(x3)
        x4 = self.maxpool(x3)
        x4 = self.conv_rest3(x4)
        x5 = self.maxpool(x4)
        x5 = self.conv_rest4(x5)
        
        x = self.upsample1(x5)
        x = self.conv_up1(torch.cat((x, x4), 1))
        x = self.upsample2(x)
        x = self.conv_up2(torch.cat((x, x3), 1))
        x = self.upsample3(x)
        x = self.conv_up3(torch.cat((x, x2), 1))
        x = self.upsample4(x)
        x = self.conv_up4(torch.cat((x, x1), 1))

        x = self.conv_final(x)
        return x

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, input_channels):
        super(Embeddings, self).__init__()
        self.config = config
        down_num = config.down_num
        patch_size = _triple(config.patches["size"])
        n_patches = int((img_size[0]/2**down_num// patch_size[0]) * (img_size[1]/2**down_num// patch_size[1]) * (img_size[2]/2**down_num// patch_size[2]))
        self.hybrid_model = CNNEncoder_stride(config, input_channels)
        in_channels = config['encoder_channels'][-1]
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        #B = x.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1)

        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/3), n_patches^(1/3), , n_patches^(1/3))
        x = x.flatten(2) # (B, hidden. n_patches)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        #x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, input_channels):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, input_channels=input_channels)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features

class ConsecutiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = 0):
        super(ConsecutiveConv, self).__init__()

        if mid_channels == 0:
            mid_channels = out_channels
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.convs(x)

class ConsecutiveConv_res(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = 0):
        super(ConsecutiveConv_res, self).__init__()

        if mid_channels == 0:
            mid_channels = out_channels
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.convs(x) + x

class ConsecutiveConv_res_orig(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = 0):
        super(ConsecutiveConv_res_orig, self).__init__()

        if mid_channels == 0:
            mid_channels = out_channels
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.convs(x) + x

class CNNEncoderBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(CNNEncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.MaxPool3d(2, stride=2),
            ConsecutiveConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)

class CNNEncoderBlock_stride(nn.Module):
    """Downscaling with strided convolution then max pooling"""

    def __init__(self, in_channels, out_channels, stride):
        super(CNNEncoderBlock_stride, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, stride=stride),
            ConsecutiveConv_res(out_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)

class CNNEncoder(nn.Module):
    def __init__(self, config, n_channels=1):
        super(CNNEncoder, self).__init__()
        self.n_channels = n_channels
        encoder_channels = config.encoder_channels
        self.down_num = config.down_num
        self.inc = ConsecutiveConv(n_channels, encoder_channels[0])
        self.down1 = CNNEncoderBlock(encoder_channels[0], encoder_channels[1])
        self.down2 = CNNEncoderBlock(encoder_channels[1], encoder_channels[2])
        self.width = encoder_channels[-1]

    def forward(self, x):
        features = []
        x1 = self.inc(x)
        features.append(x1)
        x2 = self.down1(x1)
        features.append(x2)
        feats = self.down2(x2)
        features.append(feats)
        feats_down = feats
        for i in range(self.down_num):
            feats_down = nn.MaxPool3d(2)(feats_down)
            features.append(feats_down)
        return feats, features[::-1]

class CNNEncoder_stride_orig(nn.Module):
    def __init__(self, config, n_channels=1):
        super(CNNEncoder_stride_orig, self).__init__()
        self.n_channels = n_channels
        encoder_channels = config.encoder_channels
        self.down_num = config.down_num
        self.inc = ConsecutiveConv(n_channels, encoder_channels[0])
        self.dropout = Dropout(0.2)
        self.enblock1 = ConsecutiveConv_res_orig(encoder_channels[0], encoder_channels[0])
        blocks = [
            CNNEncoderBlock_stride_orig(encoder_channels[i], encoder_channels[i+1], config.down_factor) for i in range(self.down_num)
        ]
        self.blocks = nn.ModuleList(blocks)
        #self.enblock2 = ConsecutiveConv_res_orig(encoder_channels[self.down_num], encoder_channels[self.down_num])
        #self.enblock3 = ConsecutiveConv_res_orig(encoder_channels[self.down_num], encoder_channels[self.down_num])

    def forward(self, x):
        features = []
        x = self.inc(x)
        x = self.dropout(x)
        x = self.enblock1(x)
        features.append(x)
        for encoder_block in self.blocks:
            x = encoder_block(x)
            features.append(x)
        #x = self.enblock2(x)
        #x = self.enblock3(x)

        return x, features[::-1][1:]

class CNNEncoder_stride_orig2(nn.Module):
    def __init__(self, config, n_channels=1):
        super(CNNEncoder_stride_orig2, self).__init__()
        self.n_channels = n_channels
        encoder_channels = config.encoder_channels
        self.down_num = config.down_num
        self.inc = ConsecutiveConv(n_channels, encoder_channels[0])
        blocks = [
            CNNEncoderBlock_stride_orig2(encoder_channels[i], encoder_channels[i+1], config.down_factor) for i in range(self.down_num)
        ]
        self.blocks = nn.ModuleList(blocks)
        #self.enblock2 = ConsecutiveConv_res_orig(encoder_channels[self.down_num], encoder_channels[self.down_num])
        #self.enblock3 = ConsecutiveConv_res_orig(encoder_channels[self.down_num], encoder_channels[self.down_num])

    def forward(self, x):
        features = []
        x = self.inc(x)
        features.append(x)
        for encoder_block in self.blocks:
            x = encoder_block(x)
            features.append(x)
        #x = self.enblock2(x)
        #x = self.enblock3(x)

        return x, features[::-1][1:]

class CNNEncoder_stride(nn.Module):
    def __init__(self, config, n_channels=1):
        super(CNNEncoder_stride, self).__init__()
        self.n_channels = n_channels
        encoder_channels = config.encoder_channels
        self.down_num = config.down_num
        self.inc = ConsecutiveConv(n_channels, encoder_channels[0])

        blocks = [
            CNNEncoderBlock_stride(encoder_channels[i], encoder_channels[i+1], config.down_factor) for i in range(self.down_num)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        features = []
        x = self.inc(x)
        features.append(x)
        for encoder_block in self.blocks:
            x = encoder_block(x)
            features.append(x)

        return x, features[::-1][1:]

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        ins = nn.InstanceNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, ins, relu)


class ConsecutiveConv_up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(ConsecutiveConv_up, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=1)

    def forward(self, x, feat):
        x = self.conv1(x)
        x = self.conv2(x)
        if feat is not None:
            x = torch.cat((x,feat), dim=1)
        x = self.conv3(x)
        return x


class CNNDecoderBlock_transpose(nn.Module):
    """Upsampling with transposed convolution"""

    def __init__(self, in_channels, out_channels, skip_channels):
        super(CNNDecoderBlock_transpose, self).__init__()
        self.upblock = ConsecutiveConv_up(in_channels, out_channels, skip_channels)
        self.block = ConsecutiveConv_res(out_channels, out_channels)

    def forward(self, x, feat):
        x = self.upblock(x, feat)
        x = self.block(x)
        return x



class DecoderCupBTS(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.config = config
        self.down_num = config.down_num
        head_channels = config.conv_first_channel
        self.img_size = img_size
        self.conv_more = ConsecutiveConv(config.hidden_size, head_channels)

        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.patch_size = _triple(config.patches["size"])
        skip_channels = self.config.skip_channels
        blocks = [
            CNNDecoderBlock_transpose(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        if max(self.patch_size) != 1:
            self.up = nn.Upsample(scale_factor=self.patch_size, mode='trilinear', align_corners=False)


    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        l, h, w = (self.img_size[0]//2**self.down_num//self.patch_size[0]), (self.img_size[1]//2**self.down_num//self.patch_size[1]), (self.img_size[2]//2**self.down_num//self.patch_size[2])
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, l, h, w)
        if max(self.patch_size) != 1:
            x = self.up(x)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
                #print(skip.shape)
            else:
                skip = None
            x = decoder_block(x, skip)
        return x

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)


class HFTrans(nn.Module):
    def __init__(self, config, img_size=(128, 128, 128), input_channels=1, num_classes=1, vis=False):
        super(HFTrans, self).__init__()
        self.transformer = Transformer(config, img_size, vis, input_channels)
        self.decoder = DecoderCupBTS(config, img_size)
        self.seg_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        seg = self.seg_head(x)
        return seg, attn_weights


class Embeddings_HFTrans(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, input_channels, num_encoders):
        super(Embeddings_HFTrans, self).__init__()
        self.config = config
        down_factor = config.down_num
        patch_size = _triple(config.patches["size"])
        n_patches = num_encoders * int(
            (img_size[0] / 2 ** down_factor // patch_size[0]) * (img_size[1] / 2 ** down_factor // patch_size[1]) * (
                        img_size[2] / 2 ** down_factor // patch_size[2]))

        self.early_encoder = CNNEncoder_stride(config, input_channels)
        hybrid_encoders = []
        for i in range(num_encoders - 1):
            hybrid_encoders.append(CNNEncoder_stride(config, 1))
        self.hybrid_encoders = nn.ModuleList(hybrid_encoders)
        in_channels = config['encoder_channels'][-1]
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches // num_encoders, num_encoders, config.hidden_size))
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.num_encoders = num_encoders

    def forward(self, x):
        # B = x.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        x_enc = []
        xx, features = self.early_encoder(x)  # all

        x_enc.append(xx)
        features_enc = [[f] for f in features]
        for i, hybrid_encoder in enumerate(self.hybrid_encoders):
            xx, features = hybrid_encoder(x[:, i:i + 1, :, :, :])
            x_enc.append(xx)
            for n, f in enumerate(features):
                features_enc[n].append(f)

        x_enc = [self.patch_embeddings(x) for x in x_enc]
        x_enc = [torch.unsqueeze(x, 5) for x in x_enc]

        x = torch.cat(x_enc, 5)
        x = x.flatten(2, 4)
        x = x.permute(0, 2, 3, 1)

        features = [torch.cat((f), 1) for f in features_enc]
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features

class Transformer_HFTrans(nn.Module):
    def __init__(self, config, img_size, vis, input_channels, num_encoders):
        super(Transformer_HFTrans, self).__init__()
        self.embeddings = Embeddings_HFTrans(config, img_size=img_size, input_channels=input_channels, num_encoders=num_encoders)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        embedding_output = embedding_output.flatten(1,2)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features

class DecoderHFTrans(nn.Module):
    def __init__(self, config, img_size, num_encoders):
        super().__init__()
        self.config = config
        self.down_num = config.down_num
        head_channels = config.conv_first_channel
        self.img_size = img_size
        self.conv_more = ConsecutiveConv(config.hidden_size*num_encoders, head_channels)

        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.patch_size = _triple(config.patches["size"])
        skip_channels = self.config.skip_channels
        blocks = [
            CNNDecoderBlock_transpose(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        if max(self.patch_size) != 1:
            self.up = nn.Upsample(scale_factor=self.patch_size, mode='trilinear', align_corners=False)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        l, h, w = (self.img_size[0]//2**self.down_num//self.patch_size[0]), (self.img_size[1]//2**self.down_num//self.patch_size[1]), (self.img_size[2]//2**self.down_num//self.patch_size[2])
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, l, h, w)
        if max(self.patch_size) != 1:
            x = self.up(x)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip)
        return x

class HFTrans2(nn.Module):
    def __init__(self, config, img_size=(128, 128, 128), input_channels=1, num_classes=1, vis=False):
        super(HFTrans2, self).__init__()
        self.num_encoders = input_channels + 1

        self.transformer = Transformer_HFTrans(config, img_size, vis, input_channels, self.num_encoders)
        self.decoder = DecoderHFTrans(config, img_size, self.num_encoders)
        self.seg_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=1,
        )
        self.config = config

    def forward(self, x):
        x, attn_weights, features = self.transformer(x)
        B, n_patch, h = x.size()
        x = x.view(B, n_patch//self.num_encoders, self.num_encoders*h)
        x = self.decoder(x, features)
        seg = self.seg_head(x)
        return seg, attn_weights

CONFIGS = {
    'BTS' : get_Hybrid3DTransformer_BTS_config(),
    'HFTrans2' : get_HFTrans2_16_config(),
}