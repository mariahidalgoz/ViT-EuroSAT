import math

import torch
from torch import nn


class VisionTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches,
                 dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size ** 2), embed_dim)

        # self.moduleList = nn.ModuleList()
        # for _ in range(num_layers):
        #     layer2 = AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)
        #     self.moduleList.append(copy.deepcopy(layer2))

        self.modules = []
        for _ in range(num_layers):
            layer = AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)
            self.modules.append(layer)
        self.transformer = nn.Sequential(*self.modules)
        # self.transformer = nn.Sequential(*[
        #     AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        # ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, :T + 1]

        # Apply Transformer
        x = self.dropout(x)
        x = x.transpose(0, 1)

        hidden_states = x
        # x = self.transformer(x)
        attn_weights = []
        for layer_block in self.modules:
            # hidden_states, weights = layer_block(hidden_states)
            x, weights = layer_block(x)
            attn_weights.append(weights)
        # x = hidden_states

        # Perform classification prediction
        cls = x[0]  # cls = x      # cls = x[0]
        out = self.mlp_head(cls)
        return out, attn_weights


class MHAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_dim, self.all_head_size)
        self.key = nn.Linear(hidden_dim, self.all_head_size)
        self.value = nn.Linear(hidden_dim, self.all_head_size)

        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

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
        # return attention_output, weights
        return attention_output  # , weights


class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)  # LN
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)  # MSA
        # self.attn = MHAttention(embed_dim, hidden_dim, num_heads)     # MSA
        # self.attn = MHAttention(embed_dim, num_heads)     # MSA
        self.layer_norm_2 = nn.LayerNorm(embed_dim)  # LN
        self.linear = nn.Sequential(  # MLP
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = x
        inp_x = self.layer_norm_1(x)  # LN    # self.attention_norm(x)
        x, weights = self.attn(inp_x, inp_x, inp_x, need_weights=True, average_attn_weights=False)  # self.attn(x)
        # print("weights", weights)
        x = x + h  # MSA

        # x = x + self.linear(self.layer_norm_2(x))         # LN & MLP
        h = x
        x = self.layer_norm_2(x)  # LN
        x = self.linear(x)  # MLP
        x = x + h
        return x, weights

        # inp_x = self.layer_norm_1(x)                    # LN
        # x = x + self.attn(inp_x, inp_x, inp_x)[0]       # MSA
        # x = x + self.linear(self.layer_norm_2(x))       # LN & MLP
        # return x


def img_to_patch(x: torch.Tensor, patch_size: int, flatten_channels: bool = True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x
