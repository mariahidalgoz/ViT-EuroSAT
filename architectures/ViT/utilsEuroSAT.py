import torch
from torchvision import transforms

from config import WEIGHTS_DIR
from model import VisionTransformer

# Hyperparameter
IMAGE_SIZE = 64  # "The runs_copy resolution is 224"  # 32
PATCH_SIZE = 16  # patch_4, 8, 16
NUM_CHANNELS = 3
NUM_CLASSES = 10
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.2

BATCH_SIZE = 128
WORKERS = 4


def get_pretrained_model_EuroSAT(image_size: int, patch_size: int):
    best_weights = f'{WEIGHTS_DIR}/image_{image_size}/patch_{patch_size}/best.pt'
    model_saved = torch.load(best_weights, map_location='cpu')
    print("ViT bias numel", model_saved['model_state']['mlp_head.1.bias'].numel())
    return model_saved


def get_transform_EuroSAT(image_size: int):
    tr = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    return tr


def get_vit_EuroSAT(
        image_size: int, patch_size: int,
        num_channels: int = NUM_CHANNELS, num_classes: int = NUM_CLASSES,
        embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS, num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT):
    # image_size: "The runs_copy resolution is 224"  # 32
    # P - patch_size
    # C - num_channels
    # num_classes: "1000 classes in ImageNet"

    height = image_size  # H
    width = image_size  # W
    num_patches = (height * width) // (patch_size ** 2)  # N
    # num_patches = image_64
    # embed_dim = image_size

    model = VisionTransformer(embed_dim=embed_dim, hidden_dim=hidden_dim,
                              num_heads=num_heads, num_layers=num_layers,
                              patch_size=patch_size, num_channels=num_channels,
                              num_patches=num_patches, num_classes=num_classes, dropout=dropout)
    return model
