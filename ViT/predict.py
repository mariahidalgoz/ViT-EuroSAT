import argparse
from collections import namedtuple

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from dataset import EuroSAT, ImageFiles, random_split
from model import VisionTransformer
from config import IMAGE_SIZE, PATCH_SIZE, NUM_HEADS, NUM_LAYERS

# to be sure that we don't mix them, use this instead of a tuple
TestResult = namedtuple('TestResult', 'truth predictions')


@torch.no_grad()
def predict(model: nn.Module, dl: torch.utils.data.DataLoader, paths=None, show_progress=True):
    """
    Run the model on the specified data.
    Automatically moves the samples to the same device as the model.
    """
    if show_progress:
        dl = tqdm(dl, "Predict", unit="batch")
    device = next(model.parameters()).device

    model.eval()
    preds = []
    truth = []
    i = 0
    for images, labels in dl:
        images = images.to(device, non_blocking=True)
        pr, w = model(images)
        p = pr.argmax(1).tolist()
        preds += p
        truth += labels.tolist()

        if paths:
            for pred in p:
                print(f"{paths[i]!r}, {pred}")
                i += 1

    return TestResult(truth=torch.as_tensor(truth), predictions=torch.as_tensor(preds))


def report(result: TestResult, label_names):
    from sklearn.metrics import classification_report, confusion_matrix

    cr = classification_report(result.truth, result.predictions, target_names=label_names, digits=3)
    confusion = confusion_matrix(result.truth, result.predictions)

    try:  # add names if pandas is installed, otherwise don't bother but don't crash
        import pandas as pd

        # keep only initial for columns (or it's too wide when printed)
        confusion = pd.DataFrame(confusion, index=label_names, columns=[s[:3] for s in label_names])
    except ImportError:
        pass

    print("Classification report")
    print(cr)
    print("Confusion matrix")
    print(confusion)


def get_pretrained_model_EuroSAT(best_weights: str = '../weights/best.pt'):
    model_saved = torch.load(best_weights, map_location='cpu')
    print("ViT bias numel", model_saved['model_state']['mlp_head.1.bias'].numel())
    return model_saved


def get_transform_EuroSAT(image_size: int = IMAGE_SIZE):
    tr = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    return tr


def get_vit_EuroSAT(image_size: int = IMAGE_SIZE, patch_size: int = PATCH_SIZE, num_channels: int = 3, num_classes: int = 10,
                    embed_dim: int = 256, hidden_dim: int = 512, num_heads: int = NUM_HEADS, num_layers: int = NUM_LAYERS,
                    dropout: float = 0.2):
    # image_size: "The training resolution is 224"  # 32
    # P - patch_size
    # C - num_channels
    # num_classes: "1000 classes in ImageNet"

    height = image_size  # H
    width = image_size  # W
    num_patches = (height * width) // (patch_size ** 2)  # N
    # num_patches = 64
    # embed_dim = image_size

    model = VisionTransformer(embed_dim=embed_dim, hidden_dim=hidden_dim,
                              num_heads=num_heads, num_layers=num_layers,
                              patch_size=patch_size, num_channels=num_channels,
                              num_patches=num_patches, num_classes=num_classes, dropout=dropout)
    return model


def main(args):
    model_saved = get_pretrained_model_EuroSAT(args.model)

    model = get_vit_EuroSAT()
    model.load_state_dict(model_saved['model_state'])
    model = model.to(args.device)

    tr = get_transform_EuroSAT()
    normalization = model_saved['normalization']
    tr.transforms.append(transforms.Normalize(**normalization))

    if args.files:
        test = ImageFiles(args.files, transform=tr)
    else:
        dataset = EuroSAT(transform=tr)
        trainval, test = random_split(dataset, 0.9, random_state=42)

    test_dl = torch.utils.data.DataLoader(
        test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    result = predict(model, test_dl, paths=args.files)

    if not args.files:  # this is the test, so we need to analyze results
        report(result, dataset.classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Predict the label on the specified files and outputs the results in csv format.
            If no file is specified, then run on the test set of ResNet50_EuroSAT and produce a report.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-m', '--model', default='weights/best.pt', type=str, help="Model to use for prediction"
    )
    parser.add_argument(
        '-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help="Number of workers for the DataLoader",
    )
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')  # "The ViT paper states the use of a batch size of 4096 for training"
    parser.add_argument('files', nargs='*', help="Files to run prediction on")
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
