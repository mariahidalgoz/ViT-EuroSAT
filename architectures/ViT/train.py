import argparse
import os
import shutil

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm, trange

from dataset import EuroSAT, random_split
from predict import predict, get_vit_EuroSAT
from config import WEIGHTS_DIR, RUNS_DIR, ROOT_PATH


class State:
    # Keep some global state here (ex best accuracy on val)
    best_acc = 0
    writer: SummaryWriter = None
    normalization = None


def calc_normalization(train_dl: torch.utils.data.DataLoader):
    "Calculate the mean and std of each channel on images from `train_dl`"
    mean = torch.zeros(3)
    m2 = torch.zeros(3)
    n = len(train_dl)
    for images, labels in tqdm(train_dl, "Compute normalization"):
        mean += images.mean([0, 2, 3]) / n
        m2 += (images ** 2).mean([0, 2, 3]) / n
    var = m2 - mean ** 2
    return mean, var.sqrt()


def main(args):
    root_weights_image_size = f'{ROOT_PATH}{WEIGHTS_DIR}/image_{args.image_size}/patch_{args.patch_size}'
    root_runs_image_size = f'{ROOT_PATH}{RUNS_DIR}/image_{args.image_size}/patch_{args.patch_size}'
    os.makedirs(root_weights_image_size, exist_ok=True)

    height = args.image_size         # H
    width = args.image_size          # W

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((height, width), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    dataset = EuroSAT(transform=transform)
    train_val, test_ds = random_split(dataset, 0.9, random_state=42)
    train_ds, val_ds = random_split(train_val, 0.9, random_state=7)


    # We define a set of data loaders that we can use for various purposes later.
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True
    )

    # load train dataset with computed normalization
    mean, std = calc_normalization(train_dl)
    dataset.transform.transforms.append(transforms.Normalize(mean, std))
    State.normalization = {'mean': mean, 'std': std}

    # load val dataset
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.image_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True
    )

    # create/load model, changing the head for our number of classes
    model = get_vit_EuroSAT(args.image_size, args.patch_size)

    # if args.pretrained:
    #     for param in model.parameters():
    #         param.requires_grad = False
    # model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model = model.to(args.device)
    loss = nn.CrossEntropyLoss()

    # params = model.fc.parameters() if args.pretrained else model.parameters()
    # optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    State.writer = SummaryWriter(root_runs_image_size)
    # display some examples in tensorboard
    images, labels = next(iter(train_dl))
    originals = images * std.view(3, 1, 1) + mean.view(3, 1, 1)
    State.writer.add_images('images/original', originals, 0)
    State.writer.add_images('images/normalized', images, 0)
    # State.writer.add_graph(model, images)

    for epoch in trange(args.epochs, desc="Epochs"):
        train_epoch(train_dl, model, loss, optimizer, epoch, args)
        truth, preds = predict(model, val_dl)

        torch.save(
            {'normalization': State.normalization, 'model_state': model.state_dict()},
            f'{root_weights_image_size}/checkpoint.pt',
        )

        val_acc = (truth == preds).float().mean()
        State.writer.add_scalar('acc/val', val_acc, epoch * len(train_dl))
        if val_acc > State.best_acc:
            print(f"New best validation accuracy: {val_acc}")
            State.best_acc = val_acc
            shutil.copy(f'{root_weights_image_size}/checkpoint.pt',
                        f'{root_weights_image_size}/best.pt')


def train_epoch(train_dl, model, loss, optimizer, epoch, args):
    model.train()
    train_dl = tqdm(train_dl, "Train", unit="batch")
    for i, (images, labels) in enumerate(train_dl):
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)

        preds, _ = model(images)
        _loss = loss(preds, labels)
        acc = (labels == preds.argmax(1)).float().mean()

        optimizer.zero_grad()
        _loss.backward()
        optimizer.step()

        State.writer.add_scalar('loss/train', _loss, epoch * len(train_dl) + i)
        State.writer.add_scalar('acc/train', acc, epoch * len(train_dl) + i)


if __name__ == '__main__':

    def parse_bool(s: str):
        if s.casefold() in ['1', 'true', 'yes']:
            return True
        if s.casefold() in ['0', 'false', 'no']:
            return False
        raise ValueError()


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-j', '--workers', default=4, type=int, metavar='N', help="Number of workers for the DataLoader",
    )
    parser.add_argument('--epochs', default=5, type=int, metavar='N', help="Epochs")
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help="Batch size")
    parser.add_argument(
        '--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help="Learning rate"
    )
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument(
        '--wd', '--weight-decay', default=0, type=float, metavar='WD', help="Weight decay"
    )
    parser.add_argument(
        '--pretrained', default=True, type=parse_bool, help="Finetune a pre-trained model"
    )
    parser.add_argument(
        '-is', '--image-size', default=64, type=int, help="Image size"
    )
    parser.add_argument(
        '-ps', '--patch-size', default=8, type=int, help="Patch size"
    )
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
