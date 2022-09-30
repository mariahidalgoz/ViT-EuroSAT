import argparse
import torch
from collections import namedtuple
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from dataset import EuroSAT, ImageFiles, random_split
from utilsEuroSAT import get_pretrained_model_EuroSAT, get_vit_EuroSAT, get_transform_EuroSAT

# to be sure that we don't mix them, use this instead of a tuple
TestResult = namedtuple('TestResult', 'truth predictions')


@torch.no_grad()
def predict(model: nn.Module, dl: torch.utils.data.DataLoader, paths=None, show_progress=True):
    """
    Run the model on the specified data2.
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


def main(args):
    model_saved = get_pretrained_model_EuroSAT(args.image_size, args.patch_size)

    model = get_vit_EuroSAT(args.image_size, args.patch_size)
    model.load_state_dict(model_saved['model_state'])
    model = model.to(args.device)

    tr = get_transform_EuroSAT(args.image_size)
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
        '-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help="Number of workers for the DataLoader",
    )
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')  # "The ViT paper states the use of a batch size of 4096 for training"
    parser.add_argument(
        '-is', '--image-size', default=64, type=int, help="Image size of the model to use for prediction"
    )
    parser.add_argument(
        '-ps', '--patch-size', default=16, type=int, help="Patch size of the model to use for prediction"
    )
    parser.add_argument('files', nargs='*', help="Files to run prediction on")
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
