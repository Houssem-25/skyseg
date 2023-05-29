import argparse
from torchvision.datasets.coco import CocoDetection
class Option:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Script options')
        # Add arguments
        self.parser.add_argument('--data_root', type=str, required=True, help='Root directory of the Cityscapes dataset')
        self.parser.add_argument('--class_id', type=int, default=7, help='Specific class ID to evaluate')
        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
        self.parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
        self.parser.add_argument('--device', type=int, default=0, help='Device to use ( use -1 for cpu) ')
        self.parser.add_argument('--single_model', action='store_true', help='If set use a single model for infrence ')
        self.parser.add_argument('--epochs', type=int, default=20, help='The number of epoc to train to model for')
        self.parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for the optimizer')

    def parse(self):
        return self.parser.parse_args()