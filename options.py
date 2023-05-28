import argparse

class Option:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Script options')
        # Add arguments
        self.parser.add_argument('--data_root', type=str, required=True, help='Root directory of the Cityscapes dataset')
        self.parser.add_argument('--class_id', type=int, default=7, help='Specific class ID to evaluate')
        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
        self.parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
        self.parser.add_argument('--device', type=int, default=0, help='Device to use ( use -1 for cpu) ')

    def parse(self):
        return self.parser.parse_args()