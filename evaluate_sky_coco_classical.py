from datasets.coco_sky_dataset import CocoSemantic
from torch.utils.data import DataLoader
from networks.model import SkySegmentation
from options import Option
from tqdm import tqdm
import torch
from classical_solution import segment_sky
def downsample(tensor, factor=2):
    w,h = tensor.size()[-2:]
    return torch.nn.functional.interpolate(tensor, (w//factor,h//factor))

@torch.no_grad()
def evaluate(dataloader):
    total_correct_1 = 0
    total_pixels=0
    for images, labels in tqdm(dataloader):
        images = downsample(images)
        labels = downsample(labels).squeeze(1)[0].numpy()
        _, mask_model = segment_sky(images.permute(0,2,3,1).numpy()[0])
        correct = (mask_model == labels).sum()
        total_correct_1 += correct.item()
        total_pixels += labels.size

    return total_correct_1 / total_pixels


# Load your trained model
if __name__ == "__main__":
    # init model
    opt = Option().parse()
    eval_dataset = CocoSemantic(opt.data_root)
    # Create the DataLoader for evaluation
    batch_size = 1
    num_workers = 0
    eval_dataloader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    # Evaluate the model
    accuracy_1 = evaluate(eval_dataloader)
    print(accuracy_1)
