from datasets.cityscapes_dataset import CityscapesSegmentation
from torch.utils.data import DataLoader
from networks.model import SkySegmentation
from options import Option
from tqdm import tqdm
import torch
def downsample(tensor, factor=2):
    w,h = tensor.size()[-2:]
    return torch.nn.functional.interpolate(tensor, (w//factor,h//factor))

@torch.no_grad()
def evaluate(model, dataloader, class_id, single_model):
    model.eval()
    with torch.no_grad():
        total_correct_1 = 0
        total_correct_2 = 0
        total_pixels = 0
        for images, labels in tqdm(dataloader):
            images = downsample(images.cuda())
            labels = downsample(labels.unsqueeze(1).cuda()).squeeze(1)
            labels = (labels == class_id)
            if single_model:
                mask_model_1 = model.segment_sky(images)
            else:
                mask_model_1,mask_model_2 = model.segment_sky(images)
                correct = (mask_model_2 == labels).sum()
                total_correct_2 += correct.item()

            correct = (mask_model_1 == labels).sum()
            total_correct_1 += correct.item()


            total_pixels += labels.numel()

        return total_correct_1 / total_pixels , total_correct_2 / total_pixels


# Load your trained model
if __name__ == "__main__":
    # init model
    opt = Option().parse()
    model = SkySegmentation(opt)
    class_id = model.sky_id_model_1
    eval_dataset = CityscapesSegmentation(opt.data_root, split='val')
    device = torch.device(opt.device)

    # Create the DataLoader for evaluation
    batch_size = 1
    num_workers = 4
    eval_dataloader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # Evaluate the model
    accuracy_1, accuracy_2 = evaluate(model, eval_dataloader, class_id, opt.single_model)
    print(f"Accuracy for class {class_id} using OR : {accuracy_1}")
    print(f"Accuracy for class {class_id} using AND: {accuracy_2}")