from datasets.cityscapes_dataset import CityscapesSegmentation
from torch.utils.data import DataLoader
from networks.model import SkySegmentation
from options import Option
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter


def downsample(tensor, factor=2):
    w, h = tensor.size()[-2:]
    return torch.nn.functional.interpolate(tensor, (w // factor, h // factor))
@torch.no_grad()
def compute_mean_iou(predictions, labels, num_classes=19):
    # Compute IoU for each class
    ious = []
    for class_id in range(num_classes):
        pred_mask = (predictions == class_id).float()
        label_mask = (labels == class_id).float()
        intersection = (pred_mask * label_mask).sum()
        union = (pred_mask + label_mask).sum() - intersection
        iou = intersection / (union + 1e-8)
        ious.append(iou.item())

    # Return the mean IoU
    mean_iou = sum(ious) / len(ious)
    return mean_iou


@torch.no_grad()
def evaluate(model, dataloader, writer, epoch,device):
    model.eval()
    total_iou = 0
    total_loss = 0
    num_samples = 0
    for images, labels in tqdm(dataloader):
        images = downsample(images.to(device))
        labels = downsample(labels.unsqueeze(1).to(device)).squeeze(1)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)

        # Compute mIoU
        iou = compute_mean_iou(predictions, labels)
        total_iou += iou * images.size(0)
        total_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

    avg_iou = total_iou / num_samples
    avg_loss = total_loss / num_samples

    writer.add_scalar('mIoU/eval', avg_iou, global_step=epoch)
    writer.add_scalar('Loss/eval', avg_loss, global_step=epoch)




def train(model, dataloader, optimizer, criterion, writer, epoch,device):
    model.train()
    total_loss = 0
    total_iou = 0
    num_samples = 0

    for images, labels in tqdm(dataloader):
        images = downsample(images.to(device), factor=16)
        labels = downsample(labels.unsqueeze(1).to(device), factor=16).squeeze(1).long()
        logits = model(images)
        logits = torch.nn.functional.interpolate(logits, (images.size(2),images.size(3)))
        loss = criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)

        # Compute mIoU
        iou = compute_mean_iou(predictions, labels)
        total_iou += iou * images.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

    avg_loss = total_loss / num_samples
    avg_iou = total_iou / num_samples

    writer.add_scalar('Loss/train', avg_loss, global_step=epoch)
    writer.add_scalar('mIoU/train', avg_iou, global_step=epoch)


# Load your trained model
if __name__ == "__main__":
    # init model
    opt = Option().parse()
    assert opt.single_model, "Train only with a single model. Please set --single_model in the arguments"
    model = SkySegmentation(opt)
    device = torch.device(opt.device)
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.lr)
    # Create the DataLoaders
    train_dataset = CityscapesSegmentation(opt.data_root, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.num_workers)
    eval_dataset = CityscapesSegmentation(opt.data_root, split='val')
    eval_dataloader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # Set up the criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    # Set up the tensorboard writer
    writer = SummaryWriter()

    # Training loop
    for epoch in range(opt.epochs):
        train(model, train_dataloader, optimizer, criterion, writer, epoch,device)
        evaluate(model, eval_dataloader,writer, epoch,device)

    # Close the tensorboard writer
    writer.close()