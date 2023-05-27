import torch
from dataloader import CityscapesSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
from model import SkySegmentation

def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        total_correct_1 = 0
        total_correct_2 = 0
        total_pixels = 0
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda()

            mask_model_1, mask_model_2 = model.segment_sky(images)

            correct = (mask_model_1 == labels).sum()
            total_correct_1 += correct.item()
            correct = (mask_model_2 == labels).sum()
            total_correct_2 += correct.item()

            total_pixels += labels.numel()

        return total_correct_1 / total_pixels , total_correct_2 / total_pixels


# Load your trained model
if __name__ == "__main__":
    # init model
    model = SkySegmentation("")
    # Set the root directory of the Cityscapes dataset
    data_root = '/path/to/cityscapes/dataset/'
    # Set the specific class ID to evaluate
    class_id = model.sky_id_model_1  # Modify this based on the specific class ID you want to evaluate
    # Create the Cityscapes dataset for evaluation
    eval_dataset = CityscapesSegmentation(data_root, split='val')

    # Filter the dataset to keep only the specific class ID
    filtered_dataset = []
    for image, label in eval_dataset:
        filtered_label = (label == class_id).long()
        filtered_dataset.append((image, filtered_label))

    # Create the DataLoader for evaluation
    batch_size = 1
    num_workers = 4
    eval_dataloader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Evaluate the model
    accuracy_1, accuracy_2 = evaluate(model, eval_dataloader)
    print(f"Accuracy for class {class_id} using OR : {accuracy_1}")
    print(f"Accuracy for class {class_id} using AND: {accuracy_2}")