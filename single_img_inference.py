from model import SkySegmentation
import torch
from  PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def plot_mask_overlay(image, mask):
    # Convert mask to a binary array
    mask = mask.detach().cpu().numpy()[0]
    mask_binary = mask.astype(np.uint8)

    # Create a copy of the image
    image_copy = np.array(image)

    # Apply mask overlay by setting masked pixels to a specific color (e.g., green)
    image_copy[mask_binary == 1] = [0, 255, 0]  # Modify the color as per your preference

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Plot the image with mask overlay
    plt.subplot(1, 2, 2)
    plt.imshow(image_copy)
    plt.title('Segmentation Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

transform = transforms.ToTensor()
image = Image.open("demo.jpeg")
tensor_image = transform(image).cuda()


model = SkySegmentation("")
out1, out2 = model.segment_sky(tensor_image.unsqueeze(0))
plot_mask_overlay(image, out1)
plot_mask_overlay(image, out2)