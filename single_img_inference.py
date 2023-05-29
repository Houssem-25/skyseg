from networks.model import SkySegmentation
from  PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from options import Option

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



opt = Option().parse()

image = Image.open(opt.data_root)
transform = transforms.ToTensor()
tensor_image = transform(image).cuda()

model = SkySegmentation(opt)
mask_model_1, mask_model_2 = model.segment_sky(tensor_image.unsqueeze(0))
plot_mask_overlay(image, mask_model_1)
plot_mask_overlay(image, mask_model_2)