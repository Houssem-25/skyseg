import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
import torch
import torchvision.transforms as transforms
from networks.model import SkySegmentation
import numpy as np
from PIL import Image
from options import Option

# Define the GUI application
class SegmentationApp:
    def __init__(self, root):
        self.opt = Option().parse()
        self.model =  SkySegmentation(self.opt).cuda()
        self.model.eval()
        self.root = root
        self.root.title("Image Segmentation App")
        self.root.geometry("800x600")

        self.image_path = None
        self.segmented_image = None

        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        self.seg_label = tk.Label(self.root)
        self.seg_label.pack()

        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.segment_button = tk.Button(self.root, text="Segment Image", command=self.segment_image)
        self.segment_button.pack()

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=(("all files", "*.*"), ("Image files", "*.jpg;*.jpeg;*.png")))
        if self.image_path:
            image = Image.open(self.image_path)
            image.thumbnail((500, 500))  # Resize for display
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

    def segment_image(self):
        if self.image_path:
            image = Image.open(self.image_path)
            image_tensor = transforms.ToTensor()(image).unsqueeze(0).cuda()  # Add batch dimension

            with torch.no_grad():
                mask,_ = self.model.segment_sky(image_tensor)

            mask = mask.detach().cpu().numpy()[0]
            mask_binary = mask.astype(np.uint8)

            # Create a copy of the image
            image_copy = np.array(image)

            # Apply mask overlay by setting masked pixels to a specific color (e.g., green)
            image_copy[mask_binary == 1] = [0, 255, 0]  # Modify the color as per your preference

            # Apply any visualization or post-processing to the mask as desired

            # Create a PIL image from the mask

            image= Image.fromarray(image_copy)
            image.thumbnail((500, 500))
            photo = ImageTk.PhotoImage( image )
            self.seg_label.configure(image=photo)
            self.seg_label.image = photo

# Create the Tkinter root window
root = tk.Tk()

# Create an instance of the SegmentationApp
app = SegmentationApp(root)

# Run the Tkinter event loop
root.mainloop()