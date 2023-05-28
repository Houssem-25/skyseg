import torch
import torch.nn as nn
from mmengine.config import Config
from mmseg.apis import init_model, inference_model
import torch.nn.functional as F

class SkySegmentation(nn.Module):
    def __init__(self, opt):
        super(SkySegmentation, self).__init__()
        self.single_model = opt.single_model
        self.opt = opt
        config_file = 'networks/configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py'
        cfg = Config.fromfile(config_file)
        self.model_1 = init_model(cfg, cfg._cfg_dict["checkpoint_file"], device='cuda:0')
        self.sky_id_model_1 = self.model_1.dataset_meta["classes"].index("sky")

        if not self.single_model:
            config_file = 'networks/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'
            cfg = Config.fromfile(config_file)
            self.model_2 = init_model(cfg, cfg._cfg_dict["checkpoint_file"], device='cuda:0')
            self.sky_id_model_2 = self.model_2.dataset_meta["classes"].index("sky")


    def forward(self, image):
        if self.single_model:
            logits_model = self.model_1(image)
            return logits_model
        else:
            logits_model_1 = self.model_1(image)
            logits_model_2 = self.model_2(image)
            return logits_model_1, logits_model_2

    @torch.no_grad()
    def segment_sky(self, image):
        _, _, w, h = image.size()

        if self.single_model:
            logits_model = self.forward(image)
            logits_model = torch.nn.functional.interpolate(logits_model, (w, h))
            mask_model = self.logits_to_mask(logits_model) == self.sky_id_model_1
            return mask_model
        else:
            logits_model_1, logits_model_2 = self.forward(image)
            logits_model_1 = torch.nn.functional.interpolate(logits_model_1, (w, h))
            logits_model_2 = torch.nn.functional.interpolate(logits_model_2, (w, h))

            mask_model_1 = self.logits_to_mask(logits_model_1) == self.sky_id_model_1
            mask_model_2 = self.logits_to_mask(logits_model_2) == self.sky_id_model_2

            mask_model_1, mask_model_2 = self.aggregate_masks(mask_model_1, mask_model_2)
            return mask_model_1, mask_model_2

    def logits_to_mask(self, logits):
        # Apply softmax activation along the class dimension
        probabilities = F.softmax(logits, dim=1)

        # Get the class index with the maximum probability for each pixel
        _, predicted_classes = torch.max(probabilities, dim=1)

        return predicted_classes

    def aggregate_masks(self, mask1, mask2):
        return mask1 + mask2, mask1 * mask2