from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from torchvision import transforms

from data import MEAN, STD
from model import PneumoniaModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(__file__).with_name("pneumonia_model.pth")


class Predictor:
    def __init__(self, model_path=MODEL_PATH):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.classes = checkpoint["classes"]
        self.threshold = checkpoint["threshold"]
        img_size = checkpoint["img_size"]

        self.model = PneumoniaModel().to(DEVICE)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def predict(self, image):
        x = self.transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return torch.sigmoid(self.model(x)).item()

    def predict_with_cam(self, image):
        prob, cam = self.gradcam(image)
        return prob, overlay_heatmap(image, cam)

    def gradcam(self, image):
        x = self.transform(image).unsqueeze(0).to(DEVICE)
        activations, gradients = {}, {}

        layer = self.model.last_conv_layer
        fwd = layer.register_forward_hook(lambda m, i, o: activations.update(value=o))
        bwd = layer.register_full_backward_hook(lambda m, gi, go: gradients.update(value=go[0]))
        try:
            self.model.zero_grad()
            logit = self.model(x)
            logit.backward()
        finally:
            fwd.remove()
            bwd.remove()

        prob = torch.sigmoid(logit).item()

        acts, grads = activations["value"][0], gradients["value"][0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=0)).detach().cpu().numpy()
        cam = cam / cam.max() if cam.max() > 0 else cam

        return prob, cam


def overlay_heatmap(image, cam, alpha=0.4):
    base = image.convert("RGB")
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(base.size, Image.BILINEAR)
    heat = colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    blended = (1 - alpha) * (np.asarray(base) / 255.0) + alpha * heat
    return Image.fromarray(np.uint8(blended * 255))
