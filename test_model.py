import torch
from model import PneumoniaModel

model = PneumoniaModel()
dummy = torch.randn(4, 1, 224, 224)

output = model(dummy)

print("Форма выхода:", output.shape)
print("Число обучаемых параметров:", sum(p.numel() for p in model.parameters() if p.requires_grad))
