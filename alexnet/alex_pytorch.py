import numpy as np
import torch
import torchvision

if __name__ == "__main__":
    model = torchvision.models.alexnet(weights="AlexNet_Weights.DEFAULT")
    model.eval()

    with torch.no_grad():
        x = torch.rand((1, 3, 224, 224), dtype=torch.float32)
        output = model(x)
        print(output.shape)
