import torch
from torch import nn
import torchvision
import os
import struct
from torchsummary import summary

def main():
    model = torchvision.models.alexnet(pretrained=False)

    checkpoint = torch.load("./weights/alexnet-owt-7be5be79.pth")
    model.load_state_dict(checkpoint)
    model.eval()
    model = model.cuda()

    device = torch.device("cuda")
    dummy_input = torch.ones(1, 3, 224, 224).cuda()

    output = model(dummy_input)
    summary(model, (3, 224, 224))

    f = open("alexnet.wts", 'w')
    f.write("{}\n".format(len(model.state_dict().keys())))
    for k,v in model.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

if __name__ == "__main__":
    main()
