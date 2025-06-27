from resnet34 import resnet34
import torch
from torchsummary import summary
import struct

if __name__ == '__main__':
    model_path = 'resnet_best.pth'
    net = resnet34(num_classes=100, pretrained=model_path)
    print("model_structure")
    print(net)

    net = net.to("cuda:0")
    net.eval()
    print("model: ", net)
    tmp = torch.ones(1, 3, 224, 224).to("cuda:0")
    print("input: ", tmp)
    out = net(tmp)
    print("output:", out)

    summary(net, (3, 224, 224))

    f = open("resnet34.wts", "w")
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k, v in net.state_dict().items():
        print("key: ", k)
        print("value: ", v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
