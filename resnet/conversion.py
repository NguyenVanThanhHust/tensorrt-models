import sys
import torchvision
import torch
from loguru import logger

sys.path.append('../third_party/torch2trt')
from torch2trt import torch2trt
from torch2trt import TRTModule

@logger.catch
@torch.no_grad()
def main():
    model = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1').cuda().eval()
    data = torch.randn((1, 3, 224, 224)).cuda()
    model_trt = torch2trt(model, 
                          [data], 
                          fp16_mode=True,
                          max_workspace_size=(1 << 32),
                          max_batch_size=1
                          )
    output_trt = model_trt(data)
    
    output = model(data)

    print(output.flatten()[0:10])
    print(output_trt.flatten()[0:10])
    print('max error: %f' % float(torch.max(torch.abs(output - output_trt))))

    torch.save(model_trt.state_dict(), 'resnet18_trt.pth')

    model_trt = TRTModule()

    model_trt.load_state_dict(torch.load('resnet18_trt.pth'))

    engine_file = "resnet_trt.engine"
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())

if __name__ == '__main__':
    main()