import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1 # For the basic block, output channels are same as input channels times expansion

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample # Optional downsampling layer for shortcut connection
        self.stride = stride

    def forward(self, x):
        identity = x # Store the input for the skip connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsample to the identity if dimensions don't match
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity # Add the skip connection (residual)
        out = self.relu(out) # Apply ReLU after adding the shortcut

        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64 # Initial number of channels after the first conv

        # Initial convolutional layer and pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        # layers parameter is a list: [num_blocks_layer1, num_blocks_layer2, ...]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Final average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Output size (1,1) regardless of input
        self.fc = nn.Linear(512 * block.expansion, num_classes) # 512 * 1 for BasicBlock

        # Initialize weights (common practice)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        # Check if downsampling is needed for the shortcut connection
        # This happens if stride > 1 (spatial downsampling) OR
        # if the number of input channels doesn't match the expected output channels * expansion
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # Add the first block in the layer (potentially with downsampling)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion # Update in_channels for subsequent blocks

        # Add remaining blocks (with stride 1)
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Flatten for the fully connected layer
        x = self.fc(x)

        return x
    
def resnet34(num_classes=1000, pretrained=None):
    """
    ResNet-34 model implementation.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model

if __name__ == '__main__':
    # Create an instance of ResNet-34
    model = resnet34(num_classes=1000) # Example for ImageNet classification

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Print the model architecture
    print("Custom ResNet-34 Architecture:")
    print(model)

    # Test with a dummy input
    dummy_input = torch.randn(16, 3, 224, 224).to(device) # Batch size 1, 3 channels, 224x224 image
    output = model(dummy_input)

    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be [1, num_classes] (e.g., [1, 1000])

    # Example for a custom classification task (e.g., 10 classes)
    print("\n--- ResNet-34 for a custom 10-class task ---")
    custom_model = resnet34(num_classes=10)
    custom_model.to(device)
    print(custom_model)
    dummy_output_custom = custom_model(dummy_input)
    print(f"Output shape for 10 classes: {dummy_output_custom.shape}")