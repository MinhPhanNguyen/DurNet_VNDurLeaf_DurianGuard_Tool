import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.rep = nn.Sequential()
        filters = in_filters
        if grow_first:
            self.rep.add_module('relu1', nn.ReLU(True))
            self.rep.add_module('conv1', SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1))
            self.rep.add_module('bn1', nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            self.rep.add_module(f'relu{i + 2}', nn.ReLU(True))
            self.rep.add_module(f'conv{i + 2}', SeparableConv2d(filters, filters, 3, stride=1, padding=1))
            self.rep.add_module(f'bn{i + 2}', nn.BatchNorm2d(filters))

        if not grow_first:
            self.rep.add_module('relu_final', nn.ReLU(True))
            self.rep.add_module('conv_final', SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1))
            self.rep.add_module('bn_final', nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            self.rep = self.rep[1:]
        else:
            self.rep[0] = nn.ReLU(False)

        if strides != 1:
            self.rep.add_module('maxpool', nn.MaxPool2d(3, strides, 1))

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

class DurNet(nn.Module):
    """
    Modified Xception for Durian Disease Classification
    Adapted for 6 classes: ['Leaf_Blight', 'Leaf_Rhizoctonia', 'Leaf_Phomopsis', 'Leaf_Algal', 'Leaf_Colletotrichum', 'Leaf_Healthy']
    """
    def __init__(self, num_classes=6):
        super(DurNet, self).__init__()
        self.num_classes = num_classes

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(True)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # Middle flow
        self.middle_blocks = nn.ModuleList()
        for i in range(8):
            self.middle_blocks.append(Block(728, 728, 3, 1, start_with_relu=True, grow_first=True))

        # Exit flow
        self.block4 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(True)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4 = nn.ReLU(True)

        # Classifier
        self.fc = nn.Linear(2048, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        for block in self.middle_blocks:
            x = block(x)

        # Exit flow
        x = self.block4(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.fc(x)

        return x

def create_durnet(num_classes=6, pretrained=False):
    """Create DurNet model"""
    model = DurNet(num_classes=num_classes)
    
    if pretrained:
        # Load pretrained weights if available
        try:
            checkpoint = torch.load('durnet.pth', map_location='cpu')
            model.load_state_dict(checkpoint, strict=True)
            print("Loaded pretrained DurNet weights")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
    
    return model

# For backward compatibility
def durnet(num_classes=6, pretrained=False):
    return create_durnet(num_classes, pretrained)