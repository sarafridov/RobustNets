# Contains model definitions used for models in the RobustNets dataset.

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import flatten, rand, zeros_like
from utilities import sparse_dict_to_dense_dict as load_sparse_dict

class Conv8(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            conv3x3(3, 64),
            nn.ReLU(),
            conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(64, 128),
            nn.ReLU(),
            conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(128, 256),
            nn.ReLU(),
            conv3x3(256, 256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(256, 512),
            nn.ReLU(),
            conv3x3(512, 512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.output = conv1x1(256, 10)
        self.linear = nn.Sequential(
            conv1x1(512 * 2 * 2, 256),
            nn.ReLU(),
            conv1x1(256, 256),
            nn.ReLU(),
            self.output
        )

        self.apply(init_fn)

    def forward(self, x):
        out = self.convs(x)
        out = out.reshape(out.size(0), 512 * 2 * 2, 1, 1)
        out = self.linear(out)
        return out.view(-1,10)

# Functions for models
def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResNet18(torchvision.models.ResNet):
    def __init__(self):
        nn.Module.__init__(self)  # Skip the parent constructor. This replaces it.
        block = torchvision.models.resnet.BasicBlock
        layers = [2, 2, 2, 2]
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # The initial convolutional layer.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # The subsequent blocks.
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 64*2, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(block, 64*4, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(block, 64*8, layers[3], stride=2, dilate=False)

        # The last layers.
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(64*8*block.expansion, 10)
        
        self.apply(init_fn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.fc(x)

        return x

class VGG16(nn.Module):
    """A VGG16-style neural network designed for CIFAR-10."""

    class ConvModule(nn.Module):
        """A single convolutional module in a VGG network."""

        def __init__(self, in_filters, out_filters):
            super().__init__()
            self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_filters)

        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))

    def __init__(self):
        super().__init__()

        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]

        layers = []
        filters = 3

        for spec in plan:
            if spec == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(VGG16.ConvModule(filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, 10)

        self.apply(init_fn)

    def forward(self, x):
        x = self.layers(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def kaiming_normal_conv_linear(w):
    if isinstance(w, nn.Linear) or isinstance(w, nn.Conv2d):
        nn.init.kaiming_normal_(w.weight)

def uniform_bn(w):
    if isinstance(w, nn.BatchNorm2d):
        w.weight.data = rand(w.weight.data.shape)
        w.bias.data = zeros_like(w.bias.data)
        
def init_fn(w):
    kaiming_normal_conv_linear(w)
    uniform_bn(w)

# ----- BPEP Model Definitions ----- #
# Defining models used for biprop, edgepopup, GMP 
# here as they have minor differences (e.g. in layer
# names) from LRR models.

# --- Conv8 --- #
class cConv8(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            conv3x3(3, 64),
            nn.ReLU(),
            conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(64, 128),
            nn.ReLU(),
            conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(128, 256),
            nn.ReLU(),
            conv3x3(256, 256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(256, 512),
            nn.ReLU(),
            conv3x3(512, 512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            conv1x1(512 * 2 * 2, 256),
            nn.ReLU(),
            conv1x1(256, 256),
            nn.ReLU(),
            conv1x1(256, 10)
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.reshape(out.size(0), 512 * 2 * 2, 1, 1)
        out = self.linear(out)
        return out.view(-1,10)
        #return out.squeeze()

# --- ResNet18 --- #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, affine_bn=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_bn)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_bn)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride),
                nn.BatchNorm2d(self.expansion * planes, affine=affine_bn),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, affine_bn):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64, stride=1)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_bn)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1, affine_bn=affine_bn)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, affine_bn=affine_bn)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, affine_bn=affine_bn)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, affine_bn=affine_bn)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = conv1x1(512 * block.expansion, 10)

    def _make_layer(self, block, planes, num_blocks, stride, affine_bn):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, affine_bn))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = self.fc(out)
        return out.flatten(1)

def cResNet18(affine_bn=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], affine_bn)

# --- VGG16 --- #
class cVGG16(nn.Module):
    def __init__(self, affine_bn=True):
        super(cVGG16, self).__init__()
        self.convs = nn.Sequential(
            # Two 64 blocks
            conv3x3(3, 64),
            nn.BatchNorm2d(64, affine=affine_bn),
            nn.ReLU(),
            conv3x3(64, 64),
            nn.BatchNorm2d(64, affine=affine_bn),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # Two 128 blocks
            conv3x3(64, 128),
            nn.BatchNorm2d(128, affine=affine_bn),
            nn.ReLU(),
            conv3x3(128, 128),
            nn.BatchNorm2d(128, affine=affine_bn),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # Three 256 blocks
            conv3x3(128, 256),
            nn.BatchNorm2d(256, affine=affine_bn),
            nn.ReLU(),
            conv3x3(256, 256),
            nn.BatchNorm2d(256, affine=affine_bn),
            nn.ReLU(),
            conv3x3(256, 256),
            nn.BatchNorm2d(256, affine=affine_bn),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # Three 512 blocks
            conv3x3(256, 512),
            nn.BatchNorm2d(512, affine=affine_bn),
            nn.ReLU(),
            conv3x3(512, 512),
            nn.BatchNorm2d(512, affine=affine_bn),
            nn.ReLU(),
            conv3x3(512, 512),
            nn.BatchNorm2d(512, affine=affine_bn),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # Three 512 blocks
            conv3x3(512, 512),
            nn.BatchNorm2d(512, affine=affine_bn),
            nn.ReLU(),
            conv3x3(512, 512),
            nn.BatchNorm2d(512, affine=affine_bn),
            nn.ReLU(),
            conv3x3(512, 512),
            nn.BatchNorm2d(512, affine=affine_bn),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
          conv1x1(512, 10),
        )

        self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.convs(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), 512, 1, 1)
        out = self.linear(out)
        return out.squeeze()

# Code for loading models
def instantiate_model(model_string, PATH_TO_RobustNets):
    """
    Instantiates model in the RobustNets dataset, based on the
    sparsification algorithm used when training the model.
    """
    state_dict_name = model_string + '_state_dict.pt'
    model_name = model_string.split('_')[0]
    model_dict = {'Conv8': Conv8, 
                  'ResNet18': ResNet18, 
                  'VGG16': VGG16,
                  'cConv8': cConv8, 
                  'cResNet18': cResNet18, 
                  'cVGG16': cVGG16}
    if ('lrr' in model_string) or ('lth' in model_string):
        model = model_dict[model_name]()
    elif ('Conv8' in model_string) or ('GMP' in model_string) or ('FT' in model_string):
        model = model_dict[f"c{model_name}"]()
    else: #('biprop' in model_string) or ('edgepopup' in model_string):
        # biprop and edgepopup do not use trainable batchnorm parameters
        model = model_dict[f"c{model_name}"](affine_bn=False)
    model.load_state_dict(load_sparse_dict(PATH_TO_RobustNets/state_dict_name))
    return model 
