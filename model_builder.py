"""
Contains PyTorch model code to instantiate a VGG19 model.
"""
import torch
from torch import nn 

class VGG19(nn.Module):
    """
    Creates the VGG19 architecture.

    Replicates the VGG19, the a convolutional neural network introduced by Karen Simonyan, that consists of 19 layers, 16 convolution layers, and three fully connected layers.
    Original Resource : https://arxiv.org/abs/1409.1556 

    Args:
    num_classes: An integer indicating number of output units.
    kernel_size: An integer indicating number N for N x N shaped kernel box.
    padding: An integer indicating number of hidden outer values for each layer.
    padding: An integer indicating number of steps skipped for each iteration for kernel.
    """
    def __init__(self, num_classes=10, kernel_size=3, padding=1, stride=1):
        super(VGG19, self).__init__()
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv_3_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv_3_4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv_4_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv_4_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv_4_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv_5_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv_5_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv_5_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv_5_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes)
        )

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
    def forward(self, x):
        out = self.conv_1_1(x)
        out = self.conv_1_2(out)
        out = self.pool(out)
        out = self.conv_2_1(out)
        out = self.conv_2_2(out)
        out = self.pool(out)
        out = self.conv_3_1(out)
        out = self.conv_3_2(out)
        out = self.conv_3_3(out)
        out = self.conv_3_4(out)
        out = self.pool(out)
        out = self.conv_4_1(out)
        out = self.conv_4_2(out)
        out = self.conv_4_3(out)
        out = self.conv_4_4(out)
        out = self.pool(out)
        out = self.conv_5_1(out)
        out = self.conv_5_2(out)
        out = self.conv_5_3(out)
        out = self.conv_5_4(out)
        out = self.pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class Block(nn.Module):

    """
     a basic ResNet18 building block. It differs a little from larger ResNet architectures, but the overall logic is the same. The block consist of two convolutional layers and supports skip connections.
    """
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):

        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)

        return x

class ResNet18(nn.Module):
    """
    ResNet18 architecture. It has some feature extraction at the beginning and four ResNet blocks with skip connections. At the end we use adaptive average pooling (for the model to be agnostic to the input image size) and a fully-connected layer.

        Args:
            image_channels (int) : Number of image channels to be processed in the model.
            num_of_classes (int) : Number of classes in which the model will classify images into.
    """

    def __init__(self, num_classes: int):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)


    def __make_layer(self, in_channels, out_channels, stride) -> nn.Sequential:
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )


    def identity_downsample(self, in_channels, out_channels) -> nn.Sequential:
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )


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
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 