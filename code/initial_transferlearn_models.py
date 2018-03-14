import torch.nn as nn
from torchvision import models
def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

class BaseModel(nn.Module):
    def __init__(self, resnet):
        super(BaseModel, self).__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        #self.fc = nn.Linear(512, 38)
        #weights_init(self.fc)
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
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x

class FCModel(nn.Module):
    def __init__(self, input_dim):
        super(FCModel, self).__init__()
        self.fc = nn.Linear(input_dim, 38)
        weights_init(self.fc)
    def forward(self, x):
        x = self.fc(x)
        return x

def resnet18Base(model):
    base_model = BaseModel(model)
    return base_model

def resnet34Base(model):
    base_model = BaseModel(model)
    return base_model

def resnet50Base(model):
    base_model = BaseModel(model)
    return base_model

def resnet101Base(model):
    base_model = BaseModel(model)
    return base_model

def resnet152Base(model):
    base_model = BaseModel(model)
    return base_model

def resnet18t():
    retrain_model = FCModel(512)
    return retrain_model

def resnet34t():
    retrain_model = FCModel(512)
    return retrain_model

def resnet50t():
    retrain_model = FCModel(2048)
    return retrain_model

def resnet101t():
    retrain_model = FCModel(2048)
    return retrain_model

def resnet152t():
    retrain_model = FCModel(2048)
    return retrain_model
