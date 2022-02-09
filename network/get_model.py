from torch import nn
from se_viewfusion_resnet50 import se_viewfusion_resnet50

num_class = 431


def get_model(model, pretrained=False):
    if model == 'se_viewfusion_resnet50':
        model = se_viewfusion_resnet50(num_classes=1000, pretrained=pretrained)
        model.last_linear = nn.Linear(2048, 431)

    return model

