import torch
import torch.nn as nn
import torchvision as tv


def get_resnet50(num_classes, pretrained=True):
    """
    获取ResNet50模型
    
    参数:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    返回:
        配置好的ResNet50模型
    """
    if pretrained:
        # 使用新版本的权重加载方式
        weights = tv.models.ResNet50_Weights.IMAGENET1K_V1
        model = tv.models.resnet50(weights=weights)
    else:
        model = tv.models.resnet50(pretrained=False)
    
    # 替换最后一层全连接层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def get_efficientnet_b0(num_classes, pretrained=True):
    """获取EfficientNet-B0模型（可选）"""
    if pretrained:
        weights = tv.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = tv.models.efficientnet_b0(weights=weights)
    else:
        model = tv.models.efficientnet_b0(pretrained=False)
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model


def get_model(model_name, num_classes, pretrained=True):
    """模型工厂函数，支持多种模型选择"""
    models = {
        'resnet50': get_resnet50,
        'efficientnet_b0': get_efficientnet_b0
    }
    
    if model_name not in models:
        raise ValueError(f"不支持的模型: {model_name}，可选模型: {list(models.keys())}")
    
    return models[model_name](num_classes, pretrained)