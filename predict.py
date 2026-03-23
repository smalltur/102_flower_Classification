import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import time

# 导入自定义模块
from utils import load_model_info, load_image_safe
from model import get_model

# 加载标签映射的函数（和训练代码保持一致）
def load_label_mapping(load_path='label_mapping.json'):
    """
    从JSON文件加载类别映射关系
    :param load_path: 标签映射文件路径
    :return: class_to_idx, idx_to_class
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"标签映射文件不存在: {load_path}")
    
    with open(load_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    # 转换str键回int（JSON中idx是字符串，需转回整数）
    idx_to_class = {int(k): v for k, v in label_mapping['idx_to_class'].items()}
    return label_mapping['class_to_idx'], idx_to_class

# 测试集数据预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class FolderTestDataset(Dataset):
    """无CSV依赖的测试集类：直接遍历文件夹下所有图片"""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # 递归遍历所有子文件夹下的图片
        self.img_paths = []
        self.img_names = []
        for root, dirs, files in os.walk(img_dir):
            for img_name in files:
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.img_paths.append(os.path.join(root, img_name))
                    # 保存相对路径
                    relative_path = os.path.relpath(os.path.join(root, img_name), img_dir)
                    self.img_names.append(relative_path)
        
        if len(self.img_paths) == 0:
            raise ValueError(f"测试文件夹 {img_dir} 中未找到任何图片文件")
        
        print(f"测试集加载完成: {len(self.img_paths)} 个样本")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_name = self.img_names[index]
        
        # 安全加载图片
        img = load_image_safe(img_path)
        if self.transform:
            img = self.transform(img)
            
        return img, img_name

def predict():
    # 配置参数
    test_img_dir = r"C:\Users\smtur\Desktop\submission\data\train"  # 你的测试图片目录
    model_dir = 'saved_models'                    # 模型保存目录
    label_mapping_path = 'label_mapping.json'     # 标签映射文件
    model_architecture = 'resnet50'               # 直接指定模型架构（关键修复）
    # model_name = "model_20251112_100000"        # 可选：指定模型名称
    model_name = None                             # 使用最新模型
    batch_size = 16

    # 检查必要文件
    check_list = [
        (test_img_dir, "测试图片目录"),
        (model_dir, "模型目录"),
        (label_mapping_path, "标签映射文件")
    ]
    for path, name in check_list:
        if not os.path.exists(path):
            print(f"错误: {name} 不存在 -> {path}")
            return

    # 加载标签映射
    try:
        class_to_idx, idx_to_class = load_label_mapping(label_mapping_path)
        num_classes = len(class_to_idx)
        print(f"✅ 成功加载标签映射，共 {num_classes} 个类别")
        print(f"类别列表: {list(class_to_idx.keys())}")
    except Exception as e:
        print(f"❌ 加载标签映射失败: {e}")
        return

    # 加载模型（仅加载权重，模型架构手动指定）
    model_info, model_path = load_model_info(model_dir, model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    print(f"📦 加载模型: {model_path}")

    # 关键修复：使用手动指定的model_architecture，不再依赖model_info
    model = get_model(model_architecture, num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 加载测试数据
    try:
        test_dataset = FolderTestDataset(test_img_dir, test_transform)
    except ValueError as e:
        print(f"❌ 加载测试集失败: {e}")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Windows系统设为0
        pin_memory=True
    )

    # 批量预测
    predictions = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="📊 预测进度")
        for inputs, filenames in pbar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            for i in range(len(inputs)):
                pred_idx = torch.argmax(outputs[i]).item()
                confidence = round(torch.max(probs[i]).item(), 4)
                pred_class = idx_to_class[pred_idx]

                predictions.append({
                    'filename': filenames[i],
                    'category_id': pred_class,
                    'confidence': confidence
                })

    # 保存预测结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f'predictions_{timestamp}.csv'
    try:
        pd.DataFrame(predictions).to_csv(
            output_file,
            index=False,
            encoding='utf-8-sig'
        )
        print(f"\n✅ 预测完成！结果保存至: {os.path.abspath(output_file)}")
        print("\n📝 预测结果示例:")
        sample = pd.DataFrame(predictions).head(5)
        print(sample.to_string(index=False))
    except Exception as e:
        print(f"❌ 保存预测结果失败: {e}")
        return

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    predict()