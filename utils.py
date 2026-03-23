import os
import chardet
import pandas as pd
import json
import time
from PIL import Image


def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"检测到文件编码: {encoding} (置信度: {confidence:.2f})")
        return encoding


def read_csv_with_encoding(csv_file_path):
    """读取CSV文件并自动处理编码问题"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_file_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取文件")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"使用 {encoding} 编码时出错: {e}")
            continue

    try:
        detected_encoding = detect_encoding(csv_file_path)
        df = pd.read_csv(csv_file_path, encoding=detected_encoding)
        print(f"使用检测到的编码 {detected_encoding} 读取文件成功")
        return df
    except Exception as e:
        print(f"自动检测编码也失败: {e}")

    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8', errors='ignore')
        print("使用错误忽略模式读取文件成功")
        return df
    except Exception as e:
        print(f"所有方法都失败: {e}")
        raise


def save_model_info(model, dataset, model_dir='saved_models'):
    """保存模型权重和类别信息"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = f"model_{timestamp}"

    # 保存模型权重
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)

    # 保存类别映射信息
    model_info = {
        'model_name': model_name,
        'num_classes': dataset.num_classes,
        'classes': dataset.classes,
        'class_to_idx': dataset.class_to_idx,
        'idx_to_class': dataset.idx_to_class,
        'timestamp': timestamp
    }

    info_path = os.path.join(model_dir, f"{model_name}_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    print(f"✅ 模型已保存到: {model_path}")
    print(f"✅ 模型信息已保存到: {info_path}")
    return model_name


def load_model_info(model_dir='saved_models', model_name=None):
    """加载模型权重和类别信息"""
    if model_name is None:
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_info.json')]
        if not model_files:
            raise FileNotFoundError("没有找到保存的模型")
        model_files.sort(reverse=True)
        model_name = model_files[0].replace('_info.json', '')

    info_path = os.path.join(model_dir, f"{model_name}_info.json")
    model_path = os.path.join(model_dir, f"{model_name}.pth")

    if not os.path.exists(info_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_name}")

    with open(info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)

    return model_info, model_path


def load_image_safe(img_path, size=(224, 224)):
    """安全加载图像，处理文件不存在或损坏的情况"""
    if not os.path.exists(img_path):
        print(f"警告: 图像文件不存在: {img_path}")
        return Image.new('RGB', size, color='black')
    
    try:
        return Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"警告: 无法加载图像 {img_path}: {e}")
        return Image.new('RGB', size, color='black')


# 为了避免循环导入，在utils中提前声明torch
import torch