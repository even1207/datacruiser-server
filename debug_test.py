#!/usr/bin/env python3
"""
调试脚本 - 检查数据加载和搜索功能
"""

import json
import numpy as np
import torch
import os
from dotenv import load_dotenv

load_dotenv()

def test_data_loading():
    """测试数据加载"""
    print("🔍 Testing data loading...")

    json_path = os.path.join("dataProcess", "data.json")
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return False

    print("📂 Loading JSON data...")
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    print(f"✅ Loaded {len(records)} records")

    # 检查前几条记录
    for i, record in enumerate(records[:3]):
        print(f"Record {i+1}: {record}")

        # 检查数值字段
        try:
            series = np.array([
                float(record.get("TotalCount", 0)),
                float(record.get("LastWeek", 0)),
                float(record.get("Previous4DayTimeAvg", 0)),
                float(record.get("LastYear", 0)),
                float(record.get("Previous52DayTimeAvg", 0))
            ], dtype=np.float32)
            print(f"  数值特征: {series}")
        except Exception as e:
            print(f"  ❌ 数值转换错误: {e}")

    return True, records

def test_device_detection():
    """测试设备检测"""
    print("\n🔍 Testing device detection...")

    cuda_available = torch.cuda.is_available()
    device_type = "cuda" if cuda_available else "cpu"

    print(f"CUDA available: {cuda_available}")
    print(f"Device type: {device_type}")

    if cuda_available:
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    print(f"CPU count: {os.cpu_count()}")

    return device_type

def test_simple_search():
    """测试简单搜索逻辑"""
    print("\n🔍 Testing simple search logic...")

    # 模拟一些数据
    mock_records = [
        {
            "Location_Name": "Park Street",
            "TotalCount": "3234",
            "LastWeek": "4241",
            "Previous4DayTimeAvg": "3906",
            "LastYear": "3242",
            "Previous52DayTimeAvg": "3735"
        },
        {
            "Location_Name": "Market Street",
            "TotalCount": "2454",
            "LastWeek": "2788",
            "Previous4DayTimeAvg": "2676",
            "LastYear": "2679",
            "Previous52DayTimeAvg": "2622"
        }
    ]

    # 测试数值提取
    for i, record in enumerate(mock_records):
        print(f"Record {i+1}: {record['Location_Name']}")
        try:
            series = np.array([
                float(record.get("TotalCount", 0)),
                float(record.get("LastWeek", 0)),
                float(record.get("Previous4DayTimeAvg", 0)),
                float(record.get("LastYear", 0)),
                float(record.get("Previous52DayTimeAvg", 0))
            ], dtype=np.float32)
            print(f"  数值特征: {series}")
            print(f"  特征和: {np.sum(series)}")
        except Exception as e:
            print(f"  ❌ 错误: {e}")

def main():
    print("🚀 Starting debug tests...")

    # 测试数据加载
    success, records = test_data_loading()
    if not success:
        return

    # 测试设备检测
    device_type = test_device_detection()

    # 测试简单搜索
    test_simple_search()

    # 检查OPENAI_API_KEY
    print("\n🔍 Testing OpenAI API key...")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OpenAI API key found (length: {len(api_key)})")
    else:
        print("❌ OpenAI API key not found")

    print("\n✅ Debug tests completed!")

if __name__ == "__main__":
    main()
