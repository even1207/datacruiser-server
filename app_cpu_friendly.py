#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG-based Footfall Data Analysis API Server (CPU Friendly Version)
使用 FAISS + TimesFM Embedding + OpenAI LLM 的人流量数据查询服务
支持CPU和GPU运行，自动检测和fallback
"""

import os
import json
import numpy as np
import faiss
import torch
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from transformers import TimesFmModelForPrediction
from openai import OpenAI
import logging
from typing import List, Dict, Any, Tuple

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)

# 全局变量
model = None
client = None
faiss_index = None
records = []
embeddings = None
device_info = {}

def detect_device():
    """检测并返回设备信息"""
    global device_info

    # 检测CUDA
    cuda_available = torch.cuda.is_available()
    device_type = "cuda" if cuda_available else "cpu"

    device_info = {
        "type": device_type,
        "cuda_available": cuda_available,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "cpu_count": os.cpu_count()
    }

    if cuda_available:
        device_info["gpu_name"] = torch.cuda.get_device_name(0)
        device_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

    logger.info(f"🔍 Device detection: {device_info}")
    return device_info

def initialize_timesfm():
    """初始化 TimesFM 模型 - CPU友好版本"""
    global model

    try:
        # 检测设备
        device_info = detect_device()

        if device_info["cuda_available"]:
            logger.info("🚀 Attempting to load TimesFM model on GPU...")
            try:
                model = TimesFmModelForPrediction.from_pretrained(
                    "google/timesfm-2.0-500m-pytorch",
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                logger.info(f"✅ TimesFM model loaded successfully on GPU: {device_info.get('gpu_name', 'Unknown')}")
                return True
            except Exception as gpu_error:
                logger.warning(f"⚠️ GPU loading failed: {gpu_error}")
                logger.info("🔄 Falling back to CPU...")

        # CPU模式加载
        logger.info("🚀 Loading TimesFM model on CPU...")
        model = TimesFmModelForPrediction.from_pretrained(
            "google/timesfm-2.0-500m-pytorch",
            torch_dtype=torch.float32,  # CPU用float32更稳定
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        logger.info(f"✅ TimesFM model loaded successfully on CPU ({device_info['cpu_count']} cores)")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load TimesFM model: {str(e)}")
        logger.error("Troubleshooting tips:")
        logger.error("1. Ensure you have sufficient memory (8GB+ recommended)")
        logger.error("2. Check internet connection for model download")
        logger.error("3. Try reducing max_records in load_and_process_data()")
        return False

def timesfm_embed(series: np.ndarray) -> np.ndarray:
    """对数值序列生成 embedding - 自适应设备版本"""
    if model is None:
        raise ValueError("TimesFM model not initialized")

    device = model.device

    # 根据设备类型选择数据类型
    if device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    try:
        x = torch.tensor(series, dtype=dtype, device=device).unsqueeze(0)
        f = torch.tensor([0], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(
                past_values=[x],
                freq=f,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden = out.hidden_states[-1][0]
            emb = last_hidden.mean(dim=0).float()

        return emb.cpu().numpy()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("⚠️ GPU memory insufficient, trying CPU fallback...")
            # 如果GPU内存不足，强制使用CPU
            x = torch.tensor(series, dtype=torch.float32, device="cpu").unsqueeze(0)
            f = torch.tensor([0], dtype=torch.long, device="cpu")

            # 将模型临时移到CPU
            model_cpu = model.cpu()
            with torch.no_grad():
                out = model_cpu(
                    past_values=[x],
                    freq=f,
                    output_hidden_states=True,
                    return_dict=True
                )
                last_hidden = out.hidden_states[-1][0]
                emb = last_hidden.mean(dim=0).float()

            return emb.numpy()
        else:
            raise e

def load_and_process_data():
    """加载JSON数据并生成embeddings - 内存优化版本"""
    global records, embeddings, faiss_index

    try:
        # 加载JSON数据
        json_path = os.path.join("dataProcess", "data.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        logger.info("📂 Loading JSON data...")
        with open(json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        logger.info(f"✅ Loaded {len(records)} records from JSON")

        # 根据可用内存动态调整处理数量
        available_memory_gb = 8  # 默认假设8GB
        if device_info.get("cuda_available"):
            # GPU模式可以处理更多数据
            max_records = min(len(records), 100000)
        else:
            # CPU模式限制数据量以节省内存
            max_records = min(len(records), 30000)

        if len(records) > max_records:
            logger.info(f"⚠️ Limiting to first {max_records} records for memory efficiency")
            records = records[:max_records]

        # 生成embeddings - 批处理以节省内存
        logger.info("🔮 Generating embeddings...")
        embeddings_list = []
        batch_size = 100 if device_info.get("type") == "cpu" else 500  # CPU用更小的批次

        for i in range(0, len(records), batch_size):
            batch_end = min(i + batch_size, len(records))
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size} (records {i}-{batch_end})")

            batch_embeddings = []
            for j in range(i, batch_end):
                rec = records[j]
                try:
                    # 提取数值特征
                    series = np.array([
                        float(rec.get("TotalCount", 0)),
                        float(rec.get("LastWeek", 0)),
                        float(rec.get("Previous4DayTimeAvg", 0)),
                        float(rec.get("LastYear", 0)),
                        float(rec.get("Previous52DayTimeAvg", 0))
                    ], dtype=np.float32)

                    # 处理NaN值
                    series = np.nan_to_num(series, nan=0.0)

                    # 生成embedding
                    emb = timesfm_embed(series)
                    # 标准化
                    norm = np.linalg.norm(emb)
                    if norm > 1e-12:
                        emb = emb / norm

                    batch_embeddings.append(emb)

                except Exception as e:
                    logger.warning(f"Error processing record {j}: {str(e)}")
                    # 使用零向量作为fallback
                    emb_dim = 512  # TimesFM的embedding维度
                    batch_embeddings.append(np.zeros(emb_dim, dtype=np.float32))

            embeddings_list.extend(batch_embeddings)

            # 定期清理GPU缓存
            if device_info.get("cuda_available"):
                torch.cuda.empty_cache()

        embeddings = np.stack(embeddings_list).astype("float32")
        logger.info(f"✅ Generated embeddings: {embeddings.shape}")

        # 创建FAISS索引
        logger.info("🔍 Creating FAISS index...")
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)  # 使用内积相似度
        faiss_index.add(embeddings)

        logger.info(f"✅ FAISS index created with {faiss_index.ntotal} vectors")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load and process data: {str(e)}")
        return False

def search_similar_records(query_data: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """搜索相似的记录"""
    if faiss_index is None or model is None:
        raise ValueError("System not properly initialized")

    try:
        # 从查询数据中提取数值特征
        query_series = np.array([
            float(query_data.get("TotalCount", 0)),
            float(query_data.get("LastWeek", 0)),
            float(query_data.get("Previous4DayTimeAvg", 0)),
            float(query_data.get("LastYear", 0)),
            float(query_data.get("Previous52DayTimeAvg", 0))
        ], dtype=np.float32)

        # 处理NaN值
        query_series = np.nan_to_num(query_series, nan=0.0)

        # 生成查询embedding
        q_emb = timesfm_embed(query_series)
        norm = np.linalg.norm(q_emb)
        if norm > 1e-12:
            q_emb = q_emb / norm

        q_emb = q_emb.astype("float32").reshape(1, -1)

        # 搜索相似记录
        scores, ids = faiss_index.search(q_emb, top_k)

        # 返回相似记录
        similar_records = []
        for i, record_id in enumerate(ids[0]):
            if record_id < len(records):
                record = records[record_id].copy()
                record['similarity_score'] = float(scores[0][i])
                similar_records.append(record)

        return similar_records

    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        return []

def generate_llm_response(query: str, similar_records: List[Dict[str, Any]]) -> str:
    """使用LLM生成响应"""
    global client

    if client is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        client = OpenAI(api_key=openai_api_key)

    # 构建上下文
    context_parts = []
    for i, record in enumerate(similar_records):
        context_parts.append(
            f"{i+1}. {record['Location_Name']} ({record['Date']}): "
            f"TotalCount={record['TotalCount']}, "
            f"LastWeek={record.get('LastWeek', 'N/A')}, "
            f"LastYear={record.get('LastYear', 'N/A')}, "
            f"相似度={record.get('similarity_score', 0):.3f}"
        )

    context = "\n".join(context_parts)

    prompt = f"""
你是一个专业的人流量分析助手。用户询问: "{query}"

以下是通过向量相似度检索到的最相关的历史记录：
{context}

请基于这些检索到的历史数据来回答用户的问题。注意：
1. 重点关注TotalCount（总人数）这个主要指标
2. 可以分析时间趋势、地点对比等
3. 如果数据中有LastWeek、LastYear等历史对比数据，可以用来分析趋势
4. 用中文回答，语言要专业但易懂
5. 如果检索到的数据与问题不太相关，请诚实说明

请提供有洞察力的分析和回答。
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        return f"抱歉，生成回答时出现错误: {str(e)}"

@app.route("/", methods=["GET"])
def health_check():
    """健康检查接口"""
    status = {
        "service": "RAG-based Footfall Analysis API (CPU Friendly)",
        "status": "running",
        "device_info": device_info,
        "timesfm_loaded": model is not None,
        "faiss_index_loaded": faiss_index is not None,
        "records_count": len(records) if records else 0,
        "embedding_dimension": embeddings.shape[1] if embeddings is not None else None
    }
    return jsonify(status)

@app.route("/ask", methods=["POST"])
def ask_question():
    """RAG查询接口"""
    try:
        # 检查系统是否已初始化
        if model is None or faiss_index is None:
            return jsonify({
                "error": "System not initialized. Please check server logs.",
                "success": False
            }), 500

        # 获取请求数据
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({
                "error": "Missing 'question' field in request body",
                "success": False
            }), 400

        question = data["question"].strip()
        if not question:
            return jsonify({
                "error": "Question cannot be empty",
                "success": False
            }), 400

        logger.info(f"📝 Received question: {question}")

        # 提取查询参数（如果有的话）
        query_params = data.get("query_params", {})
        top_k = data.get("top_k", 5)

        # 如果没有提供具体的查询参数，使用默认值进行搜索
        if not query_params:
            # 可以从问题中尝试提取一些信息，或使用平均值
            query_params = {
                "TotalCount": 2000,  # 默认值
                "LastWeek": 2000,
                "Previous4DayTimeAvg": 2000,
                "LastYear": 2000,
                "Previous52DayTimeAvg": 2000
            }

        # 搜索相似记录
        similar_records = search_similar_records(query_params, top_k)

        if not similar_records:
            return jsonify({
                "question": question,
                "answer": "抱歉，没有找到相关的数据记录。",
                "success": True,
                "similar_records": []
            })

        # 使用LLM生成回答
        answer = generate_llm_response(question, similar_records)

        logger.info(f"✅ Generated response for question: {question}")

        return jsonify({
            "question": question,
            "answer": answer,
            "success": True,
            "similar_records": similar_records[:3],  # 返回前3个最相似的记录
            "total_records": len(records),
            "device_used": device_info.get("type", "unknown")
        })

    except Exception as e:
        logger.error(f"❌ Error processing question: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route("/search", methods=["POST"])
def search_similar():
    """基于数值特征的相似度搜索接口"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "Request body is required",
                "success": False
            }), 400

        top_k = data.get("top_k", 5)
        query_params = data.get("query_params", {})

        if not query_params:
            return jsonify({
                "error": "query_params is required",
                "success": False
            }), 400

        similar_records = search_similar_records(query_params, top_k)

        return jsonify({
            "similar_records": similar_records,
            "success": True,
            "query_params": query_params,
            "device_used": device_info.get("type", "unknown")
        })

    except Exception as e:
        logger.error(f"❌ Error in similarity search: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route("/data/info", methods=["GET"])
def data_info():
    """获取数据集信息的接口"""
    if not records:
        return jsonify({
            "error": "Data not loaded",
            "success": False
        }), 500

    # 计算一些统计信息
    sample_records = records[:3] if len(records) >= 3 else records

    return jsonify({
        "total_records": len(records),
        "embedding_dimension": embeddings.shape[1] if embeddings is not None else None,
        "faiss_index_size": faiss_index.ntotal if faiss_index is not None else None,
        "sample_records": sample_records,
        "available_fields": list(records[0].keys()) if records else [],
        "device_info": device_info,
        "success": True
    })

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        "error": "Endpoint not found",
        "success": False,
        "available_endpoints": ["/", "/ask", "/search", "/data/info"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({
        "error": "Internal server error",
        "success": False
    }), 500

def initialize_system():
    """初始化整个系统"""
    logger.info("🚀 Initializing RAG-based Footfall Analysis System (CPU Friendly)...")

    # 检测设备
    detect_device()

    # 初始化TimesFM模型
    if not initialize_timesfm():
        return False

    # 加载数据并生成embeddings
    if not load_and_process_data():
        return False

    logger.info("✅ System initialization completed successfully!")
    logger.info(f"📊 Final stats: {len(records)} records, device: {device_info.get('type', 'unknown')}")
    return True

if __name__ == "__main__":
    try:
        # 初始化系统
        if not initialize_system():
            raise Exception("System initialization failed")

        # 启动Flask服务
        logger.info("🚀 Starting RAG-based Footfall Analysis API Server (CPU Friendly)...")
        app.run(
            host="0.0.0.0",
            port=5080,
            debug=False,
            threaded=True
        )

    except Exception as e:
        logger.error(f"❌ Failed to start server: {str(e)}")
        print(f"\n❌ Server startup failed: {str(e)}")
        print("Please check:")
        print("1. OPENAI_API_KEY is set in .env file")
        print("2. dataProcess/data.json file exists")
        print("3. All required packages are installed (pip install -r requirements.txt)")
        print("4. Sufficient system memory (8GB+ recommended)")
        print("5. Internet connection for downloading TimesFM model")
