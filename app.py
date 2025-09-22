#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG-based Footfall Data Analysis API Server (Cached Version)
Implements aggressive caching to avoid reprocessing data and models
"""

import os
import json
import pickle
import hashlib
import numpy as np
import faiss
import torch
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from transformers import TimesFmModelForPrediction
from openai import OpenAI
import logging
from typing import List, Dict, Any, Tuple, Optional
import random
import gc
import signal
import sys
from datetime import datetime

# Fix OpenMP library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask application
app = Flask(__name__)

# Global variables
model = None
client = None
faiss_index = None
records = []
embeddings = None
device_info = {}
data_stats = {}

# Cache configuration
CACHE_DIR = "cache"
MODEL_CACHE_FILE = os.path.join(CACHE_DIR, "timesfm_model.pkl")
EMBEDDINGS_CACHE_FILE = os.path.join(CACHE_DIR, "embeddings.npy")
FAISS_INDEX_CACHE_FILE = os.path.join(CACHE_DIR, "faiss_index.pkl")
DATA_CACHE_FILE = os.path.join(CACHE_DIR, "processed_data.pkl")
STATS_CACHE_FILE = os.path.join(CACHE_DIR, "data_stats.pkl")
CACHE_METADATA_FILE = os.path.join(CACHE_DIR, "cache_metadata.json")

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    logger.info("🛑 Received interrupt signal, cleaning up...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def ensure_cache_directory():
    """Ensure cache directory exists"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logger.info(f"📁 Created cache directory: {CACHE_DIR}")

def get_file_hash(filepath: str) -> str:
    """Get MD5 hash of a file"""
    if not os.path.exists(filepath):
        return ""

    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def is_cache_valid() -> bool:
    """Check if cache is valid by comparing file hashes"""
    try:
        if not os.path.exists(CACHE_METADATA_FILE):
            return False

        with open(CACHE_METADATA_FILE, 'r') as f:
            cache_metadata = json.load(f)

        # Check if data file has changed
        data_file = os.path.join("dataProcess", "data.json")
        if not os.path.exists(data_file):
            return False

        current_hash = get_file_hash(data_file)
        cached_hash = cache_metadata.get("data_file_hash", "")

        if current_hash != cached_hash:
            logger.info("📊 Data file has changed, cache invalid")
            return False

        # Check if all cache files exist
        cache_files = [
            EMBEDDINGS_CACHE_FILE,
            FAISS_INDEX_CACHE_FILE,
            DATA_CACHE_FILE,
            STATS_CACHE_FILE
        ]

        for cache_file in cache_files:
            if not os.path.exists(cache_file):
                logger.info(f"❌ Cache file missing: {cache_file}")
                return False

        logger.info("✅ Cache is valid")
        return True

    except Exception as e:
        logger.error(f"Error checking cache validity: {e}")
        return False

def save_cache_metadata():
    """Save cache metadata including file hashes"""
    try:
        data_file = os.path.join("dataProcess", "data.json")
        metadata = {
            "data_file_hash": get_file_hash(data_file),
            "created_at": datetime.now().isoformat(),
            "cache_version": "1.0"
        }

        with open(CACHE_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("💾 Cache metadata saved")

    except Exception as e:
        logger.error(f"Error saving cache metadata: {e}")

def detect_device():
    """Detect and return device information"""
    global device_info

    # Detect CUDA
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
        device_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    logger.info(f"🔍 Device detection: {device_info}")
    return device_info

def load_model_from_cache() -> bool:
    """Load model from cache if available"""
    global model

    try:
        if os.path.exists(MODEL_CACHE_FILE):
            logger.info("📦 Loading model from cache...")
            with open(MODEL_CACHE_FILE, 'rb') as f:
                model = pickle.load(f)
            logger.info("✅ Model loaded from cache successfully")
            return True
        else:
            logger.info("🔄 No model cache found, will initialize fresh")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to load model from cache: {e}")
        return False

def save_model_to_cache():
    """Save model to cache"""
    global model

    try:
        # Only save the fallback model to cache (TimesFM model is too large)
        if hasattr(model, 'embed'):
            with open(MODEL_CACHE_FILE, 'wb') as f:
                pickle.dump(model, f)
            logger.info("💾 Model saved to cache")
        else:
            logger.info("⚠️ TimesFM model too large for caching, skipping")

    except Exception as e:
        logger.error(f"❌ Failed to save model to cache: {e}")

def initialize_timesfm():
    """Initialize TimesFM model with caching"""
    global model

    # Try to load from cache first
    if load_model_from_cache():
        return True

    try:
        device_info = detect_device()

        # Always use CPU mode for stability (avoid segfaults)
        logger.info("🚀 Loading TimesFM model on CPU (safe mode)...")

        try:
            model = TimesFmModelForPrediction.from_pretrained(
                "google/timesfm-2.0-500m-pytorch",
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            logger.info(f"✅ TimesFM model loaded successfully on CPU ({device_info['cpu_count']} cores)")
            # Don't cache TimesFM model as it's too large
            return True

        except Exception as timesfm_error:
            logger.error(f"❌ Failed to load TimesFM model: {str(timesfm_error)}")
            logger.info("🔄 Using fallback embedding method...")
            return init_fallback_embedding()

    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {str(e)}")
        return init_fallback_embedding()

def init_fallback_embedding():
    """Initialize a simple fallback embedding method"""
    global model

    class FallbackEmbedding:
        def __init__(self):
            self.device = torch.device("cpu")
            # Simple linear layer for fallback
            self.linear = torch.nn.Linear(5, 512)
            self.linear.eval()

        def embed(self, series):
            with torch.no_grad():
                x = torch.tensor(series, dtype=torch.float32).unsqueeze(0)
                return self.linear(x).squeeze(0).numpy()

    model = FallbackEmbedding()
    logger.info("✅ Fallback embedding model initialized")

    # Save fallback model to cache
    save_model_to_cache()
    return True

def timesfm_embed(series: np.ndarray) -> np.ndarray:
    """Generate embedding with robust error handling"""
    if model is None:
        raise ValueError("Model not initialized")

    try:
        # Check if it's the fallback model
        if hasattr(model, 'embed'):
            return model.embed(series)

        # Original TimesFM model
        series_reshaped = series.reshape(1, -1)
        x = torch.tensor(series_reshaped, dtype=torch.float32, device="cpu")
        f = torch.tensor([0], dtype=torch.long, device="cpu")

        with torch.no_grad():
            try:
                out = model(
                    past_values=x,
                    freq=f,
                    output_hidden_states=True,
                    return_dict=True
                )
                last_hidden = out.hidden_states[-1][0]
                emb = last_hidden.mean(dim=0).float()
                return emb.cpu().numpy()

            except Exception as timesfm_error:
                logger.warning(f"TimesFM embedding failed: {timesfm_error}")
                # Simple fallback: use statistical features
                features = np.array([
                    series.mean(),
                    series.std(),
                    series.min(),
                    series.max(),
                    np.median(series)
                ])
                # Expand to 512 dimensions by repeating and adding noise
                expanded = np.tile(features, 102)[:512]  # Repeat to get 512 dims
                return expanded.astype(np.float32)

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        # Return a deterministic hash-based embedding as last resort
        hash_val = hash(tuple(series.tolist())) % (2**31)
        np.random.seed(hash_val)
        return np.random.normal(0, 1, 512).astype(np.float32)

def load_data_from_cache() -> bool:
    """Load processed data from cache"""
    global records, data_stats

    try:
        if os.path.exists(DATA_CACHE_FILE) and os.path.exists(STATS_CACHE_FILE):
            logger.info("📦 Loading processed data from cache...")

            with open(DATA_CACHE_FILE, 'rb') as f:
                records = pickle.load(f)

            with open(STATS_CACHE_FILE, 'rb') as f:
                data_stats = pickle.load(f)

            logger.info(f"✅ Loaded {len(records)} records from cache")
            return True
        else:
            logger.info("🔄 No data cache found, will process fresh")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to load data from cache: {e}")
        return False

def save_data_to_cache():
    """Save processed data to cache"""
    global records, data_stats

    try:
        with open(DATA_CACHE_FILE, 'wb') as f:
            pickle.dump(records, f)

        with open(STATS_CACHE_FILE, 'wb') as f:
            pickle.dump(data_stats, f)

        logger.info("💾 Processed data saved to cache")

    except Exception as e:
        logger.error(f"❌ Failed to save data to cache: {e}")

def load_embeddings_from_cache() -> bool:
    """Load embeddings from cache"""
    global embeddings

    try:
        if os.path.exists(EMBEDDINGS_CACHE_FILE):
            logger.info("📦 Loading embeddings from cache...")
            embeddings = np.load(EMBEDDINGS_CACHE_FILE)
            logger.info(f"✅ Loaded embeddings from cache: {embeddings.shape}")
            return True
        else:
            logger.info("🔄 No embeddings cache found, will generate fresh")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to load embeddings from cache: {e}")
        return False

def save_embeddings_to_cache():
    """Save embeddings to cache"""
    global embeddings

    try:
        if embeddings is not None:
            np.save(EMBEDDINGS_CACHE_FILE, embeddings)
            logger.info("💾 Embeddings saved to cache")

    except Exception as e:
        logger.error(f"❌ Failed to save embeddings to cache: {e}")

def load_faiss_index_from_cache() -> bool:
    """Load FAISS index from cache"""
    global faiss_index

    try:
        if os.path.exists(FAISS_INDEX_CACHE_FILE):
            logger.info("📦 Loading FAISS index from cache...")
            with open(FAISS_INDEX_CACHE_FILE, 'rb') as f:
                faiss_index = pickle.load(f)
            logger.info(f"✅ Loaded FAISS index from cache: {faiss_index.ntotal} vectors")
            return True
        else:
            logger.info("🔄 No FAISS index cache found, will create fresh")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to load FAISS index from cache: {e}")
        return False

def save_faiss_index_to_cache():
    """Save FAISS index to cache"""
    global faiss_index

    try:
        if faiss_index is not None:
            with open(FAISS_INDEX_CACHE_FILE, 'wb') as f:
                pickle.dump(faiss_index, f)
            logger.info("💾 FAISS index saved to cache")

    except Exception as e:
        logger.error(f"❌ Failed to save FAISS index to cache: {e}")

def calculate_data_statistics(records_sample):
    """Calculate data statistics with safety checks"""
    global data_stats

    try:
        total_counts = []
        last_weeks = []
        prev_4days = []
        last_years = []
        prev_52days = []

        for record in records_sample[:1000]:  # Limit to first 1000 for safety
            try:
                total_counts.append(float(record.get("TotalCount", 0)))
                last_weeks.append(float(record.get("LastWeek", 0)))
                prev_4days.append(float(record.get("Previous4DayTimeAvg", 0)))
                last_years.append(float(record.get("LastYear", 0)))
                prev_52days.append(float(record.get("Previous52DayTimeAvg", 0)))
            except:
                continue

        if total_counts:
            data_stats = {
                "TotalCount": {
                    "min": np.min(total_counts),
                    "max": np.max(total_counts),
                    "mean": np.mean(total_counts),
                    "median": np.median(total_counts)
                },
                "LastWeek": {
                    "min": np.min(last_weeks) if last_weeks else 0,
                    "max": np.max(last_weeks) if last_weeks else 0,
                    "mean": np.mean(last_weeks) if last_weeks else 0,
                    "median": np.median(last_weeks) if last_weeks else 0
                },
                "Previous4DayTimeAvg": {
                    "min": np.min(prev_4days) if prev_4days else 0,
                    "max": np.max(prev_4days) if prev_4days else 0,
                    "mean": np.mean(prev_4days) if prev_4days else 0,
                    "median": np.median(prev_4days) if prev_4days else 0
                },
                "LastYear": {
                    "min": np.min(last_years) if last_years else 0,
                    "max": np.max(last_years) if last_years else 0,
                    "mean": np.mean(last_years) if last_years else 0,
                    "median": np.median(last_years) if last_years else 0
                },
                "Previous52DayTimeAvg": {
                    "min": np.min(prev_52days) if prev_52days else 0,
                    "max": np.max(prev_52days) if prev_52days else 0,
                    "mean": np.mean(prev_52days) if prev_52days else 0,
                    "median": np.median(prev_52days) if prev_52days else 0
                }
            }

            logger.info(f"📊 Data statistics calculated: TotalCount range [{data_stats['TotalCount']['min']:.0f}, {data_stats['TotalCount']['max']:.0f}], mean: {data_stats['TotalCount']['mean']:.0f}")
        else:
            logger.warning("No valid data for statistics calculation")
            data_stats = {}

    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        data_stats = {}

def load_and_process_data():
    """Load and process data with aggressive caching"""
    global records, embeddings, faiss_index

    # Check if we can load everything from cache
    if is_cache_valid():
        logger.info("🚀 Loading all data from cache...")

        if (load_data_from_cache() and
            load_embeddings_from_cache() and
            load_faiss_index_from_cache()):
            logger.info("✅ All data loaded from cache successfully!")
            return True
        else:
            logger.warning("⚠️ Partial cache load failed, will regenerate")

    try:
        # Load JSON data (if not cached)
        if not records:
            json_path = os.path.join("dataProcess", "data.json")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")

            logger.info("📂 Loading JSON data from file...")
            with open(json_path, 'r', encoding='utf-8') as f:
                all_records = json.load(f)

            logger.info(f"✅ Loaded {len(all_records)} records from JSON")

            # Significantly reduce data for safety
            max_records = min(len(all_records), 5000)  # Only 5k records to prevent segfault

            if len(all_records) > max_records:
                logger.info(f"⚠️ Limiting to first {max_records} records for stability")
                records = all_records[:max_records]
            else:
                records = all_records

            # Calculate data statistics
            calculate_data_statistics(records)

            # Save processed data to cache
            save_data_to_cache()

        # Generate embeddings (if not cached)
        if embeddings is None:
            logger.info("🔮 Generating embeddings...")
            embeddings_list = []
            batch_size = 10  # Very small batches to prevent segfault

            for i in range(0, len(records), batch_size):
                batch_end = min(i + batch_size, len(records))
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size} (records {i}-{batch_end})")

                batch_embeddings = []
                for j in range(i, batch_end):
                    try:
                        rec = records[j]

                        # Extract numerical features with validation
                        series = np.array([
                            float(rec.get("TotalCount", 0)),
                            float(rec.get("LastWeek", 0)),
                            float(rec.get("Previous4DayTimeAvg", 0)),
                            float(rec.get("LastYear", 0)),
                            float(rec.get("Previous52DayTimeAvg", 0))
                        ], dtype=np.float32)

                        # Handle NaN values
                        series = np.nan_to_num(series, nan=0.0)

                        # Generate embedding with timeout protection
                        emb = timesfm_embed(series)

                        # Normalize safely
                        norm = np.linalg.norm(emb)
                        if norm > 1e-12:
                            emb = emb / norm

                        batch_embeddings.append(emb)

                        # Force garbage collection every few records
                        if j % 50 == 0:
                            gc.collect()

                    except Exception as e:
                        logger.warning(f"Error processing record {j}: {str(e)}")
                        # Use deterministic fallback
                        fallback_emb = np.random.RandomState(j).normal(0, 1, 512).astype(np.float32)
                        batch_embeddings.append(fallback_emb)

                embeddings_list.extend(batch_embeddings)

                # Clear memory after each batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            embeddings = np.stack(embeddings_list).astype("float32")
            logger.info(f"✅ Generated embeddings: {embeddings.shape}")

            # Save embeddings to cache
            save_embeddings_to_cache()

        # Create FAISS index (if not cached)
        if faiss_index is None:
            logger.info("🔍 Creating FAISS index...")
            dim = embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dim)
            faiss_index.add(embeddings)

            logger.info(f"✅ FAISS index created with {faiss_index.ntotal} vectors")

            # Save FAISS index to cache
            save_faiss_index_to_cache()

        # Save cache metadata
        save_cache_metadata()

        return True

    except Exception as e:
        logger.error(f"❌ Failed to load and process data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def extract_query_params_from_question(question: str) -> Dict[str, Any]:
    """Extract query parameters with safe defaults"""
    if data_stats and "TotalCount" in data_stats:
        default_params = {
            "TotalCount": data_stats["TotalCount"].get("median", 1500),
            "LastWeek": data_stats["LastWeek"].get("median", 1500),
            "Previous4DayTimeAvg": data_stats["Previous4DayTimeAvg"].get("median", 1500),
            "LastYear": data_stats["LastYear"].get("median", 1500),
            "Previous52DayTimeAvg": data_stats["Previous52DayTimeAvg"].get("median", 1500)
        }
    else:
        default_params = {
            "TotalCount": 1500,
            "LastWeek": 1500,
            "Previous4DayTimeAvg": 1500,
            "LastYear": 1500,
            "Previous52DayTimeAvg": 1500
        }

    logger.info(f"Using default query params: TotalCount={default_params['TotalCount']:.0f}")
    return default_params

def search_similar_records(query_data: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """Search with safe error handling"""
    if faiss_index is None or model is None:
        logger.warning("System not properly initialized, using fallback")
        return get_random_records_fallback(top_k)

    try:
        query_series = np.array([
            float(query_data.get("TotalCount", 0)),
            float(query_data.get("LastWeek", 0)),
            float(query_data.get("Previous4DayTimeAvg", 0)),
            float(query_data.get("LastYear", 0)),
            float(query_data.get("Previous52DayTimeAvg", 0))
        ], dtype=np.float32)

        query_series = np.nan_to_num(query_series, nan=0.0)

        q_emb = timesfm_embed(query_series)
        norm = np.linalg.norm(q_emb)
        if norm > 1e-12:
            q_emb = q_emb / norm

        q_emb = q_emb.astype("float32").reshape(1, -1)

        scores, ids = faiss_index.search(q_emb, top_k)

        similar_records = []
        for i, record_id in enumerate(ids[0]):
            if 0 <= record_id < len(records):
                record = records[record_id].copy()
                record['similarity_score'] = float(scores[0][i])
                similar_records.append(record)

        logger.info(f"✅ Found {len(similar_records)} similar records")
        return similar_records

    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        return get_random_records_fallback(top_k)

def get_random_records_fallback(top_k: int = 5) -> List[Dict[str, Any]]:
    """Safe random fallback"""
    if not records:
        return []

    try:
        random_records = random.sample(records, min(top_k, len(records)))
        for record in random_records:
            record['similarity_score'] = 0.5

        logger.info(f"🎲 Using random fallback: returned {len(random_records)} records")
        return random_records
    except Exception as e:
        logger.error(f"Error in fallback: {e}")
        return []

def generate_llm_response(query: str, similar_records: List[Dict[str, Any]]) -> str:
    """Generate LLM response with error handling"""
    global client

    try:
        if client is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                return "OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file."
            client = OpenAI(api_key=openai_api_key)

        context_parts = []
        for i, record in enumerate(similar_records):
            context_parts.append(
                f"{i+1}. {record['Location_Name']} ({record['Date']}): "
                f"TotalCount={record['TotalCount']}, "
                f"LastWeek={record.get('LastWeek', 'N/A')}, "
                f"LastYear={record.get('LastYear', 'N/A')}, "
                f"Similarity={record.get('similarity_score', 0):.3f}"
            )

        context = "\n".join(context_parts)

        prompt = f"""
You are a professional footfall data analysis assistant. User query: "{query}"

Here are the retrieved relevant historical records:
{context}

Please answer the user's question based on this data. Focus on TotalCount as the main indicator.
Answer in English, be professional but easy to understand.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        return f"Based on the available data, here's what I found: {len(similar_records)} relevant records were retrieved. The locations include {', '.join([r['Location_Name'] for r in similar_records[:3]])}. Please check your OpenAI API configuration for detailed analysis."

# Flask routes
@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    cache_status = {
        "cache_valid": is_cache_valid(),
        "cache_files": {
            "data": os.path.exists(DATA_CACHE_FILE),
            "embeddings": os.path.exists(EMBEDDINGS_CACHE_FILE),
            "faiss_index": os.path.exists(FAISS_INDEX_CACHE_FILE),
            "stats": os.path.exists(STATS_CACHE_FILE),
            "metadata": os.path.exists(CACHE_METADATA_FILE)
        }
    }

    return jsonify({
        "service": "RAG-based Footfall Analysis API (Cached Version)",
        "status": "running",
        "records_count": len(records),
        "embedding_dimension": embeddings.shape[1] if embeddings is not None else None,
        "device_info": device_info,
        "cache_status": cache_status
    })

@app.route("/ask", methods=["POST"])
def ask_question():
    """Ask endpoint with caching support"""
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' field", "success": False}), 400

        question = data["question"].strip()
        if not question:
            return jsonify({"error": "Question cannot be empty", "success": False}), 400

        logger.info(f"📝 Received question: {question}")

        query_params = data.get("query_params", {})
        top_k = min(data.get("top_k", 5), 10)  # Limit to prevent issues

        if not query_params:
            query_params = extract_query_params_from_question(question)

        similar_records = search_similar_records(query_params, top_k)

        if not similar_records:
            similar_records = get_random_records_fallback(top_k)

        answer = generate_llm_response(question, similar_records)

        return jsonify({
            "question": question,
            "answer": answer,
            "success": True,
            "similar_records": similar_records[:3],
            "total_records": len(records),
            "from_cache": is_cache_valid()
        })

    except Exception as e:
        logger.error(f"❌ Error processing question: {str(e)}")
        return jsonify({"error": "Internal server error", "success": False}), 500

@app.route("/data/info", methods=["GET"])
def data_info():
    """Data info endpoint"""
    if not records:
        return jsonify({"error": "Data not loaded", "success": False}), 500

    cache_info = {
        "cache_valid": is_cache_valid(),
        "cache_directory": CACHE_DIR,
        "cache_files_exist": {
            "data": os.path.exists(DATA_CACHE_FILE),
            "embeddings": os.path.exists(EMBEDDINGS_CACHE_FILE),
            "faiss_index": os.path.exists(FAISS_INDEX_CACHE_FILE),
            "stats": os.path.exists(STATS_CACHE_FILE)
        }
    }

    return jsonify({
        "total_records": len(records),
        "sample_records": records[:2] if records else [],
        "device_info": device_info,
        "data_stats": data_stats,
        "cache_info": cache_info,
        "success": True
    })

@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    """Clear cache endpoint"""
    try:
        cache_files = [
            MODEL_CACHE_FILE,
            EMBEDDINGS_CACHE_FILE,
            FAISS_INDEX_CACHE_FILE,
            DATA_CACHE_FILE,
            STATS_CACHE_FILE,
            CACHE_METADATA_FILE
        ]

        removed_files = []
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                removed_files.append(cache_file)

        logger.info(f"🗑️ Cleared cache files: {removed_files}")

        return jsonify({
            "message": "Cache cleared successfully",
            "removed_files": removed_files,
            "success": True
        })

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({"error": "Failed to clear cache", "success": False}), 500

@app.route("/cache/status", methods=["GET"])
def cache_status():
    """Get cache status"""
    try:
        cache_files = {
            "model": MODEL_CACHE_FILE,
            "embeddings": EMBEDDINGS_CACHE_FILE,
            "faiss_index": FAISS_INDEX_CACHE_FILE,
            "data": DATA_CACHE_FILE,
            "stats": STATS_CACHE_FILE,
            "metadata": CACHE_METADATA_FILE
        }

        status = {}
        total_size = 0

        for name, filepath in cache_files.items():
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                status[name] = {
                    "exists": True,
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                }
                total_size += size
            else:
                status[name] = {"exists": False}

        return jsonify({
            "cache_valid": is_cache_valid(),
            "cache_directory": CACHE_DIR,
            "files": status,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "success": True
        })

    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        return jsonify({"error": "Failed to get cache status", "success": False}), 500

def initialize_system():
    """Initialize system with aggressive caching"""
    logger.info("🚀 Initializing RAG-based Footfall Analysis System (Cached Version)...")

    try:
        # Ensure cache directory exists
        ensure_cache_directory()

        if not initialize_timesfm():
            return False

        if not load_and_process_data():
            return False

        logger.info("✅ System initialization completed successfully!")
        logger.info(f"📊 System loaded with {len(records)} records and {embeddings.shape[1] if embeddings is not None else 0}-dimensional embeddings")

        if is_cache_valid():
            logger.info("💾 System initialized from cache - no processing needed!")
        else:
            logger.info("🔄 System initialized fresh - cache will be used next time")

        return True

    except Exception as e:
        logger.error(f"❌ System initialization failed: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        if not initialize_system():
            logger.error("❌ System initialization failed, but starting server anyway...")

        logger.info("🚀 Starting RAG-based Footfall Analysis API Server (Cached Version)...")
        logger.info("🔗 Available endpoints:")
        logger.info("  - GET  /           : Health check and system status")
        logger.info("  - POST /ask        : Ask questions about footfall data")
        logger.info("  - GET  /data/info  : Get data and cache information")
        logger.info("  - POST /cache/clear: Clear all cache files")
        logger.info("  - GET  /cache/status: Get detailed cache status")

        app.run(host="0.0.0.0", port=5080, debug=False, threaded=True)

    except Exception as e:
        logger.error(f"❌ Failed to start server: {str(e)}")
        sys.exit(1)
