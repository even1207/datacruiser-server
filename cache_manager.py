#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache Manager for RAG Footfall Analysis API
缓存管理工具
"""

import os
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

CACHE_DIR = "cache"
CACHE_FILES = {
    "data": "processed_data.pkl",
    "embeddings": "embeddings.npy",
    "faiss_index": "faiss_index.pkl",
    "stats": "data_stats.pkl",
    "model": "timesfm_model.pkl",
    "metadata": "cache_metadata.json"
}

def get_cache_info():
    """获取缓存信息"""
    if not os.path.exists(CACHE_DIR):
        return {"exists": False, "files": {}, "total_size": 0}

    info = {"exists": True, "files": {}, "total_size": 0}

    for name, filename in CACHE_FILES.items():
        filepath = os.path.join(CACHE_DIR, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            modified = datetime.fromtimestamp(os.path.getmtime(filepath))
            info["files"][name] = {
                "exists": True,
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2),
                "modified": modified.strftime("%Y-%m-%d %H:%M:%S")
            }
            info["total_size"] += size
        else:
            info["files"][name] = {"exists": False}

    info["total_size_mb"] = round(info["total_size"] / (1024 * 1024), 2)
    return info

def clear_cache():
    """清理缓存"""
    if not os.path.exists(CACHE_DIR):
        print("📁 Cache directory does not exist.")
        return

    removed_files = []
    for name, filename in CACHE_FILES.items():
        filepath = os.path.join(CACHE_DIR, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                removed_files.append(filename)
                print(f"🗑️  Removed: {filename}")
            except Exception as e:
                print(f"❌ Failed to remove {filename}: {e}")

    if removed_files:
        print(f"✅ Cleared {len(removed_files)} cache files")
    else:
        print("📭 No cache files to remove")

def show_status():
    """显示缓存状态"""
    print("📊 Cache Status")
    print("=" * 50)

    info = get_cache_info()

    if not info["exists"]:
        print("📁 Cache directory does not exist")
        return

    print(f"📂 Cache Directory: {os.path.abspath(CACHE_DIR)}")
    print(f"💾 Total Size: {info['total_size_mb']} MB")
    print()

    print("📋 Cache Files:")
    for name, file_info in info["files"].items():
        if file_info["exists"]:
            print(f"  ✅ {name:12s}: {file_info['size_mb']:8.2f} MB  (modified: {file_info['modified']})")
        else:
            print(f"  ❌ {name:12s}: Not cached")

    # Check if cache is likely valid
    essential_files = ["data", "embeddings", "faiss_index"]
    valid_cache = all(info["files"].get(f, {}).get("exists", False) for f in essential_files)

    print()
    if valid_cache:
        print("✅ Cache appears to be complete and valid")
    else:
        print("⚠️  Cache appears to be incomplete or invalid")

def validate_cache():
    """验证缓存完整性"""
    print("🔍 Validating Cache")
    print("=" * 50)

    info = get_cache_info()

    if not info["exists"]:
        print("❌ Cache directory does not exist")
        return False

    # Check essential files
    essential_files = ["data", "embeddings", "faiss_index", "metadata"]
    missing_files = []

    for file_type in essential_files:
        if not info["files"].get(file_type, {}).get("exists", False):
            missing_files.append(file_type)

    if missing_files:
        print(f"❌ Missing essential cache files: {', '.join(missing_files)}")
        return False

    # Check metadata
    metadata_file = os.path.join(CACHE_DIR, "cache_metadata.json")
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"📅 Cache created: {metadata.get('created_at', 'Unknown')}")
        print(f"🔖 Cache version: {metadata.get('cache_version', 'Unknown')}")

        # Check if source data file exists and matches
        data_file = os.path.join("dataProcess", "data.json")
        if os.path.exists(data_file):
            import hashlib
            hash_md5 = hashlib.md5()
            with open(data_file, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            current_hash = hash_md5.hexdigest()
            cached_hash = metadata.get("data_file_hash", "")

            if current_hash == cached_hash:
                print("✅ Source data file matches cache")
            else:
                print("⚠️  Source data file has changed since cache creation")
                return False
        else:
            print("❌ Source data file not found")
            return False

    except Exception as e:
        print(f"❌ Error reading cache metadata: {e}")
        return False

    print("✅ Cache validation passed")
    return True

def main():
    parser = argparse.ArgumentParser(description="Cache Manager for RAG Footfall Analysis API")
    parser.add_argument("action", choices=["status", "clear", "validate"],
                       help="Action to perform")

    args = parser.parse_args()

    print("🔧 RAG Footfall Analysis - Cache Manager")
    print("=" * 50)

    if args.action == "status":
        show_status()
    elif args.action == "clear":
        print("🗑️  Clearing cache...")
        clear_cache()
    elif args.action == "validate":
        validate_cache()

if __name__ == "__main__":
    main()
