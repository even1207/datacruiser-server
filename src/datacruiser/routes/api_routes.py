"""
Main API routes for the RAG system
"""

from flask import Blueprint, request, jsonify, g
from typing import Dict, Any
import logging

from ..models.data_models import QueryParams

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)


@api_bp.route("/ask", methods=["POST"])
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

        # Get services from Flask g object
        rag_service = getattr(g, 'rag_service', None)
        llm_service = getattr(g, 'llm_service', None)
        
        if not rag_service or not llm_service:
            return jsonify({"error": "Services not initialized", "success": False}), 500

        # Extract query parameters
        query_params_dict = data.get("query_params", {})
        top_k = min(data.get("top_k", 5), 10)  # Limit to prevent issues

        # Use context-aware search that handles date filtering
        similar_records = rag_service.search_similar_records_with_context(question, top_k)

        if not similar_records:
            # Generate a helpful response even when no records are found
            answer = llm_service.generate_response(question, similar_records)
            return jsonify({
                "question": question,
                "answer": answer,
                "success": True,
                "similar_records": [],
                "total_records": len(rag_service.records),
                "from_cache": rag_service.cache_manager.is_cache_valid(),
                "message": "No records found matching the criteria"
            })

        # Generate LLM response
        answer = llm_service.generate_response(question, similar_records)

        return jsonify({
            "question": question,
            "answer": answer,
            "success": True,
            "similar_records": [record.to_dict() for record in similar_records[:3]],
            "total_records": len(rag_service.records),
            "from_cache": rag_service.cache_manager.is_cache_valid()
        })

    except Exception as e:
        logger.error(f"❌ Error processing question: {str(e)}")
        return jsonify({"error": "Internal server error", "success": False}), 500


@api_bp.route("/ask-hybrid", methods=["POST"])
def ask_hybrid_question():
    """Hybrid endpoint supporting both statistical and trend queries"""
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' field", "success": False}), 400

        question = data["question"].strip()
        if not question:
            return jsonify({"error": "Question cannot be empty", "success": False}), 400

        logger.info(f"🔍 Received hybrid question: {question}")

        # Get hybrid service from Flask g object
        hybrid_service = getattr(g, 'hybrid_service', None)
        
        if not hybrid_service:
            return jsonify({"error": "Hybrid service not initialized", "success": False}), 500

        # Extract parameters
        top_k = min(data.get("top_k", 5), 10)  # Limit to prevent issues

        # Process the question using hybrid approach
        result = hybrid_service.process_question(question, top_k)

        # Return the result
        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ Error processing hybrid question: {str(e)}")
        return jsonify({"error": "Internal server error", "success": False}), 500


@api_bp.route("/system-info", methods=["GET"])
def get_system_info():
    """Get system information and status"""
    try:
        # Get hybrid service from Flask g object
        hybrid_service = getattr(g, 'hybrid_service', None)
        
        if not hybrid_service:
            return jsonify({"error": "Hybrid service not initialized", "success": False}), 500

        info = hybrid_service.get_system_info()
        return jsonify({
            "success": True,
            "system_info": info
        })

    except Exception as e:
        logger.error(f"❌ Error getting system info: {str(e)}")
        return jsonify({"error": "Internal server error", "success": False}), 500


@api_bp.route("/datasets/upload", methods=["POST"])
def upload_datasets():
    """Upload one or more CSV files and build a chunked dataset."""
    try:
        dataset_service = getattr(g, 'dataset_service', None)
        if not dataset_service:
            return jsonify({"error": "Dataset service not initialized", "success": False}), 500

        if 'files' not in request.files:
            return jsonify({"error": "No files part in the request", "success": False}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files provided", "success": False}), 400

        result = dataset_service.ingest_files(files)
        return jsonify({
            "success": True,
            "dataset": result
        })

    except ValueError as e:
        return jsonify({"error": str(e), "success": False}), 400
    except Exception as e:
        logger.error(f"❌ Error uploading datasets: {str(e)}")
        return jsonify({"error": "Internal server error", "success": False}), 500


@api_bp.route("/datasets/<dataset_id>/ask", methods=["POST"])
def ask_uploaded_dataset(dataset_id: str):
    """Answer questions about an uploaded dataset using structured prompts."""
    try:
        dataset_service = getattr(g, 'dataset_service', None)
        llm_service = getattr(g, 'llm_service', None)

        if not dataset_service:
            return jsonify({"error": "Dataset service not initialized", "success": False}), 500

        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Missing 'question' field", "success": False}), 400

        result = dataset_service.answer_question(dataset_id, question, llm_service)
        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": str(e), "success": False}), 404
    except ValueError as e:
        return jsonify({"error": str(e), "success": False}), 400
    except Exception as e:
        logger.error(f"❌ Error processing dataset question: {str(e)}")
        return jsonify({"error": "Internal server error", "success": False}), 500
