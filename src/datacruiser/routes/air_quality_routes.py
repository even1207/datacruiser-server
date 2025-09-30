"""
Air Quality API routes for NSW Air Quality Data
"""

from flask import Blueprint, request, jsonify, g
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

air_quality_bp = Blueprint('air_quality', __name__)

@air_quality_bp.route("/locations", methods=["GET"])
def get_air_quality_locations():
    """Get air quality monitoring locations with latest data"""
    try:
        # Get air quality service from Flask g object
        air_quality_service = getattr(g, 'air_quality_service', None)
        
        if not air_quality_service:
            return jsonify({"error": "Air quality service not initialized", "success": False}), 500
        
        # Get query parameters
        time = request.args.get('time')
        sensor_types = request.args.getlist('sensor_types') if request.args.get('sensor_types') else None
        
        # Validate time parameter if provided
        if time:
            try:
                datetime.strptime(time, "%Y-%m-%d")
            except ValueError:
                return jsonify({"error": "Invalid time format. Use YYYY-MM-DD", "success": False}), 400
        
        # Get locations
        locations = air_quality_service.get_locations(time=time, sensor_types=sensor_types)
        
        # Format response for map visualization
        formatted_locations = []
        for location in locations:
            formatted_location = {
                'id': f"{location['device_code']}_{location['sensor_code']}",
                'device_code': location['device_code'],
                'device_name': location['device_name'],
                'sensor_code': location['sensor_code'],
                'sensor_name': location['sensor_name'],
                'coordinates': [location['longitude'], location['latitude']],
                'description': location['description'],
                'latest_value': location['value'],
                'unit': location['uom'],
                'phenomenon': location['phenomenon_name'],
                'timestamp': location['timestamp'].isoformat() if location['timestamp'] else None,
                'type': 'air_quality'
            }
            formatted_locations.append(formatted_location)
        
        return jsonify({
            "success": True,
            "locations": formatted_locations,
            "count": len(formatted_locations),
            "filters": {
                "time": time,
                "sensor_types": sensor_types
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting air quality locations: {str(e)}")
        return jsonify({"error": "Internal server error", "success": False}), 500

@air_quality_bp.route("/time-series", methods=["GET"])
def get_time_series_data():
    """Get time series data for a specific device and sensor"""
    try:
        # Get air quality service from Flask g object
        air_quality_service = getattr(g, 'air_quality_service', None)
        
        if not air_quality_service:
            return jsonify({"error": "Air quality service not initialized", "success": False}), 500
        
        # Get query parameters
        device_code = request.args.get('device_code')
        sensor_code = request.args.get('sensor_code')
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        
        if not device_code or not sensor_code:
            return jsonify({"error": "device_code and sensor_code are required", "success": False}), 400
        
        # Validate time parameters if provided
        if start_time:
            try:
                datetime.strptime(start_time, "%Y-%m-%d")
            except ValueError:
                return jsonify({"error": "Invalid start_time format. Use YYYY-MM-DD", "success": False}), 400
        
        if end_time:
            try:
                datetime.strptime(end_time, "%Y-%m-%d")
            except ValueError:
                return jsonify({"error": "Invalid end_time format. Use YYYY-MM-DD", "success": False}), 400
        
        # Get time series data
        time_series = air_quality_service.get_time_series_data(
            device_code=device_code,
            sensor_code=sensor_code,
            start_time=start_time,
            end_time=end_time
        )
        
        # Format response
        formatted_data = []
        for point in time_series:
            formatted_point = {
                'timestamp': point['timestamp'].isoformat() if point['timestamp'] else None,
                'value': point['value'],
                'unit': point['uom'],
                'device_name': point['device_name'],
                'sensor_name': point['sensor_name'],
                'phenomenon': point['phenomenon_name']
            }
            formatted_data.append(formatted_point)
        
        return jsonify({
            "success": True,
            "device_code": device_code,
            "sensor_code": sensor_code,
            "time_series": formatted_data,
            "count": len(formatted_data),
            "filters": {
                "start_time": start_time,
                "end_time": end_time
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting time series data: {str(e)}")
        return jsonify({"error": "Internal server error", "success": False}), 500

@air_quality_bp.route("/sensor-types", methods=["GET"])
def get_sensor_types():
    """Get all available sensor types"""
    try:
        # Get air quality service from Flask g object
        air_quality_service = getattr(g, 'air_quality_service', None)
        
        if not air_quality_service:
            return jsonify({"error": "Air quality service not initialized", "success": False}), 500
        
        # Get sensor types
        sensor_types = air_quality_service.get_available_sensor_types()
        
        return jsonify({
            "success": True,
            "sensor_types": sensor_types,
            "count": len(sensor_types)
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting sensor types: {str(e)}")
        return jsonify({"error": "Internal server error", "success": False}), 500

@air_quality_bp.route("/summary", methods=["GET"])
def get_air_quality_summary():
    """Get summary information about air quality data"""
    try:
        # Get air quality service from Flask g object
        air_quality_service = getattr(g, 'air_quality_service', None)
        
        if not air_quality_service:
            return jsonify({"error": "Air quality service not initialized", "success": False}), 500
        
        # Get summary
        summary = air_quality_service.get_device_summary()
        
        return jsonify({
            "success": True,
            "summary": summary
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting air quality summary: {str(e)}")
        return jsonify({"error": "Internal server error", "success": False}), 500

@air_quality_bp.route("/health", methods=["GET"])
def air_quality_health():
    """Health check for air quality service"""
    try:
        # Get air quality service from Flask g object
        air_quality_service = getattr(g, 'air_quality_service', None)
        
        if not air_quality_service:
            return jsonify({
                "status": "unhealthy",
                "message": "Air quality service not initialized"
            }), 500
        
        # Try a simple query to test the service
        summary = air_quality_service.get_device_summary()
        
        if summary and summary.get('device_count', 0) > 0:
            return jsonify({
                "status": "healthy",
                "message": "Air quality service is operational",
                "data_summary": {
                    "total_records": summary.get('total_records', 0),
                    "device_count": summary.get('device_count', 0),
                    "sensor_count": summary.get('sensor_count', 0)
                }
            })
        else:
            return jsonify({
                "status": "unhealthy",
                "message": "No air quality data available"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Air quality health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}"
        }), 500
