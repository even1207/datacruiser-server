"""
Air Quality Service for NSW Air Quality Data Processing
"""

import os
import json
import duckdb
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import glob

logger = logging.getLogger(__name__)

class AirQualityService:
    """Service for processing and querying NSW Air Quality data"""
    
    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.connection = None
        self.devices = {}
        self.sensors = {}
        self.csv_files_cache = {}
        self._initialize_connection()
        self._load_metadata()
        self._build_csv_files_cache()
        
    def _initialize_connection(self):
        """Initialize DuckDB connection"""
        try:
            self.connection = duckdb.connect()
            logger.info("✅ DuckDB connection initialized for air quality data")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DuckDB: {e}")
            raise
    
    def _load_metadata(self):
        """Load device and sensor metadata from JSON files"""
        try:
            # Load devices
            devices_path = os.path.join(self.data_directory, 'devices.json')
            if os.path.exists(devices_path):
                with open(devices_path, 'r') as f:
                    self.devices = json.load(f)
                logger.info(f"✅ Loaded {len(self.devices)} devices")
            
            # Load sensors
            sensors_path = os.path.join(self.data_directory, 'sensors.json')
            if os.path.exists(sensors_path):
                with open(sensors_path, 'r') as f:
                    self.sensors = json.load(f)
                logger.info(f"✅ Loaded {len(self.sensors)} sensors")
                
        except Exception as e:
            logger.error(f"❌ Failed to load metadata: {e}")
            raise
    
    def _build_csv_files_cache(self):
        """Build a cache of CSV files organized by device and sensor codes"""
        try:
            csv_pattern = os.path.join(self.data_directory, "*.csv")
            csv_files = glob.glob(csv_pattern)
            
            for csv_file in csv_files:
                filename = os.path.basename(csv_file)
                parts = filename.replace('.csv', '').split('.')
                
                if len(parts) >= 4:
                    device_code = parts[2]
                    sensor_code = parts[3]
                    
                    if device_code not in self.csv_files_cache:
                        self.csv_files_cache[device_code] = {}
                    self.csv_files_cache[device_code][sensor_code] = csv_file
            
            logger.info(f"✅ Built CSV files cache with {len(self.csv_files_cache)} devices")
                
        except Exception as e:
            logger.error(f"❌ Failed to build CSV files cache: {e}")
            raise
    
    def process_csv_files(self, force_reprocess: bool = False):
        """No-op method for compatibility - data is now read on-demand using read_csv_auto"""
        logger.info("✅ Air quality service ready - using on-demand CSV reading")
        return True
    
    def get_locations(self, time: Optional[str] = None, sensor_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get air quality locations with latest data using read_csv_auto"""
        try:
            locations = []
            
            # Filter devices based on sensor types if specified
            target_devices = self.csv_files_cache
            if sensor_types:
                # Filter devices that have the requested sensor types
                filtered_devices = {}
                for device_code, sensors in self.csv_files_cache.items():
                    matching_sensors = {k: v for k, v in sensors.items() if k in sensor_types}
                    if matching_sensors:
                        filtered_devices[device_code] = matching_sensors
                target_devices = filtered_devices
            
            # Process each device
            for device_code, sensors in target_devices.items():
                device_key = f"open/NSW-AIRQ/{device_code}"
                device_info = self.devices.get(device_key, {})
                
                # Extract device metadata
                geometry = device_info.get('geometry', {})
                coordinates = geometry.get('coordinates', [None, None])
                longitude, latitude = coordinates[0], coordinates[1]
                
                if longitude is None or latitude is None:
                    continue
                
                # Process each sensor for this device
                for sensor_code, csv_file in sensors.items():
                    try:
                        # Get sensor metadata
                        sensor_key = f"open/NSW-AIRQ/{device_code}/{sensor_code}"
                        sensor_info = self.sensors.get(sensor_key, {})
                        phenomenon = sensor_info.get('phenomenon', {})
                        
                        # Query the CSV file directly for latest data after 2020
                        query = f"""
                        SELECT 
                            begin as timestamp,
                            v as value
                        FROM read_csv_auto('{csv_file}', header=true)
                        WHERE begin >= '2020-01-01'
                        AND v IS NOT NULL
                        """
                        
                        # Add time filter if specified
                        if time:
                            query += f" AND begin LIKE '{time}%'"
                        
                        query += " ORDER BY begin DESC LIMIT 1"
                        
                        result = self.connection.execute(query).fetchone()
                        
                        if result and result[1] is not None:  # result[1] is the value
                            location = {
                                'device_code': device_code,
                                'device_name': device_info.get('name', ''),
                                'sensor_code': sensor_code,
                                'sensor_name': sensor_info.get('name', ''),
                                'longitude': longitude,
                                'latitude': latitude,
                                'description': device_info.get('description', ''),
                                'uom': phenomenon.get('uom', ''),
                                'phenomenon_name': phenomenon.get('name', ''),
                                'timestamp': result[0],
                                'value': float(result[1])
                            }
                            locations.append(location)
                            
                    except Exception as e:
                        logger.debug(f"⚠️ Failed to process {device_code}/{sensor_code}: {e}")
                        continue
            
            logger.info(f"✅ Retrieved {len(locations)} air quality locations")
            return locations
            
        except Exception as e:
            logger.error(f"❌ Failed to get locations: {e}")
            return []
    
    def get_time_series_data(self, device_code: str, sensor_code: str, 
                           start_time: Optional[str] = None, 
                           end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get time series data for a specific device and sensor using read_csv_auto"""
        try:
            # Find the CSV file for this device/sensor combination
            if device_code not in self.csv_files_cache:
                logger.warning(f"⚠️ Device {device_code} not found")
                return []
            
            if sensor_code not in self.csv_files_cache[device_code]:
                logger.warning(f"⚠️ Sensor {sensor_code} not found for device {device_code}")
                return []
            
            csv_file = self.csv_files_cache[device_code][sensor_code]
            
            # Get device and sensor metadata
            device_key = f"open/NSW-AIRQ/{device_code}"
            sensor_key = f"open/NSW-AIRQ/{device_code}/{sensor_code}"
            
            device_info = self.devices.get(device_key, {})
            sensor_info = self.sensors.get(sensor_key, {})
            phenomenon = sensor_info.get('phenomenon', {})
            
            # Build query
            query = f"""
            SELECT 
                begin as timestamp,
                v as value
            FROM read_csv_auto('{csv_file}', header=true)
            WHERE begin >= '2020-01-01'
            AND v IS NOT NULL
            """
            
            if start_time:
                query += f" AND begin >= '{start_time}'"
            
            if end_time:
                query += f" AND begin <= '{end_time}'"
            
            query += " ORDER BY begin ASC"
            
            result = self.connection.execute(query).fetchall()
            
            time_series = []
            for row in result:
                data_point = {
                    'timestamp': row[0],
                    'value': float(row[1]) if row[1] is not None else None,
                    'device_name': device_info.get('name', ''),
                    'sensor_name': sensor_info.get('name', ''),
                    'uom': phenomenon.get('uom', ''),
                    'phenomenon_name': phenomenon.get('name', '')
                }
                time_series.append(data_point)
            
            logger.info(f"✅ Retrieved {len(time_series)} time series points for {device_code}/{sensor_code}")
            return time_series
            
        except Exception as e:
            logger.error(f"❌ Failed to get time series data: {e}")
            return []
    
    def get_available_sensor_types(self) -> List[Dict[str, str]]:
        """Get all available sensor types from metadata"""
        try:
            sensor_types = []
            
            # Collect unique sensor types from the CSV files cache
            seen_sensors = set()
            
            for device_code, sensors in self.csv_files_cache.items():
                for sensor_code in sensors.keys():
                    sensor_key = f"open/NSW-AIRQ/{device_code}/{sensor_code}"
                    sensor_info = self.sensors.get(sensor_key, {})
                    phenomenon = sensor_info.get('phenomenon', {})
                    
                    # Create a unique key for this sensor type
                    sensor_key_unique = sensor_code
                    
                    if sensor_key_unique not in seen_sensors:
                        seen_sensors.add(sensor_key_unique)
                        
                        sensor_type = {
                            'sensor_code': sensor_code,
                            'sensor_name': sensor_info.get('name', ''),
                            'uom': phenomenon.get('uom', ''),
                            'phenomenon_name': phenomenon.get('name', '')
                        }
                        sensor_types.append(sensor_type)
            
            # Sort by sensor name
            sensor_types.sort(key=lambda x: x['sensor_name'])
            
            logger.info(f"✅ Found {len(sensor_types)} unique sensor types")
            return sensor_types
            
        except Exception as e:
            logger.error(f"❌ Failed to get sensor types: {e}")
            return []
    
    def get_device_summary(self) -> Dict[str, Any]:
        """Get summary information about devices and data coverage"""
        try:
            # Get device count from cache
            device_count = len(self.csv_files_cache)
            
            # Get sensor type count
            all_sensor_types = self.get_available_sensor_types()
            sensor_count = len(all_sensor_types)
            
            # Get top sensors by availability (devices that have this sensor)
            sensor_counts = {}
            for device_code, sensors in self.csv_files_cache.items():
                for sensor_code in sensors.keys():
                    sensor_key = f"open/NSW-AIRQ/{device_code}/{sensor_code}"
                    sensor_info = self.sensors.get(sensor_key, {})
                    sensor_name = sensor_info.get('name', sensor_code)
                    
                    key = f"{sensor_code}_{sensor_name}"
                    if key not in sensor_counts:
                        sensor_counts[key] = {'sensor_code': sensor_code, 'sensor_name': sensor_name, 'device_count': 0}
                    sensor_counts[key]['device_count'] += 1
            
            # Sort by device count and get top 10
            top_sensors = sorted(sensor_counts.values(), key=lambda x: x['device_count'], reverse=True)[:10]
            
            return {
                'device_count': device_count,
                'sensor_count': sensor_count,
                'total_records': 'N/A (on-demand reading)',
                'date_range': {
                    'min_date': '2020-01-01',
                    'max_date': 'Current (on-demand reading)'
                },
                'top_sensors': top_sensors,
                'data_access_method': 'read_csv_auto (on-demand)'
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get device summary: {e}")
            return {}
    
    def close(self):
        """Close the DuckDB connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("🔒 DuckDB connection closed")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()
