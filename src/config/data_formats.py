"""
Data Formats Module

This module defines standardized data formats for input/output operations,
data exchange between modules, and external data integration. It provides
consistent schemas and validation for all data types used in the analysis.

References:
- NASA Data Format Standards
- ESA Data Exchange Protocols
- JSON Schema specifications
- CSV format standards
- HDF5 data organization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from pydantic import BaseModel, Field, validator


class DataFormatType(str, Enum):
    """Supported data format types"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    HDF5 = "hdf5"
    MATLAB = "matlab"
    YAML = "yaml"


class DataVersion(str, Enum):
    """Data format versioning"""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


@dataclass
class DataSchema:
    """Data schema definition"""
    name: str
    version: DataVersion
    fields: Dict[str, type]
    required_fields: List[str]
    description: str
    units: Dict[str, str] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


class OrbitalDataFormat(BaseModel):
    """Standard format for orbital data"""
    timestamp: datetime = Field(..., description="Timestamp in UTC")
    position_x_km: float = Field(..., description="X position in ECI coordinates (km)")
    position_y_km: float = Field(..., description="Y position in ECI coordinates (km)")
    position_z_km: float = Field(..., description="Z position in ECI coordinates (km)")
    velocity_x_km_s: float = Field(..., description="X velocity in ECI coordinates (km/s)")
    velocity_y_km_s: float = Field(..., description="Y velocity in ECI coordinates (km/s)")
    velocity_z_km_s: float = Field(..., description="Z velocity in ECI coordinates (km/s)")
    altitude_km: float = Field(..., description="Altitude above Earth surface (km)")
    latitude_deg: float = Field(..., description="Geodetic latitude (degrees)")
    longitude_deg: float = Field(..., description="Geodetic longitude (degrees)")

    @validator('altitude_km')
    def validate_altitude(cls, v):
        if v < 100:
            raise ValueError("Altitude too low (minimum 100 km)")
        return v


class RadiationDataFormat(BaseModel):
    """Standard format for radiation data"""
    timestamp: datetime = Field(..., description="Timestamp in UTC")
    particle_type: str = Field(..., description="Particle type (electron, proton, heavy_ion)")
    energy_mev: float = Field(..., description="Particle energy (MeV)")
    flux_particles_cm2_s_sr_mev: float = Field(..., description="Particle flux (particles/cm²/s/sr/MeV)")
    dose_rads: Optional[float] = Field(None, description="Radiation dose (rads)")
    ddd_mev_cm2_g: Optional[float] = Field(None, description="Displacement damage dose (MeV·cm²/g)")

    @validator('particle_type')
    def validate_particle_type(cls, v):
        allowed = ['electron', 'proton', 'alpha', 'heavy_ion']
        if v not in allowed:
            raise ValueError(f"Particle type must be one of: {allowed}")
        return v


class ThermalDataFormat(BaseModel):
    """Standard format for thermal data"""
    timestamp: datetime = Field(..., description="Timestamp in UTC")
    temperature_K: float = Field(..., description="Panel temperature (K)")
    heat_flux_solar_W_m2: float = Field(..., description="Solar heat flux (W/m²)")
    heat_flux_albedo_W_m2: float = Field(0.0, description="Earth albedo heat flux (W/m²)")
    heat_flux_earth_ir_W_m2: float = Field(0.0, description="Earth IR heat flux (W/m²)")
    heat_flux_radiated_W_m2: float = Field(..., description="Radiated heat flux (W/m²)")
    net_heat_flux_W_m2: float = Field(..., description="Net heat flux (W/m²)")
    eclipse_status: bool = Field(False, description="True if in eclipse")
    temperature_gradient_K_m: float = Field(0.0, description="Temperature gradient (K/m)")

    @validator('temperature_K')
    def validate_temperature(cls, v):
        if v < 100 or v > 500:
            raise ValueError("Temperature out of reasonable range (100-500 K)")
        return v


class PowerDataFormat(BaseModel):
    """Standard format for power data"""
    timestamp: datetime = Field(..., description="Timestamp in UTC")
    power_W: float = Field(..., description="Total power output (W)")
    power_density_W_m2: float = Field(..., description="Power density (W/m²)")
    current_A: float = Field(..., description="Output current (A)")
    voltage_V: float = Field(..., description="Output voltage (V)")
    efficiency: float = Field(..., description="Conversion efficiency (0-1)")
    irradiance_W_m2: float = Field(..., description="Solar irradiance (W/m²)")
    temperature_K: float = Field(..., description="Cell temperature (K)")
    degradation_factor: float = Field(..., description="Overall degradation factor (0-1)")
    eclipse_fraction: float = Field(0.0, description="Eclipse shadow fraction (0-1)")

    @validator('efficiency', 'degradation_factor', 'eclipse_fraction')
    def validate_fractions(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Value must be between 0 and 1")
        return v


class DegradationDataFormat(BaseModel):
    """Standard format for degradation data"""
    timestamp: datetime = Field(..., description="Timestamp in UTC")
    mission_time_hours: float = Field(..., description="Time since mission start (hours)")
    initial_power_W: float = Field(..., description="Initial power output (W)")
    current_power_W: float = Field(..., description="Current power output (W)")
    power_degradation_percent: float = Field(..., description="Power degradation (%)")
    efficiency_factor: float = Field(..., description="Relative efficiency (0-1)")
    radiation_damage_factor: float = Field(..., description="Radiation damage factor (0-1)")
    thermal_damage_factor: float = Field(..., description="Thermal damage factor (0-1)")
    contamination_damage_factor: float = Field(..., description="Contamination damage factor (0-1)")
    aging_damage_factor: float = Field(..., description="Normal aging damage factor (0-1)")
    total_radiation_dose_rads: float = Field(..., description="Total accumulated radiation dose (rads)")
    total_thermal_cycles: int = Field(..., description="Total thermal cycles")
    max_temperature_K: float = Field(..., description="Maximum temperature experienced (K)")
    min_temperature_K: float = Field(..., description="Minimum temperature experienced (K)")
    total_eclipse_hours: float = Field(..., description="Total time in eclipse (hours)")


class DataFormats:
    """
    Standardized data formats manager for solar panel degradation analysis.

    Features:
    - Standardized data schemas
    - Data validation and conversion
    - Format conversion utilities
    - Import/export helpers
    - Version management
    """

    def __init__(self):
        """Initialize data formats manager"""
        self.schemas = self._define_schemas()
        self.validators = self._setup_validators()

    def _define_schemas(self) -> Dict[str, DataSchema]:
        """Define standard data schemas"""
        return {
            "orbital": DataSchema(
                name="orbital_data",
                version=DataVersion.V1_0,
                fields={
                    "timestamp": datetime,
                    "position_x_km": float,
                    "position_y_km": float,
                    "position_z_km": float,
                    "velocity_x_km_s": float,
                    "velocity_y_km_s": float,
                    "velocity_z_km_s": float,
                    "altitude_km": float,
                    "latitude_deg": float,
                    "longitude_deg": float
                },
                required_fields=["timestamp", "position_x_km", "position_y_km", "position_z_km"],
                description="Satellite orbital state data",
                units={
                    "position": "km",
                    "velocity": "km/s",
                    "altitude": "km",
                    "latitude": "degrees",
                    "longitude": "degrees"
                }
            ),
            "radiation": DataSchema(
                name="radiation_data",
                version=DataVersion.V1_0,
                fields={
                    "timestamp": datetime,
                    "particle_type": str,
                    "energy_mev": float,
                    "flux_particles_cm2_s_sr_mev": float,
                    "dose_rads": float,
                    "ddd_mev_cm2_g": float
                },
                required_fields=["timestamp", "particle_type", "energy_mev", "flux_particles_cm2_s_sr_mev"],
                description="Space radiation environment data",
                units={
                    "energy": "MeV",
                    "flux": "particles/cm²/s/sr/MeV",
                    "dose": "rads",
                    "ddd": "MeV·cm²/g"
                }
            ),
            "thermal": DataSchema(
                name="thermal_data",
                version=DataVersion.V1_0,
                fields={
                    "timestamp": datetime,
                    "temperature_K": float,
                    "heat_flux_solar_W_m2": float,
                    "heat_flux_albedo_W_m2": float,
                    "heat_flux_earth_ir_W_m2": float,
                    "heat_flux_radiated_W_m2": float,
                    "net_heat_flux_W_m2": float,
                    "eclipse_status": bool,
                    "temperature_gradient_K_m": float
                },
                required_fields=["timestamp", "temperature_K", "net_heat_flux_W_m2"],
                description="Thermal analysis data",
                units={
                    "temperature": "K",
                    "heat_flux": "W/m²",
                    "temperature_gradient": "K/m"
                }
            ),
            "power": DataSchema(
                name="power_data",
                version=DataVersion.V1_0,
                fields={
                    "timestamp": datetime,
                    "power_W": float,
                    "power_density_W_m2": float,
                    "current_A": float,
                    "voltage_V": float,
                    "efficiency": float,
                    "irradiance_W_m2": float,
                    "temperature_K": float,
                    "degradation_factor": float,
                    "eclipse_fraction": float
                },
                required_fields=["timestamp", "power_W", "current_A", "voltage_V"],
                description="Solar panel power output data",
                units={
                    "power": "W",
                    "power_density": "W/m²",
                    "current": "A",
                    "voltage": "V",
                    "irradiance": "W/m²",
                    "temperature": "K"
                }
            ),
            "degradation": DataSchema(
                name="degradation_data",
                version=DataVersion.V1_0,
                fields={
                    "timestamp": datetime,
                    "mission_time_hours": float,
                    "initial_power_W": float,
                    "current_power_W": float,
                    "power_degradation_percent": float,
                    "efficiency_factor": float,
                    "radiation_damage_factor": float,
                    "thermal_damage_factor": float,
                    "contamination_damage_factor": float,
                    "aging_damage_factor": float,
                    "total_radiation_dose_rads": float,
                    "total_thermal_cycles": int,
                    "max_temperature_K": float,
                    "min_temperature_K": float,
                    "total_eclipse_hours": float
                },
                required_fields=["timestamp", "mission_time_hours", "current_power_W"],
                description="Lifetime degradation analysis data",
                units={
                    "power": "W",
                    "time": "hours",
                    "degradation": "percent",
                    "dose": "rads",
                    "temperature": "K"
                }
            )
        }

    def _setup_validators(self) -> Dict[str, type]:
        """Setup data validators"""
        return {
            "orbital": OrbitalDataFormat,
            "radiation": RadiationDataFormat,
            "thermal": ThermalDataFormat,
            "power": PowerDataFormat,
            "degradation": DegradationDataFormat
        }

    def validate_data(self, data_type: str, data: Union[Dict, List]) -> bool:
        """
        Validate data against schema

        Args:
            data_type: Type of data to validate
            data: Data to validate (dict or list of dicts)

        Returns:
            True if valid
        """
        if data_type not in self.validators:
            raise ValueError(f"Unknown data type: {data_type}")

        validator = self.validators[data_type]

        try:
            if isinstance(data, list):
                for item in data:
                    validator(**item)
            else:
                validator(**data)
            return True
        except Exception as e:
            print(f"Validation error for {data_type}: {e}")
            return False

    def convert_to_dataframe(self, data_type: str, data: List[Dict]) -> pd.DataFrame:
        """
        Convert validated data to pandas DataFrame

        Args:
            data_type: Type of data
            data: List of data dictionaries

        Returns:
            Pandas DataFrame
        """
        if not data:
            return pd.DataFrame()

        # Validate data first
        if not self.validate_data(data_type, data):
            raise ValueError(f"Invalid {data_type} data")

        return pd.DataFrame(data)

    def export_to_format(self, data_type: str, data: List[Dict],
                        filepath: str, format: str = "csv") -> bool:
        """
        Export data to specified format

        Args:
            data_type: Type of data
            data: Data to export
            filepath: Output file path
            format: Export format

        Returns:
            True if successful
        """
        try:
            df = self.convert_to_dataframe(data_type, data)

            if format.lower() == "csv":
                df.to_csv(filepath, index=False)
            elif format.lower() == "excel":
                df.to_excel(filepath, index=False)
            elif format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format.lower() == "hdf5":
                df.to_hdf(filepath, key=data_type, mode='w')
            else:
                raise ValueError(f"Unsupported export format: {format}")

            return True

        except Exception as e:
            print(f"Export failed: {e}")
            return False

    def import_from_format(self, data_type: str, filepath: str,
                          format: str = "csv") -> List[Dict]:
        """
        Import data from file

        Args:
            data_type: Type of data expected
            filepath: Input file path
            format: File format

        Returns:
            List of data dictionaries
        """
        try:
            if format.lower() == "csv":
                df = pd.read_csv(filepath)
            elif format.lower() == "excel":
                df = pd.read_excel(filepath)
            elif format.lower() == "json":
                with open(filepath, 'r') as f:
                    return json.load(f)
            elif format.lower() == "hdf5":
                df = pd.read_hdf(filepath, key=data_type)
            else:
                raise ValueError(f"Unsupported import format: {format}")

            # Convert to list of dicts
            data = df.to_dict('records')

            # Validate imported data
            if not self.validate_data(data_type, data):
                raise ValueError(f"Imported data failed validation for {data_type}")

            return data

        except Exception as e:
            print(f"Import failed: {e}")
            return []

    def create_template(self, data_type: str, filepath: str) -> bool:
        """
        Create empty template file for data type

        Args:
            data_type: Type of data
            filepath: Output template file path

        Returns:
            True if successful
        """
        if data_type not in self.schemas:
            raise ValueError(f"Unknown data type: {data_type}")

        schema = self.schemas[data_type]

        # Create empty record with all fields
        template_record = {}
        for field_name, field_type in schema.fields.items():
            if field_type == datetime:
                template_record[field_name] = datetime.now().isoformat()
            elif field_type == float:
                template_record[field_name] = 0.0
            elif field_type == int:
                template_record[field_name] = 0
            elif field_type == str:
                template_record[field_name] = ""
            elif field_type == bool:
                template_record[field_name] = False

        # Create template with documentation
        template_data = {
            "schema_info": {
                "name": schema.name,
                "version": schema.version.value,
                "description": schema.description,
                "required_fields": schema.required_fields,
                "units": schema.units
            },
            "template_record": template_record,
            "example_records": []
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(template_data, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Template creation failed: {e}")
            return False

    def get_schema_info(self, data_type: str) -> Dict:
        """
        Get schema information for data type

        Args:
            data_type: Type of data

        Returns:
            Schema information dictionary
        """
        if data_type not in self.schemas:
            raise ValueError(f"Unknown data type: {data_type}")

        schema = self.schemas[data_type]

        return {
            "name": schema.name,
            "version": schema.version.value,
            "description": schema.description,
            "fields": list(schema.fields.keys()),
            "field_types": {k: v.__name__ for k, v in schema.fields.items()},
            "required_fields": schema.required_fields,
            "units": schema.units,
            "validation_rules": schema.validation_rules
        }

    def merge_datasets(self, data_type: str, datasets: List[List[Dict]]) -> List[Dict]:
        """
        Merge multiple datasets of the same type

        Args:
            data_type: Type of data
            datasets: List of datasets to merge

        Returns:
            Merged dataset
        """
        if not datasets:
            return []

        # Validate all datasets
        for i, dataset in enumerate(datasets):
            if not self.validate_data(data_type, dataset):
                raise ValueError(f"Dataset {i} failed validation")

        # Merge datasets (simple concatenation)
        merged = []
        for dataset in datasets:
            merged.extend(dataset)

        # Sort by timestamp if available
        if merged and 'timestamp' in merged[0]:
            merged.sort(key=lambda x: x['timestamp'])

        return merged

    def filter_data(self, data: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """
        Filter data based on criteria

        Args:
            data: Data to filter
            filters: Filter criteria

        Returns:
            Filtered data
        """
        if not data:
            return []

        filtered_data = data.copy()

        for field, criteria in filters.items():
            if isinstance(criteria, dict):
                # Range filter
                if 'min' in criteria:
                    filtered_data = [d for d in filtered_data if d.get(field, 0) >= criteria['min']]
                if 'max' in criteria:
                    filtered_data = [d for d in filtered_data if d.get(field, 0) <= criteria['max']]
            elif isinstance(criteria, list):
                # Value list filter
                filtered_data = [d for d in filtered_data if d.get(field) in criteria]
            else:
                # Exact match filter
                filtered_data = [d for d in filtered_data if d.get(field) == criteria]

        return filtered_data

    def get_data_statistics(self, data_type: str, data: List[Dict]) -> Dict:
        """
        Calculate statistics for data

        Args:
            data_type: Type of data
            data: Data to analyze

        Returns:
            Statistics dictionary
        """
        if not data:
            return {}

        df = self.convert_to_dataframe(data_type, data)
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        stats = {
            "total_records": len(data),
            "time_range": {
                "start": min(d['timestamp'] for d in data).isoformat(),
                "end": max(d['timestamp'] for d in data).isoformat()
            },
            "field_statistics": {}
        }

        for col in numeric_columns:
            stats["field_statistics"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "count": int(df[col].count())
            }

        return stats