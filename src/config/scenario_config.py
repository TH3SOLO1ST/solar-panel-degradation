"""
Scenario Configuration Module

This module handles scenario configuration, validation, and management for solar panel
degradation analysis. It provides structured configuration schemas, validation,
and template generation for different mission scenarios.

References:
- JSON Schema validation standards
- Pydantic configuration management
- NASA Mission Configuration Guidelines
- ESA Configuration Management Standards
"""

import json
import yaml
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

try:
    from pydantic import BaseModel, Field, validator, root_validator
    from pydantic.types import confloat, confloat_ge, conint
except ImportError:
    raise ImportError("Pydantic required for configuration management. Install with: pip install pydantic")


class OrbitType(str, Enum):
    """Supported orbit types"""
    LEO = "LEO"
    MEO = "MEO"
    GEO = "GEO"
    SSO = "SSO"
    MOLNIYA = "MOLNIYA"
    CUSTOM = "CUSTOM"


class SolarCellTechnology(str, Enum):
    """Supported solar cell technologies"""
    SILICON = "silicon"
    MULTI_JUNCTION_GAAS = "multi_junction_gaas"
    MULTI_JUNCTION_INGAP = "multi_junction_inGaP"
    THIN_FILM = "thin_film"
    PEROVSKITE = "perovskite"


class RadiationModel(str, Enum):
    """Supported radiation models"""
    AE8_AP8 = "AE8/AP8"
    AP9_AT9 = "AP9/AT9"
    CREME96 = "CREME96"
    SIMPLIFIED = "simplified"


class ThermalModel(str, Enum):
    """Supported thermal models"""
    DETAILED = "detailed"
    SIMPLIFIED = "simplified"
    STEADY_STATE = "steady_state"


@dataclass
class OrbitalElements:
    """Keplerian orbital elements"""
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float  # Right Ascension of Ascending Node
    arg_perigee_deg: float  # Argument of Perigee
    mean_anomaly_deg: float
    epoch: datetime


class OrbitConfig(BaseModel):
    """Orbital configuration"""
    orbit_type: OrbitType = Field(..., description="Type of orbit")
    altitude_km: Optional[confloat_ge=100] = Field(None, description="Altitude above Earth surface")
    inclination_deg: Optional[confloat(ge=0, le=180)] = Field(None, description="Orbital inclination")
    eccentricity: Optional[confloat_ge=0] = Field(0.0, description="Orbital eccentricity")
    period_hours: Optional[float] = Field(None, description="Orbital period")
    # Advanced orbital elements for custom orbits
    semi_major_axis_km: Optional[float] = Field(None, description="Semi-major axis")
    raan_deg: Optional[float] = Field(0.0, description="Right Ascension of Ascending Node")
    arg_perigee_deg: Optional[float] = Field(0.0, description="Argument of Perigee")
    mean_anomaly_deg: Optional[float] = Field(0.0, description="Mean Anomaly at epoch")
    epoch: Optional[datetime] = Field(default_factory=datetime.now, description="Orbital epoch")

    @validator('semi_major_axis_km')
    def validate_semi_major_axis(cls, v, values):
        if v is None and values.get('altitude_km') is not None:
            # Calculate from altitude (assuming circular orbit)
            return 6378.137 + values['altitude_km']
        return v

    @root_validator
    def validate_orbit_config(cls, values):
        """Validate orbit configuration consistency"""
        orbit_type = values.get('orbit_type')
        altitude = values.get('altitude_km')
        inclination = values.get('inclination_deg')
        eccentricity = values.get('eccentricity', 0.0)

        # Validate orbit-specific requirements
        if orbit_type == OrbitType.GEO:
            if altitude is not None and abs(altitude - 35786) > 100:
                raise ValueError("GEO altitude should be approximately 35786 km")
            if inclination is not None and abs(inclination) > 5:
                raise ValueError("GEO inclination should be near 0 degrees")
            eccentricity = 0.0  # GEO is circular

        elif orbit_type == OrbitType.LEO:
            if altitude is not None and (altitude < 200 or altitude > 2000):
                raise ValueError("LEO altitude should be between 200-2000 km")

        elif orbit_type == OrbitType.MEO:
            if altitude is not None and (altitude < 2000 or altitude > 35786):
                raise ValueError("MEO altitude should be between 2000-35786 km")

        elif orbit_type == OrbitType.SSO:
            if inclination is not None and abs(inclination - 98.0) > 5:
                raise ValueError("SSO inclination should be approximately 98 degrees")

        values['eccentricity'] = eccentricity
        return values


class SolarPanelConfig(BaseModel):
    """Solar panel configuration"""
    technology: SolarCellTechnology = Field(..., description="Solar cell technology")
    area_m2: confloat_ge(0.1) = Field(..., description="Panel area in square meters")
    initial_efficiency: confloat(ge=0.05, le=0.5) = Field(..., description="Initial efficiency")
    degradation_coefficients: Dict[str, float] = Field(default_factory=dict, description="Custom degradation coefficients")
    # Physical properties
    thickness_mm: Optional[float] = Field(5.0, description="Panel thickness in mm")
    mass_kg: Optional[float] = Field(None, description="Panel mass in kg")
    # Electrical properties
    operating_voltage_V: Optional[float] = Field(None, description="Operating voltage")
    max_current_A: Optional[float] = Field(None, description="Maximum current")
    # Thermal properties
    absorptivity: Optional[float] = Field(0.92, description="Solar absorptivity")
    emissivity: Optional[float] = Field(0.85, description="Thermal emissivity")

    @validator('mass_kg')
    def validate_mass(cls, v, values):
        if v is None:
            # Estimate mass from area (typical solar array areal density)
            area = values.get('area_m2', 1.0)
            return area * 15.0  # 15 kg/m² typical for space solar arrays
        return v


class MissionConfig(BaseModel):
    """Mission configuration"""
    duration_years: confloat_ge(0.1) = Field(..., description="Mission duration in years")
    start_time: datetime = Field(default_factory=datetime.now, description="Mission start time")
    time_step_hours: confloat(ge=0.01, le=168) = Field(1.0, description="Simulation time step")
    # Analysis options
    analysis_modes: List[str] = Field(default=["radiation", "thermal", "power"], description="Analysis modes to run")
    fidelity: str = Field("high", description="Simulation fidelity (low/medium/high)")
    # Output options
    save_intermediate_results: bool = Field(True, description="Save intermediate results")
    export_format: str = Field("excel", description="Default export format")


class EnvironmentConfig(BaseModel):
    """Environmental configuration"""
    use_real_data: bool = Field(False, description="Use real space weather data")
    radiation_model: RadiationModel = Field(RadiationModel.SIMPLIFIED, description="Radiation environment model")
    thermal_model: ThermalModel = Field(ThermalModel.DETAILED, description="Thermal analysis model")
    solar_activity: confloat(ge=0, le=1) = Field(0.5, description="Solar activity level (0-1)")
    # Space weather
    include_solar_proton_events: bool = Field(True, description="Include solar particle events")
    include_galactic_cosmic_rays: bool = Field(True, description="Include galactic cosmic rays")
    # Shielding
    shielding_thickness_mm: float = Field(1.0, description="Radiation shielding thickness")
    shielding_material: str = Field("aluminum", description="Shielding material")


class SimulationConfig(BaseModel):
    """Simulation configuration"""
    orbit: OrbitConfig
    solar_panel: SolarPanelConfig
    mission: MissionConfig
    environment: EnvironmentConfig
    scenario_name: str = Field(..., description="Scenario name")
    description: Optional[str] = Field(None, description="Scenario description")
    version: str = Field("1.0", description="Configuration version")
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = Field(None, description="Configuration author")

    class Config:
        """Pydantic configuration"""
        extra = "forbid"  # Forbid extra fields
        schema_extra = {
            "example": {
                "scenario_name": "ISS_LEO_Mission",
                "description": "ISS-like LEO satellite mission",
                "orbit": {
                    "orbit_type": "LEO",
                    "altitude_km": 408,
                    "inclination_deg": 51.64,
                    "eccentricity": 0.0001
                },
                "solar_panel": {
                    "technology": "silicon",
                    "area_m2": 32.4,
                    "initial_efficiency": 0.18
                },
                "mission": {
                    "duration_years": 7.0,
                    "time_step_hours": 1.0
                },
                "environment": {
                    "radiation_model": "AE8/AP8",
                    "solar_activity": 0.5
                }
            }
        }


class ScenarioConfig:
    """
    Scenario configuration management system.

    Features:
    - JSON/YAML configuration loading and validation
    - Pre-defined mission templates
    - Configuration validation and error checking
    - Template generation for different scenarios
    - Configuration import/export
    """

    def __init__(self):
        """Initialize scenario configuration manager"""
        self.config: Optional[SimulationConfig] = None
        self.templates = self._load_default_templates()

    def load_config(self, filepath: str) -> SimulationConfig:
        """
        Load configuration from file

        Args:
            filepath: Path to configuration file

        Returns:
            Validated simulation configuration
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            # Validate configuration
            self.config = SimulationConfig(**data)
            return self.config

        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")

    def save_config(self, config: SimulationConfig, filepath: str, format: str = "json"):
        """
        Save configuration to file

        Args:
            config: Configuration to save
            filepath: Output file path
            format: Output format ("json" or "yaml")
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = config.dict()

        try:
            with open(path, 'w') as f:
                if format.lower() in ['yaml', 'yml']:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(data, f, indent=2, default=str)

        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")

    def create_from_template(self, template_name: str, **kwargs) -> SimulationConfig:
        """
        Create configuration from predefined template

        Args:
            template_name: Name of template
            **kwargs: Parameters to override

        Returns:
            Configured simulation scenario
        """
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")

        template_data = self.templates[template_name].copy()

        # Apply overrides
        self._deep_update(template_data, kwargs)

        # Validate and return configuration
        return SimulationConfig(**template_data)

    def validate_config(self, config_data: Dict) -> SimulationConfig:
        """
        Validate configuration data

        Args:
            config_data: Configuration data dictionary

        Returns:
            Validated configuration
        """
        return SimulationConfig(**config_data)

    def get_config_schema(self) -> Dict:
        """
        Get configuration schema

        Returns:
            JSON schema for configuration
        """
        return SimulationConfig.schema()

    def list_templates(self) -> List[str]:
        """
        List available templates

        Returns:
            List of template names
        """
        return list(self.templates.keys())

    def _load_default_templates(self) -> Dict[str, Dict]:
        """Load default mission templates"""
        return {
            "ISS_Like": {
                "scenario_name": "ISS_Like_Mission",
                "description": "International Space Station-like LEO mission",
                "orbit": {
                    "orbit_type": "LEO",
                    "altitude_km": 408,
                    "inclination_deg": 51.64,
                    "eccentricity": 0.0001
                },
                "solar_panel": {
                    "technology": "silicon",
                    "area_m2": 32.4,
                    "initial_efficiency": 0.18
                },
                "mission": {
                    "duration_years": 7.0,
                    "time_step_hours": 1.0
                },
                "environment": {
                    "radiation_model": "AE8/AP8",
                    "solar_activity": 0.5,
                    "use_real_data": False
                }
            },

            "GEO_Communication": {
                "scenario_name": "GEO_Communication_Satellite",
                "description": "Geostationary communication satellite",
                "orbit": {
                    "orbit_type": "GEO",
                    "altitude_km": 35786,
                    "inclination_deg": 0.0,
                    "eccentricity": 0.0
                },
                "solar_panel": {
                    "technology": "multi_junction_gaas",
                    "area_m2": 80.5,
                    "initial_efficiency": 0.32
                },
                "mission": {
                    "duration_years": 15.0,
                    "time_step_hours": 24.0
                },
                "environment": {
                    "radiation_model": "AP9/AT9",
                    "solar_activity": 0.7,
                    "use_real_data": True
                }
            },

            "SSO_Earth_Observation": {
                "scenario_name": "SSO_Earth_Observation",
                "description": "Sun-synchronous Earth observation satellite",
                "orbit": {
                    "orbit_type": "SSO",
                    "altitude_km": 785,
                    "inclination_deg": 98.6,
                    "eccentricity": 0.001
                },
                "solar_panel": {
                    "technology": "silicon",
                    "area_m2": 45.2,
                    "initial_efficiency": 0.22
                },
                "mission": {
                    "duration_years": 5.0,
                    "time_step_hours": 0.5
                },
                "environment": {
                    "radiation_model": "AE8/AP8",
                    "solar_activity": 0.3,
                    "use_real_data": False
                }
            },

            "High_Radiation_MEO": {
                "scenario_name": "High_Radiation_MEO_Satellite",
                "description": "Medium Earth Orbit satellite in high radiation environment",
                "orbit": {
                    "orbit_type": "MEO",
                    "altitude_km": 20000,
                    "inclination_deg": 55.0,
                    "eccentricity": 0.1
                },
                "solar_panel": {
                    "technology": "multi_junction_inGaP",
                    "area_m2": 25.0,
                    "initial_efficiency": 0.30
                },
                "mission": {
                    "duration_years": 10.0,
                    "time_step_hours": 6.0
                },
                "environment": {
                    "radiation_model": "AP9/AT9",
                    "solar_activity": 0.8,
                    "use_real_data": True
                }
            },

            "CubeSat_Lite": {
                "scenario_name": "CubeSat_Lite",
                "description": "Small CubeSat mission",
                "orbit": {
                    "orbit_type": "LEO",
                    "altitude_km": 500,
                    "inclination_deg": 45.0,
                    "eccentricity": 0.001
                },
                "solar_panel": {
                    "technology": "silicon",
                    "area_m2": 0.1,
                    "initial_efficiency": 0.20
                },
                "mission": {
                    "duration_years": 2.0,
                    "time_step_hours": 0.25
                },
                "environment": {
                    "radiation_model": "SIMPLIFIED",
                    "solar_activity": 0.5,
                    "use_real_data": False
                }
            }
        }

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def generate_config_summary(self, config: SimulationConfig) -> Dict:
        """
        Generate configuration summary

        Args:
            config: Simulation configuration

        Returns:
            Configuration summary dictionary
        """
        return {
            'scenario_name': config.scenario_name,
            'description': config.description,
            'mission_duration_years': config.mission.duration_years,
            'orbit_type': config.orbit.orbit_type.value,
            'orbital_parameters': {
                'altitude_km': config.orbit.altitude_km,
                'inclination_deg': config.orbit.inclination_deg,
                'eccentricity': config.orbit.eccentricity
            },
            'solar_panel': {
                'technology': config.solar_panel.technology.value,
                'area_m2': config.solar_panel.area_m2,
                'initial_efficiency': config.solar_panel.initial_efficiency
            },
            'environment': {
                'radiation_model': config.environment.radiation_model.value,
                'solar_activity': config.environment.solar_activity,
                'use_real_data': config.environment.use_real_data
            },
            'simulation_settings': {
                'time_step_hours': config.mission.time_step_hours,
                'fidelity': config.mission.fidelity,
                'analysis_modes': config.mission.analysis_modes
            }
        }

    def compare_configs(self, config1: SimulationConfig, config2: SimulationConfig) -> Dict:
        """
        Compare two configurations

        Args:
            config1: First configuration
            config2: Second configuration

        Returns:
            Comparison results
        """
        differences = {}

        # Compare basic parameters
        if config1.orbit.orbit_type != config2.orbit.orbit_type:
            differences['orbit_type'] = {
                'config1': config1.orbit.orbit_type.value,
                'config2': config2.orbit.orbit_type.value
            }

        if config1.orbit.altitude_km != config2.orbit.altitude_km:
            differences['altitude_km'] = {
                'config1': config1.orbit.altitude_km,
                'config2': config2.orbit.altitude_km
            }

        if config1.solar_panel.technology != config2.solar_panel.technology:
            differences['solar_cell_technology'] = {
                'config1': config1.solar_panel.technology.value,
                'config2': config2.solar_panel.technology.value
            }

        if config1.solar_panel.area_m2 != config2.solar_panel.area_m2:
            differences['panel_area'] = {
                'config1': config1.solar_panel.area_m2,
                'config2': config2.solar_panel.area_m2
            }

        if config1.mission.duration_years != config2.mission.duration_years:
            differences['mission_duration'] = {
                'config1': config1.mission.duration_years,
                'config2': config2.mission.duration_years
            }

        return differences

    def export_config_template(self, filepath: str, include_examples: bool = True):
        """
        Export configuration template with documentation

        Args:
            filepath: Output file path
            include_examples: Include example values
        """
        schema = self.get_config_schema()

        template_data = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Solar Panel Degradation Analysis Configuration",
            "description": "Configuration schema for solar panel degradation simulation",
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
        }

        if include_examples:
            template_data["examples"] = [
                self.templates["ISS_Like"],
                self.templates["GEO_Communication"]
            ]

        with open(filepath, 'w') as f:
            json.dump(template_data, f, indent=2)

    def validate_mission_feasibility(self, config: SimulationConfig) -> Dict:
        """
        Validate mission feasibility and provide recommendations

        Args:
            config: Simulation configuration

        Returns:
            Feasibility analysis and recommendations
        """
        issues = []
        warnings = []
        recommendations = []

        # Check orbit validity
        if config.orbit.orbit_type == OrbitType.LEO:
            if config.orbit.altitude_km < 300:
                warnings.append("Low altitude may result in rapid orbital decay")
                recommendations.append("Consider orbital altitude > 300 km for extended missions")

            if config.environment.radiation_model == RadiationModel.SIMPLIFIED:
                warnings.append("Simplified radiation model may not capture South Atlantic Anomaly effects")
                recommendations.append("Use AE8/AP8 or AP9 model for accurate LEO radiation assessment")

        # Check solar panel sizing
        expected_degradation = 0.15 * config.mission.duration_years  # Rough estimate
        if config.solar_panel.initial_efficiency * (1 - expected_degradation) < 0.1:
            warnings.append("End-of-life efficiency may be below 10%")
            recommendations.append("Consider larger panel area or higher efficiency cells")

        # Check mission duration
        if config.mission.duration_years > 20:
            warnings.append("Very long mission duration increases uncertainty")
            recommendations.append("Consider conservative degradation estimates")

        # Check analysis modes
        if "thermal" not in config.mission.analysis_modes:
            warnings.append("Thermal analysis not included - missing important degradation mechanism")
            recommendations.append("Include thermal analysis for comprehensive assessment")

        return {
            "feasible": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "estimated_eol_efficiency": config.solar_panel.initial_efficiency * (1 - expected_degradation)
        }