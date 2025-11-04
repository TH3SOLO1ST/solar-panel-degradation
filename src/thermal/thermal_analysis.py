"""
Thermal Analysis
================

Calculates orbital temperature profiles for solar panels considering
solar heating, radiative cooling, and eclipse periods.

This module implements thermal balance equations and provides temperature
profiles used for degradation analysis.

Classes:
    ThermalAnalysis: Main thermal analysis class
    ThermalProperties: Material properties for thermal calculations
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ThermalProperties:
    """Thermal properties of solar panel materials"""
    mass: float  # kg
    specific_heat: float  # J/(kg·K)
    emissivity: float  # 0 to 1
    absorptivity: float  # 0 to 1
    area: float  # m²
    thermal_capacity: float  # J/K (derived)

    def __post_init__(self):
        """Calculate thermal capacity from mass and specific heat"""
        self.thermal_capacity = self.mass * self.specific_heat

class ThermalAnalysis:
    """Main thermal analysis class for solar panels"""

    def __init__(self, thermal_props: ThermalProperties):
        """
        Initialize thermal analysis

        Args:
            thermal_props: ThermalProperties object
        """
        self.props = thermal_props

        # Physical constants
        self.stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        self.solar_constant = 1361.0  # W/m² at 1 AU
        self.space_temperature = 3.0  # K (deep space)

        # Simplified thermal parameters
        self.albedo_coefficient = 0.3  # Earth reflectance
        self.earth_ir_coefficient = 0.7  # Earth IR emission

    def calculate_temperature_profile(self, positions: np.ndarray, times: np.ndarray,
                                    sun_position: np.ndarray,
                                    eclipse_periods: List = None,
                                    in_sunlight: np.ndarray = None) -> np.ndarray:
        """
        Calculate temperature profile over time

        Args:
            positions: Nx3 array of satellite positions in km
            times: Array of time points in hours
            sun_position: Sun position vector
            eclipse_periods: List of eclipse periods (optional)
            in_sunlight: Boolean array indicating sunlight periods

        Returns:
            Array of temperatures in Kelvin
        """
        n_points = len(positions)
        temperatures = np.zeros(n_points)

        # Initial temperature (room temperature)
        temperatures[0] = 293.0  # K

        # Time step in seconds
        if len(times) > 1:
            dt_seconds = (times[1] - times[0]) * 3600
        else:
            dt_seconds = 3600  # 1 hour default

        for i in range(1, n_points):
            # Determine if in sunlight
            if in_sunlight is not None:
                sunlit = in_sunlight[i]
            else:
                # Simple calculation: sunlit if facing sun
                sunlit = self._is_sunlit(positions[i], sun_position)

            # Calculate heat fluxes
            if sunlit:
                # Solar heating
                solar_flux = self._calculate_solar_heating(positions[i], sun_position)
                # Earth albedo (simplified)
                albedo_flux = self._calculate_albedo_heating(positions[i], sun_position)
                # Earth IR radiation
                earth_ir_flux = self._calculate_earth_ir_heating(positions[i])

                # Total heating
                heating_power = (solar_flux + albedo_flux + earth_ir_flux) * self.props.area
            else:
                # Eclipse - only Earth IR heating
                earth_ir_flux = self._calculate_earth_ir_heating(positions[i])
                heating_power = earth_ir_flux * self.props.area

            # Radiative cooling
            cooling_power = self._calculate_radiative_cooling(temperatures[i-1])

            # Net heat flow
            net_heat_flow = heating_power - cooling_power

            # Temperature change
            delta_T = (net_heat_flow * dt_seconds) / self.props.thermal_capacity
            temperatures[i] = temperatures[i-1] + delta_T

            # Ensure reasonable temperature bounds
            temperatures[i] = np.clip(temperatures[i], 100, 400)  # K

        return temperatures

    def _is_sunlit(self, position: np.ndarray, sun_position: np.ndarray) -> bool:
        """
        Simple calculation to determine if satellite is in sunlight

        Args:
            position: Satellite position in km
            sun_position: Sun position vector

        Returns:
            True if satellite is in sunlight
        """
        # Simplified: check if satellite is on sun-facing side of Earth
        sun_direction = sun_position / np.linalg.norm(sun_position)
        pos_direction = position / np.linalg.norm(position)

        dot_product = np.dot(sun_direction, pos_direction)
        return dot_product > 0

    def _calculate_solar_heating(self, position: np.ndarray, sun_position: np.ndarray) -> float:
        """
        Calculate solar heating flux

        Args:
            position: Satellite position in km
            sun_position: Sun position vector

        Returns:
            Solar heating flux in W/m²
        """
        # Simplified: assume normal incidence
        incident_angle_factor = 1.0

        # Distance from sun (simplified - assumes 1 AU)
        distance_factor = 1.0

        solar_flux = (self.solar_constant * incident_angle_factor *
                     self.props.absorptivity * distance_factor)

        return solar_flux

    def _calculate_albedo_heating(self, position: np.ndarray, sun_position: np.ndarray) -> float:
        """
        Calculate Earth albedo heating

        Args:
            position: Satellite position in km
            sun_position: Sun position vector

        Returns:
            Albedo heating flux in W/m²
        """
        # Simplified albedo calculation
        altitude_km = np.linalg.norm(position) - 6371.0  # Earth radius

        if altitude_km < 0:
            return 0.0

        # Albedo decreases with altitude
        altitude_factor = np.exp(-altitude_km / 1000.0)  # Scale height ~1000 km

        albedo_flux = (self.solar_constant * self.albedo_coefficient *
                      self.props.absorptivity * altitude_factor * 0.1)  # Reduced factor

        return albedo_flux

    def _calculate_earth_ir_heating(self, position: np.ndarray) -> float:
        """
        Calculate Earth infrared radiation heating

        Args:
            position: Satellite position in km

        Returns:
            Earth IR heating flux in W/m²
        """
        # Simplified Earth IR calculation
        altitude_km = np.linalg.norm(position) - 6371.0

        if altitude_km < 0:
            return 0.0

        # Earth IR decreases with altitude
        earth_ir_flux = 237.0 * self.props.absorptivity * np.exp(-altitude_km / 1000.0)

        return earth_ir_flux

    def _calculate_radiative_cooling(self, temperature: float) -> float:
        """
        Calculate radiative cooling power

        Args:
            temperature: Current temperature in Kelvin

        Returns:
            Cooling power in Watts
        """
        # Stefan-Boltzmann law
        cooling_flux = (self.props.emissivity * self.stefan_boltzmann *
                       (temperature**4 - self.space_temperature**4))

        cooling_power = cooling_flux * self.props.area
        return cooling_power

    def get_thermal_statistics(self, temperatures: np.ndarray) -> Dict:
        """
        Calculate thermal statistics from temperature profile

        Args:
            temperatures: Array of temperatures in Kelvin

        Returns:
            Dictionary with thermal statistics
        """
        if len(temperatures) == 0:
            return {}

        # Convert to Celsius for user-friendly statistics
        temps_celsius = temperatures - 273.15

        return {
            'min_temperature_K': np.min(temperatures),
            'max_temperature_K': np.max(temperatures),
            'mean_temperature_K': np.mean(temperatures),
            'min_temperature_C': np.min(temps_celsius),
            'max_temperature_C': np.max(temps_celsius),
            'mean_temperature_C': np.mean(temps_celsius),
            'temperature_range_K': np.max(temperatures) - np.min(temperatures),
            'thermal_cycles': self._count_thermal_cycles(temperatures)
        }

    def _count_thermal_cycles(self, temperatures: np.ndarray,
                            threshold_K: float = 10.0) -> int:
        """
        Count thermal cycles in temperature profile

        Args:
            temperatures: Array of temperatures in Kelvin
            threshold_K: Temperature change threshold for cycle counting

        Returns:
            Number of thermal cycles
        """
        cycles = 0
        increasing = None

        for i in range(1, len(temperatures)):
            temp_change = temperatures[i] - temperatures[i-1]

            if abs(temp_change) > threshold_K:
                currently_increasing = temp_change > 0

                if increasing is not None and currently_increasing != increasing:
                    cycles += 1

                increasing = currently_increasing

        return cycles

    def calculate_thermal_stress(self, temperatures: np.ndarray,
                                material_cte: float = 2.6e-6) -> np.ndarray:
        """
        Calculate thermal stress from temperature changes

        Args:
            temperatures: Array of temperatures in Kelvin
            material_cte: Coefficient of thermal expansion (1/K)

        Returns:
            Array of thermal stress values (normalized)
        """
        # Reference temperature (room temperature)
        T_ref = 293.0  # K

        # Temperature difference from reference
        delta_T = temperatures - T_ref

        # Thermal strain (simplified)
        thermal_strain = material_cte * delta_T

        # Assume elastic modulus and calculate stress (normalized)
        # For simplicity, return strain as stress indicator
        thermal_stress = thermal_strain

        return thermal_stress