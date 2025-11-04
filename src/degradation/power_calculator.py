"""
Power Calculator
================

Calculates instantaneous solar panel power output considering all
environmental factors and degradation mechanisms.

This module implements the physics-based power calculation models
for different solar cell technologies under various operating conditions.

Classes:
    PowerCalculator: Main power calculation class
    SolarCellModel: Solar cell electrical characteristics
    EnvironmentalFactors: Environmental condition corrections
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SolarCellSpecs:
    """Solar panel specifications"""
    technology: str          # 'silicon', 'multi_junction'
    area_m2: float          # Panel area in m²
    initial_efficiency: float  # Initial efficiency (0 to 1)
    series_resistance: float   # Series resistance in ohms
    shunt_resistance: float    # Shunt resistance in ohms
    ideality_factor: float     # Diode ideality factor
    temperature_coefficient: float  # Temperature coefficient per K
    reference_temperature: float  # Reference temperature in K

@dataclass
class EnvironmentalConditions:
    """Environmental operating conditions"""
    solar_flux: float         # Solar flux in W/m²
    incident_angle: float     # Incident angle in radians
    temperature: float        # Cell temperature in K
    radiation_factor: float   # Radiation degradation factor
    thermal_factor: float     # Thermal degradation factor
    contamination_factor: float  # Contamination degradation factor

class SolarCellModel:
    """Solar cell electrical characteristics modeling"""

    def __init__(self, specs: SolarCellSpecs):
        """
        Initialize solar cell model

        Args:
            specs: SolarCellSpecs object
        """
        self.specs = specs
        self._initialize_electrical_parameters()

    def _initialize_electrical_parameters(self):
        """Initialize electrical parameters based on technology"""
        if self.specs.technology == "silicon":
            self.thermal_voltage = 0.0259  # V at room temperature
            self.dark_saturation_current = 1e-10  # A
            self.photo_current_coefficient = 0.032  # A per W/m²
        elif self.specs.technology == "multi_junction":
            self.thermal_voltage = 0.0259
            self.dark_saturation_current = 1e-12  # A (lower for multi-junction)
            self.photo_current_coefficient = 0.035  # A per W/m²
        else:
            raise ValueError(f"Unsupported technology: {self.specs.technology}")

    def calculate_iv_curve(self, conditions: EnvironmentalConditions,
                         voltage_range: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate I-V curve for given conditions

        Args:
            conditions: EnvironmentalConditions object
            voltage_range: Array of voltage points (optional)

        Returns:
            Tuple of (current_array, voltage_array)
        """
        if voltage_range is None:
            # Generate voltage range around expected maximum power point
            voc_estimate = self._estimate_open_circuit_voltage(conditions)
            voltage_range = np.linspace(0, voc_estimate * 1.2, 100)

        current_array = np.zeros_like(voltage_range)

        for i, v in enumerate(voltage_range):
            current_array[i] = self._calculate_current_at_voltage(v, conditions)

        return current_array, voltage_range

    def _estimate_open_circuit_voltage(self, conditions: EnvironmentalConditions) -> float:
        """Estimate open circuit voltage"""
        # Simplified estimation based on bandgap and temperature
        if self.specs.technology == "silicon":
            bandgap = 1.12  # eV
        else:  # multi_junction
            bandgap = 1.85  # eV (top cell determines voltage)

        voc = (bandgap - 0.3) * (1 + self.specs.temperature_coefficient *
                                (conditions.temperature - self.specs.reference_temperature))
        return max(0, voc)

    def _calculate_current_at_voltage(self, voltage: float,
                                    conditions: EnvironmentalConditions) -> float:
        """
        Calculate current at specific voltage point

        Args:
            voltage: Operating voltage
            conditions: Environmental conditions

        Returns:
            Current in Amperes
        """
        # Temperature-adjusted thermal voltage
        vt = self.thermal_voltage * (conditions.temperature / 298.0)

        # Photocurrent
        iph = (conditions.solar_flux * self.specs.area_m2 *
               self.photo_current_coefficient *
               np.cos(conditions.incident_angle) *
               self.specs.initial_efficiency *
               conditions.radiation_factor *
               conditions.thermal_factor *
               conditions.contamination_factor)

        # Diode current
        try:
            id = self.dark_saturation_current * (np.exp((voltage + iph * self.specs.series_resistance) /
                                                        (self.specs.ideality_factor * vt)) - 1)
        except OverflowError:
            id = 1e10  # Large current for numerical stability

        # Shunt current
        ish = voltage / self.specs.shunt_resistance

        # Total current
        current = iph - id - ish

        return max(0, current)  # Ensure non-negative current

    def find_maximum_power_point(self, conditions: EnvironmentalConditions) -> Dict:
        """
        Find maximum power point for given conditions

        Args:
            conditions: Environmental conditions

        Returns:
            Dictionary with MPP information
        """
        # Generate I-V curve
        current, voltage = self.calculate_iv_curve(conditions)

        # Calculate power
        power = current * voltage

        # Find maximum power point
        max_power_idx = np.argmax(power)
        max_power = power[max_power_idx]
        mpp_voltage = voltage[max_power_idx]
        mpp_current = current[max_power_idx]

        # Calculate efficiency
        input_power = (conditions.solar_flux * self.specs.area_m2 *
                      np.cos(conditions.incident_angle))
        efficiency = max_power / input_power if input_power > 0 else 0

        return {
            'max_power_W': max_power,
            'mpp_voltage_V': mpp_voltage,
            'mpp_current_A': mpp_current,
            'efficiency': efficiency,
            'fill_factor': max_power / (mpp_voltage * mpp_current) if mpp_voltage * mpp_current > 0 else 0
        }

class PowerCalculator:
    """Main power calculation class"""

    def __init__(self, specs: SolarCellSpecs):
        """
        Initialize power calculator

        Args:
            specs: Solar panel specifications
        """
        self.specs = specs
        self.cell_model = SolarCellModel(specs)

    def calculate_power_series(self, positions: np.ndarray, times: np.ndarray,
                             temperatures: np.ndarray, sun_position: np.ndarray,
                             degradation_factors: Dict) -> np.ndarray:
        """
        Calculate power output over time

        Args:
            positions: Nx3 array of satellite positions
            times: Array of time points
            temperatures: Array of cell temperatures
            sun_position: Sun position vector
            degradation_factors: Dictionary of degradation factors

        Returns:
            Array of power outputs in Watts
        """
        n_points = len(positions)
        power_output = np.zeros(n_points)

        for i in range(n_points):
            # Calculate environmental conditions
            conditions = self._calculate_environmental_conditions(
                positions[i], temperatures[i], sun_position, degradation_factors, i
            )

            # Calculate power at maximum power point
            mpp = self.cell_model.find_maximum_power_point(conditions)
            power_output[i] = mpp['max_power_W']

        return power_output

    def _calculate_environmental_conditions(self, position: np.ndarray,
                                          temperature: float, sun_position: np.ndarray,
                                          degradation_factors: Dict,
                                          time_index: int) -> EnvironmentalConditions:
        """
        Calculate environmental conditions for a specific time

        Args:
            position: Satellite position
            temperature: Cell temperature
            sun_position: Sun position vector
            degradation_factors: Dictionary of degradation factors
            time_index: Index in time series

        Returns:
            EnvironmentalConditions object
        """
        # Solar flux calculation
        solar_flux = self._calculate_solar_flux(position, sun_position)

        # Incident angle calculation
        incident_angle = self._calculate_incident_angle(position, sun_position)

        # Get degradation factors for this time point
        rad_factor = self._get_time_series_factor(degradation_factors.get('radiation', []), time_index, 1.0)
        thermal_factor = self._get_time_series_factor(degradation_factors.get('thermal', []), time_index, 1.0)
        contamination_factor = self._get_time_series_factor(degradation_factors.get('contamination', []), time_index, 1.0)

        return EnvironmentalConditions(
            solar_flux=solar_flux,
            incident_angle=incident_angle,
            temperature=temperature,
            radiation_factor=rad_factor,
            thermal_factor=thermal_factor,
            contamination_factor=contamination_factor
        )

    def _calculate_solar_flux(self, position: np.ndarray, sun_position: np.ndarray) -> float:
        """Calculate solar flux at satellite position"""
        # Simplified: use solar constant adjusted by distance from sun
        # (assumes 1 AU distance for Earth orbits)
        solar_constant = 1361.0  # W/m²

        # Check if satellite is in Earth's shadow (simplified)
        earth_radius = 6371.0  # km
        sat_distance = np.linalg.norm(position)

        if sat_distance < earth_radius:
            return 0.0  # Inside Earth

        # Simple shadow check
        sun_direction = sun_position / np.linalg.norm(sun_position)
        sat_direction = position / np.linalg.norm(position)

        if np.dot(sun_direction, sat_direction) < 0:
            return 0.0  # In Earth's shadow

        return solar_constant

    def _calculate_incident_angle(self, position: np.ndarray, sun_position: np.ndarray) -> float:
        """Calculate solar incident angle"""
        # Simplified: assume panels always face the sun optimally
        # In reality, this would depend on spacecraft attitude
        sun_vector = sun_position - position
        sun_vector = sun_vector / np.linalg.norm(sun_vector)

        # Assume panel normal is optimized for sun tracking
        panel_normal = sun_vector  # Perfect sun tracking
        panel_normal = panel_normal / np.linalg.norm(panel_normal)

        cos_angle = np.dot(panel_normal, sun_vector)
        cos_angle = np.clip(cos_angle, -1, 1)

        return np.arccos(cos_angle)

    def _get_time_series_factor(self, factor_array: np.ndarray, index: int, default: float) -> float:
        """Get factor from time series array"""
        if factor_array is None or len(factor_array) == 0:
            return default
        if index >= len(factor_array):
            return factor_array[-1]  # Use last value if index out of range
        return factor_array[index]

    def calculate_energy_yield(self, power_series: np.ndarray, time_hours: np.ndarray) -> Dict:
        """
        Calculate energy yield statistics

        Args:
            power_series: Array of power outputs in Watts
            time_hours: Array of time points in hours

        Returns:
            Dictionary with energy yield statistics
        """
        if len(power_series) == 0 or len(time_hours) == 0:
            return {}

        # Calculate energy (integrate power over time)
        if len(time_hours) > 1:
            dt_hours = np.diff(time_hours)
            # Use trapezoidal integration
            energy_wh = np.trapz(power_series, time_hours)
        else:
            # Single point - assume 1 hour duration
            energy_wh = power_series[0] * 1.0

        energy_kwh = energy_wh / 1000.0

        # Statistics
        avg_power = np.mean(power_series)
        max_power = np.max(power_series)
        min_power = np.min(power_series)
        total_duration = time_hours[-1] - time_hours[0] if len(time_hours) > 1 else 1.0

        return {
            'total_energy_kWh': energy_kwh,
            'average_power_W': avg_power,
            'maximum_power_W': max_power,
            'minimum_power_W': min_power,
            'performance_ratio': avg_power / max_power if max_power > 0 else 0,
            'capacity_factor': avg_power / (self.specs.area_m2 * self.specs.initial_efficiency * 1361) if max_power > 0 else 0,
            'simulation_duration_hours': total_duration
        }