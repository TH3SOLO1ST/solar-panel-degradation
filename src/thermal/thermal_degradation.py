"""
Thermal Degradation Model
==========================

Models thermal cycling degradation effects on solar panels.

This module implements thermal fatigue models including the Coffin-Manson
relationship and thermal stress calculations for solar panel degradation.

Classes:
    ThermalCycling: Thermal cycling degradation analysis
    ThermalStress: Thermal stress calculation
    FatigueModel: Material fatigue modeling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ThermalCycle:
    """Data structure for thermal cycle information"""
    start_temp: float  # Kelvin
    end_temp: float    # Kelvin
    temp_range: float  # Kelvin
    duration: float    # hours
    cycle_count: int   # cumulative count

class FatigueModel:
    """Material fatigue modeling for thermal cycling"""

    def __init__(self, material_type: str = "silicon"):
        """
        Initialize fatigue model

        Args:
            material_type: Material type ('silicon', 'germanium', 'composite')
        """
        self.material_type = material_type
        self._initialize_fatigue_parameters()

    def _initialize_fatigue_parameters(self):
        """Initialize fatigue parameters based on material type"""
        if self.material_type == "silicon":
            self.C_coefficient = 1e15  # Coffin-Manson coefficient
            self.beta_exponent = 4.0    # Coffin-Manson exponent
            self.fatigue_limit = 5.0    # K (minimum temperature range for fatigue)
        elif self.material_type == "germanium":
            self.C_coefficient = 5e14
            self.beta_exponent = 3.5
            self.fatigue_limit = 3.0
        else:  # composite materials
            self.C_coefficient = 2e16
            self.beta_exponent = 5.0
            self.fatigue_limit = 8.0

    def calculate_cycles_to_failure(self, temperature_range: float) -> float:
        """
        Calculate number of cycles to failure using Coffin-Manson relationship

        Args:
            temperature_range: Temperature swing in Kelvin

        Returns:
            Number of cycles to failure
        """
        if temperature_range < self.fatigue_limit:
            return float('inf')  # No fatigue damage for small cycles

        # Coffin-Manson: N_f = C × (ΔT)^(-β)
        cycles_to_failure = self.C_coefficient * (temperature_range ** (-self.beta_exponent))

        return cycles_to_failure

    def calculate_fatigue_damage(self, temperature_range: float,
                                cycle_count: int) -> float:
        """
        Calculate fatigue damage accumulation using Miner's rule

        Args:
            temperature_range: Temperature swing in Kelvin
            cycle_count: Number of cycles experienced

        Returns:
            Damage accumulation (0 to 1, where 1 = failure)
        """
        cycles_to_failure = self.calculate_cycles_to_failure(temperature_range)

        if cycles_to_failure == float('inf'):
            return 0.0

        # Miner's rule: D = Σ (n_i / N_f_i)
        damage = cycle_count / cycles_to_failure

        return min(1.0, damage)

class ThermalStress:
    """Thermal stress calculation and analysis"""

    def __init__(self, material_properties: Dict = None):
        """
        Initialize thermal stress calculator

        Args:
            material_properties: Dictionary of material properties
        """
        self.material_properties = material_properties or self._get_default_properties()

    def _get_default_properties(self) -> Dict:
        """Get default material properties"""
        return {
            'cte': 2.6e-6,      # Coefficient of thermal expansion (1/K)
            'elastic_modulus': 130e9,  # Elastic modulus (Pa)
            'poisson_ratio': 0.28,     # Poisson's ratio
            'yield_strength': 7e9,     # Yield strength (Pa)
            'thermal_conductivity': 130  # W/(m·K)
        }

    def calculate_thermal_stress(self, temperature_change: float,
                               constraint_factor: float = 1.0) -> float:
        """
        Calculate thermal stress from temperature change

        Args:
            temperature_change: Temperature change in Kelvin
            constraint_factor: Degree of constraint (0 = free, 1 = fully constrained)

        Returns:
            Thermal stress in Pascals
        """
        # Thermal stress: σ = E × α × ΔT × constraint_factor
        thermal_stress = (self.material_properties['elastic_modulus'] *
                         self.material_properties['cte'] *
                         temperature_change *
                         constraint_factor)

        return thermal_stress

    def calculate_stress_intensity(self, thermal_stress: float,
                                 flaw_size: float = 1e-3) -> float:
        """
        Calculate stress intensity factor for fracture mechanics

        Args:
            thermal_stress: Applied thermal stress in Pa
            flaw_size: Characteristic flaw size in meters

        Returns:
            Stress intensity factor in MPa√m
        """
        # Simplified: K_I = σ × √(π × a)
        stress_intensity = thermal_stress * np.sqrt(np.pi * flaw_size)

        return stress_intensity / 1e6  # Convert to MPa√m

class ThermalCycling:
    """Main thermal cycling degradation analysis class"""

    def __init__(self, material_type: str = "silicon"):
        """
        Initialize thermal cycling analysis

        Args:
            material_type: Material type for fatigue calculations
        """
        self.fatigue_model = FatigueModel(material_type)
        self.stress_calculator = ThermalStress()

    def analyze_thermal_cycles(self, temperatures: np.ndarray,
                             time_hours: np.ndarray) -> List[ThermalCycle]:
        """
        Identify and analyze thermal cycles in temperature profile

        Args:
            temperatures: Array of temperatures in Kelvin
            time_hours: Array of time points in hours

        Returns:
            List of ThermalCycle objects
        """
        if len(temperatures) < 2:
            return []

        cycles = []
        cycle_count = 0

        # Simple peak-valley cycle counting algorithm
        i = 0
        while i < len(temperatures) - 1:
            # Find local maximum or minimum
            if i == 0:
                direction = 1 if temperatures[1] > temperatures[0] else -1
            else:
                direction = 1 if temperatures[i+1] > temperatures[i] else -1

            # Find next extremum
            start_idx = i
            start_temp = temperatures[i]

            # Continue until direction changes
            while i < len(temperatures) - 1 and (
                (direction > 0 and temperatures[i+1] >= temperatures[i]) or
                (direction < 0 and temperatures[i+1] <= temperatures[i])
            ):
                i += 1

            if i < len(temperatures) - 1:
                # Found a cycle
                end_temp = temperatures[i]
                temp_range = abs(end_temp - start_temp)
                duration = time_hours[i] - time_hours[start_idx]

                if temp_range > 5.0:  # Minimum temperature range for significant cycles
                    cycle_count += 1
                    cycles.append(ThermalCycle(
                        start_temp=start_temp,
                        end_temp=end_temp,
                        temp_range=temp_range,
                        duration=duration,
                        cycle_count=cycle_count
                    ))

                i += 1
            else:
                break

        return cycles

    def calculate_thermal_degradation(self, temperatures: np.ndarray,
                                    time_hours: np.ndarray) -> Dict:
        """
        Calculate overall thermal degradation

        Args:
            temperatures: Array of temperatures in Kelvin
            time_hours: Array of time points in hours

        Returns:
            Dictionary with degradation results
        """
        # Analyze thermal cycles
        cycles = self.analyze_thermal_cycles(temperatures, time_hours)

        if not cycles:
            return {
                'total_degradation': 0.0,
                'fatigue_damage': 0.0,
                'thermal_stress_damage': 0.0,
                'cycle_count': 0,
                'max_temperature_range': 0.0
            }

        # Calculate fatigue damage for each cycle
        total_fatigue_damage = 0.0
        max_stress = 0.0

        for cycle in cycles:
            # Fatigue damage for this cycle
            cycle_damage = self.fatigue_model.calculate_fatigue_damage(
                cycle.temp_range, 1
            )
            total_fatigue_damage += cycle_damage

            # Maximum thermal stress
            stress = self.stress_calculator.calculate_thermal_stress(
                cycle.temp_range / 2  # Approximate as half the temperature range
            )
            max_stress = max(max_stress, stress)

        # Total thermal degradation (simplified model)
        # Combine fatigue and stress effects
        fatigue_degradation = min(1.0, total_fatigue_damage)
        stress_degradation = min(1.0, max_stress / self.stress_calculator.material_properties['yield_strength'])

        # Combined degradation (conservative approach)
        total_degradation = min(1.0, fatigue_degradation + stress_degradation)

        return {
            'total_degradation': total_degradation,
            'fatigue_damage': fatigue_degradation,
            'thermal_stress_damage': stress_degradation,
            'cycle_count': len(cycles),
            'max_temperature_range': max(c.temp_range for c in cycles) if cycles else 0.0,
            'max_thermal_stress': max_stress,
            'cycles': cycles
        }

    def get_temperature_coefficient_effect(self, temperatures: np.ndarray,
                                         reference_temp: float = 298.0) -> np.ndarray:
        """
        Calculate temperature coefficient effects on efficiency

        Args:
            temperatures: Array of temperatures in Kelvin
            reference_temp: Reference temperature for coefficient calculation

        Returns:
            Array of efficiency factors due to temperature
        """
        # Temperature coefficient for silicon solar cells: -0.004 to -0.005 per K
        temp_coefficient = -0.0045  # per K

        # Efficiency factor: η(T) = η₀ × [1 + α_T × (T - T_ref)]
        temp_diff = temperatures - reference_temp
        efficiency_factor = 1 + temp_coefficient * temp_diff

        # Ensure reasonable bounds
        efficiency_factor = np.clip(efficiency_factor, 0.3, 1.2)

        return efficiency_factor