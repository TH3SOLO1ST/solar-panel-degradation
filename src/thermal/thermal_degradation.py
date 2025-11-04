"""
Thermal Degradation Module

This module models thermal cycling degradation effects on solar panels using
Coffin-Manson relationships and other fatigue models. It handles solder joint
fatigue, delamination, and temperature-dependent efficiency degradation.

References:
- "Thermal Fatigue of Solder Joints" - IPC standards
- "Coffin-Manson Fatigue Analysis" - ASTM standards
- "Solar Panel Thermal Cycling Reliability" - NASA/ESA
- "Failure Mechanics of Electronic Materials" - Pecht
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from .thermal_analysis import ThermalAnalysis, ThermalState, ThermalCycle


@dataclass
class ThermalFatigueCoefficients:
    """Thermal fatigue coefficients for different materials"""
    # Coffin-Manson coefficients: N_f = C × (ΔT)^(-β)
    C_coefficient: float        # Fatigue coefficient
    beta_exponent: float        # Fatigue exponent
    activation_energy: float    # Activation energy (eV)
    reference_temp: float       # Reference temperature (K)


@dataclass
class ThermalDegradationState:
    """Current state of thermal degradation"""
    fatigue_damage_fraction: float     # Accumulated fatigue damage (0-1)
    efficiency_degradation_percent: float  # Efficiency loss (%)
    series_resistance_increase_percent: float  # Series resistance increase (%)
    delamination_risk: float          # Delamination risk factor (0-1)
    solder_joint_damage: float        # Solder joint damage (0-1)
    cycle_count: int                  # Total thermal cycles
    max_temp_swing_experienced: float # Maximum temperature swing experienced (K)


@dataclass
class TemperatureDependentEfficiency:
    """Temperature-dependent solar cell efficiency parameters"""
    temp_coefficient: float      # Temperature coefficient (%/K)
    reference_temp: float        # Reference temperature (K)
    reference_efficiency: float  # Efficiency at reference temperature (0-1)
    bandgap_temp_coeff: float    # Bandgap temperature coefficient (eV/K)


class ThermalDegradation:
    """
    Comprehensive thermal degradation model for solar panels.

    Features:
    - Coffin-Manson fatigue analysis
    - Solder joint fatigue modeling
    - Delamination risk assessment
    - Temperature-dependent efficiency
    - Cumulative damage accumulation
    - Material-specific fatigue parameters
    """

    # Physical constants
    k_B = 8.617333e-5  # Boltzmann constant (eV/K)

    # Default thermal fatigue coefficients for common materials
    DEFAULT_FATIGUE_COEFFICIENTS = {
        'solder_joints': ThermalFatigueCoefficients(
            C_coefficient=1e15,      # Typical for Sn-Pb solder
            beta_exponent=2.0,       # Low cycle fatigue
            activation_energy=0.6,   # eV
            reference_temp=298.15    # K (25°C)
        ),
        'silicon_cells': ThermalFatigueCoefficients(
            C_coefficient=1e20,      # Silicon is more resistant
            beta_exponent=1.8,       # Slightly better than solder
            activation_energy=0.8,   # eV
            reference_temp=298.15    # K
        ),
        'interconnects': ThermalFatigueCoefficients(
            C_coefficient=5e14,      # Most vulnerable
            beta_exponent=2.2,       # High fatigue sensitivity
            activation_energy=0.5,   # eV
            reference_temp=298.15    # K
        )
    }

    # Default temperature-dependent efficiency parameters
    DEFAULT_EFFICIENCY_PARAMS = TemperatureDependentEfficiency(
        temp_coefficient=-0.004,    # -0.4%/K for silicon
        reference_temp=298.15,      # K (25°C)
        reference_efficiency=0.20,  # 20% efficiency at reference
        bandgap_temp_coeff=-4.73e-4 # eV/K for silicon
    )

    # Temperature thresholds for degradation
    MIN_TEMP_K = 173.15     # -100°C minimum operating temperature
    MAX_TEMP_K = 423.15     # 150°C maximum operating temperature
    STRESS_TEMP_K = 373.15  # 100°C stress temperature

    def __init__(self, cell_technology: str = "silicon",
                 custom_fatigue_coeffs: Optional[Dict[str, ThermalFatigueCoefficients]] = None,
                 custom_efficiency_params: Optional[TemperatureDependentEfficiency] = None):
        """
        Initialize thermal degradation model

        Args:
            cell_technology: Solar cell technology type
            custom_fatigue_coeffs: Custom fatigue coefficients
            custom_efficiency_params: Custom efficiency parameters
        """
        self.cell_technology = cell_technology

        # Set fatigue coefficients
        if custom_fatigue_coeffs:
            self.fatigue_coeffs = custom_fatigue_coeffs
        else:
            self.fatigue_coeffs = self.DEFAULT_FATIGUE_COEFFICIENTS.copy()

        # Set efficiency parameters
        if custom_efficiency_params:
            self.efficiency_params = custom_efficiency_params
        else:
            self.efficiency_params = self.DEFAULT_EFFICIENCY_PARAMS

        # Initialize degradation state
        self.degradation_state = ThermalDegradationState(
            fatigue_damage_fraction=0.0,
            efficiency_degradation_percent=0.0,
            series_resistance_increase_percent=0.0,
            delamination_risk=0.0,
            solder_joint_damage=0.0,
            cycle_count=0,
            max_temp_swing_experienced=0.0
        )

        # Track cycle history
        self.cycle_history: List[ThermalCycle] = []

    def calculate_coffin_manson_cycles(self, temp_swing_K: float,
                                     max_temp_K: float,
                                     material: str = "solder_joints") -> float:
        """
        Calculate number of cycles to failure using Coffin-Manson relationship

        Args:
            temp_swing_K: Temperature swing ΔT (K)
            max_temp_K: Maximum temperature in cycle (K)
            material: Material type for fatigue coefficients

        Returns:
            Number of cycles to failure
        """
        if material not in self.fatigue_coeffs:
            material = "solder_joints"

        coeffs = self.fatigue_coeffs[material]

        # Temperature correction factor (Arrhenius)
        temp_correction = np.exp(
            -coeffs.activation_energy / self.k_B *
            (1/max_temp_K - 1/coeffs.reference_temp)
        )

        # Coffin-Manson relationship: N_f = C × (ΔT)^(-β)
        if temp_swing_K <= 0:
            return float('inf')

        cycles_to_failure = (coeffs.C_coefficient *
                           (temp_swing_K ** (-coeffs.beta_exponent)) *
                           temp_correction)

        return cycles_to_failure

    def calculate_fatigue_damage_contribution(self, cycle: ThermalCycle,
                                            material: str = "solder_joints") -> float:
        """
        Calculate fatigue damage contribution from a single cycle

        Args:
            cycle: Thermal cycle information
            material: Material type

        Returns:
            Damage fraction (0-1, accumulates to 1 at failure)
        """
        cycles_to_failure = self.calculate_coffin_manson_cycles(
            cycle.temperature_swing,
            cycle.max_temperature,
            material
        )

        if cycles_to_failure == float('inf'):
            return 0.0

        # Miner's rule: Damage = 1/N_f for each cycle
        damage_contribution = 1.0 / cycles_to_failure

        return min(1.0, damage_contribution)

    def calculate_delamination_risk(self, cycles: List[ThermalCycle]) -> float:
        """
        Calculate delamination risk from thermal cycling

        Args:
            cycles: List of thermal cycles

        Returns:
            Delamination risk factor (0-1)
        """
        if not cycles:
            return 0.0

        # Factors that increase delamination risk
        max_temp_swing = max(cycle.temperature_swing for cycle in cycles)
        avg_heating_rate = np.mean([cycle.heating_rate for cycle in cycles if cycle.heating_rate > 0])
        avg_cooling_rate = np.mean([cycle.cooling_rate for cycle in cycles if cycle.cooling_rate > 0])
        total_cycles = len(cycles)

        # High temperature swings increase risk
        swing_risk = min(1.0, max_temp_swing / 200.0)  # Normalize by 200K swing

        # Rapid temperature changes increase risk
        heating_risk = min(1.0, avg_heating_rate / 100.0)  # Normalize by 100K/hour
        cooling_risk = min(1.0, avg_cooling_rate / 100.0)

        # Number of cycles increases risk (logarithmic scale)
        cycle_risk = min(1.0, np.log10(total_cycles + 1) / 6.0)  # Normalize by 1e6 cycles

        # Combined risk (weighted average)
        delamination_risk = (0.4 * swing_risk +
                           0.25 * heating_risk +
                           0.25 * cooling_risk +
                           0.1 * cycle_risk)

        return min(1.0, delamination_risk)

    def calculate_temperature_dependent_efficiency(self, temperature_K: float) -> float:
        """
        Calculate solar cell efficiency at given temperature

        Args:
            temperature_K: Cell temperature in Kelvin

        Returns:
            Efficiency factor relative to reference (0-1)
        """
        # Linear temperature coefficient model
        delta_T = temperature_K - self.efficiency_params.reference_temp

        # Efficiency degradation: η(T) = η_ref × [1 + α_T × (T - T_ref)]
        efficiency_factor = 1.0 + (self.efficiency_params.temp_coefficient *
                                  delta_T / 100.0)  # Convert %/K to fraction

        # Account for bandgap changes at extreme temperatures
        if temperature_K < 200 or temperature_K > 400:
            # Additional degradation at temperature extremes
            bandgap_factor = 1.0 - 0.1 * abs(temperature_K - 298.15) / 100.0
            efficiency_factor *= bandgap_factor

        return max(0.1, min(1.0, efficiency_factor))

    def update_degradation_state(self, new_cycles: List[ThermalCycle],
                               current_temperature_K: float) -> ThermalDegradationState:
        """
        Update degradation state with new thermal cycles

        Args:
            new_cycles: List of new thermal cycles to process
            current_temperature_K: Current operating temperature

        Returns:
            Updated degradation state
        """
        if not new_cycles:
            return self.degradation_state

        # Update cycle count
        self.degradation_state.cycle_count += len(new_cycles)

        # Track maximum temperature swing
        max_swing = max(cycle.temperature_swing for cycle in new_cycles)
        self.degradation_state.max_temp_swing_experienced = max(
            self.degradation_state.max_temp_swing_experienced, max_swing
        )

        # Calculate fatigue damage for each material
        total_fatigue_damage = 0.0
        solder_damage = 0.0

        for cycle in new_cycles:
            # Solder joint damage (most critical)
            solder_damage += self.calculate_fatigue_damage_contribution(cycle, "solder_joints")

            # Silicon cell damage
            silicon_damage = self.calculate_fatigue_damage_contribution(cycle, "silicon_cells")
            total_fatigue_damage += silicon_damage

            # Interconnect damage
            interconnect_damage = self.calculate_fatigue_damage_contribution(cycle, "interconnects")
            total_fatigue_damage += interconnect_damage

        # Update damage states
        self.degradation_state.solder_joint_damage = min(1.0, self.degradation_state.solder_joint_damage + solder_damage)
        self.degradation_state.fatigue_damage_fraction = min(1.0, self.degradation_state.fatigue_damage_fraction + total_fatigue_damage)

        # Update delamination risk
        self.cycle_history.extend(new_cycles)
        self.degradation_state.delamination_risk = self.calculate_delamination_risk(self.cycle_history)

        # Calculate efficiency degradation from fatigue
        fatigue_efficiency_loss = self.degradation_state.fatigue_damage_fraction * 5.0  # Max 5% loss from fatigue

        # Temperature-dependent efficiency factor
        temp_efficiency_factor = self.calculate_temperature_dependent_efficiency(current_temperature_K)
        temp_efficiency_loss = (1.0 - temp_efficiency_factor) * 100.0  # Convert to percentage

        # Total efficiency degradation
        self.degradation_state.efficiency_degradation_percent = fatigue_efficiency_loss + max(0.0, temp_efficiency_loss)

        # Calculate series resistance increase
        self.degradation_state.series_resistance_increase_percent = (
            self.degradation_state.solder_joint_damage * 50.0 +  # Up to 50% increase from solder damage
            self.degradation_state.fatigue_damage_fraction * 20.0  # Additional from general fatigue
        )

        # Store updated cycles
        self.cycle_history.extend(new_cycles)

        return self.degradation_state

    def predict_lifetime_thermal_degradation(self, thermal_analysis: ThermalAnalysis,
                                           start_time: datetime,
                                           duration_years: float,
                                           time_step_hours: float = 1.0) -> List[ThermalDegradationState]:
        """
        Predict thermal degradation over mission lifetime

        Args:
            thermal_analysis: Thermal analysis instance
            start_time: Mission start time
            duration_years: Mission duration in years
            time_step_hours: Time step for analysis

        Returns:
            List of degradation states over time
        """
        degradation_timeline = []
        current_time = start_time
        end_time = start_time + timedelta(days=duration_years * 365.25)

        # Reset degradation state
        self.degradation_state = ThermalDegradationState(
            fatigue_damage_fraction=0.0,
            efficiency_degradation_percent=0.0,
            series_resistance_increase_percent=0.0,
            delamination_risk=0.0,
            solder_joint_damage=0.0,
            cycle_count=0,
            max_temp_swing_experienced=0.0
        )
        self.cycle_history = []

        # Panel normal (assuming Sun-pointing panel)
        panel_normal = np.array([0, 0, 1])

        # Process in batches to analyze thermal cycles
        batch_duration_hours = min(168, time_step_hours * 100)  # Weekly or 100 steps

        while current_time <= end_time:
            batch_end = min(current_time + timedelta(hours=batch_duration_hours), end_time)

            # Solve thermal analysis for this batch
            batch_states = thermal_analysis.solve_thermal_equation(
                initial_temp_K=298.15,  # Start at room temperature
                time_span_hours=(batch_end - current_time).total_seconds() / 3600.0,
                time_step_hours=time_step_hours,
                panel_normal=panel_normal,
                start_time=current_time
            )

            # Analyze thermal cycles in this batch
            new_cycles = thermal_analysis.analyze_thermal_cycles(batch_states)

            # Update degradation state
            if batch_states:
                current_temp = batch_states[-1].temperature  # Use last temperature
                self.update_degradation_state(new_cycles, current_temp)

            # Store current state (deep copy)
            current_state = ThermalDegradationState(
                fatigue_damage_fraction=self.degradation_state.fatigue_damage_fraction,
                efficiency_degradation_percent=self.degradation_state.efficiency_degradation_percent,
                series_resistance_increase_percent=self.degradation_state.series_resistance_increase_percent,
                delamination_risk=self.degradation_state.delamination_risk,
                solder_joint_damage=self.degradation_state.solder_joint_damage,
                cycle_count=self.degradation_state.cycle_count,
                max_temp_swing_experienced=self.degradation_state.max_temp_swing_experienced
            )
            degradation_timeline.append(current_state)

            current_time = batch_end

        return degradation_timeline

    def get_thermal_mitigation_recommendations(self) -> Dict[str, Union[str, List[str]]]:
        """
        Get recommendations for thermal degradation mitigation

        Returns:
            Dictionary with mitigation recommendations
        """
        recommendations = {
            'risk_level': 'low',
            'primary_concerns': [],
            'mitigation_strategies': []
        }

        # Assess risk level
        if self.degradation_state.fatigue_damage_fraction > 0.5:
            recommendations['risk_level'] = 'high'
        elif self.degradation_state.fatigue_damage_fraction > 0.2:
            recommendations['risk_level'] = 'medium'

        # Identify primary concerns
        if self.degradation_state.solder_joint_damage > 0.3:
            recommendations['primary_concerns'].append('Solder joint fatigue')

        if self.degradation_state.delamination_risk > 0.4:
            recommendations['primary_concerns'].append('Delamination risk')

        if self.degradation_state.max_temp_swing_experienced > 150:
            recommendations['primary_concerns'].append('Extreme temperature swings')

        if self.degradation_state.efficiency_degradation_percent > 5:
            recommendations['primary_concerns'].append('Significant efficiency loss')

        # Generate mitigation strategies
        if self.degradation_state.solder_joint_damage > 0.2:
            recommendations['mitigation_strategies'].extend([
                'Use lead-free solder with improved fatigue resistance',
                'Implement compliant interconnects',
                'Add strain relief mechanisms'
            ])

        if self.degradation_state.delamination_risk > 0.3:
            recommendations['mitigation_strategies'].extend([
                'Use advanced encapsulation materials',
                'Improve adhesive bonding processes',
                'Add moisture barrier layers'
            ])

        if self.degradation_state.max_temp_swing_experienced > 100:
            recommendations['mitigation_strategies'].extend([
                'Implement thermal mass or phase change materials',
                'Use active thermal control systems',
                'Optimize orbital attitude to reduce temperature extremes'
            ])

        if not recommendations['mitigation_strategies']:
            recommendations['mitigation_strategies'] = [
                'Continue monitoring thermal cycles',
                'Regular inspection during maintenance periods'
            ]

        return recommendations

    def get_degradation_summary(self) -> Dict:
        """
        Get comprehensive thermal degradation summary

        Returns:
            Dictionary with degradation analysis
        """
        return {
            'cell_technology': self.cell_technology,
            'degradation_state': {
                'fatigue_damage_fraction': self.degradation_state.fatigue_damage_fraction,
                'fatigue_damage_percent': self.degradation_state.fatigue_damage_fraction * 100.0,
                'efficiency_degradation_percent': self.degradation_state.efficiency_degradation_percent,
                'series_resistance_increase_percent': self.degradation_state.series_resistance_increase_percent,
                'delamination_risk': self.degradation_state.delamination_risk,
                'solder_joint_damage': self.degradation_state.solder_joint_damage,
                'solder_joint_damage_percent': self.degradation_state.solder_joint_damage * 100.0,
                'cycle_count': self.degradation_state.cycle_count,
                'max_temp_swing_experienced_K': self.degradation_state.max_temp_swing_experienced
            },
            'fatigue_coefficients': {
                material: {
                    'C_coefficient': coeffs.C_coefficient,
                    'beta_exponent': coeffs.beta_exponent,
                    'activation_energy': coeffs.activation_energy
                }
                for material, coeffs in self.fatigue_coeffs.items()
            },
            'efficiency_parameters': {
                'temp_coefficient_per_K': self.efficiency_params.temp_coefficient,
                'reference_temp_K': self.efficiency_params.reference_temp,
                'reference_efficiency': self.efficiency_params.reference_efficiency
            },
            'mitigation_recommendations': self.get_thermal_mitigation_recommendations()
        }