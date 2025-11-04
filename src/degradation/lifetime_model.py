"""
Lifetime Degradation Model Module

This module combines all degradation mechanisms (radiation, thermal, environmental)
into a comprehensive lifetime power degradation model. It provides end-to-end
simulation capability from launch to end-of-life.

References:
- "Space Solar Cell Degradation Analysis" - NASA/ESA Joint Study
- "Combined Radiation and Thermal Effects on Solar Cells" - IEEE
- "Lifetime Prediction of Space Solar Arrays" - AIAA
- "Multi-Mechanism Degradation Modeling" - PV Reliability Lab
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from ..orbital.orbit_propagator import OrbitPropagator
from ..orbital.eclipse_calculator import EclipseCalculator
from ..radiation.radiation_environment import RadiationEnvironment
from ..radiation.damage_model import RadiationDamageModel
from ..thermal.thermal_analysis import ThermalAnalysis
from ..thermal.thermal_degradation import ThermalDegradation
from .power_calculator import PowerCalculator, PowerOutput


@dataclass
class DegradationMechanism:
    """Individual degradation mechanism data"""
    name: str                     # Mechanism name
    contribution_percent: float    # Contribution to total degradation (%)
    rate_per_year: float          # Annual degradation rate (%/year)
    cumulative_damage: float      # Cumulative damage (0-1)
    activation: bool              # Whether mechanism is active


@dataclass
class LifetimeState:
    """Complete degradation state at a given time"""
    time: datetime
    mission_time_hours: float     # Time since launch (hours)
    initial_power_watts: float    # Initial power output (W)
    current_power_watts: float   # Current power output (W)
    power_degradation_percent: float  # Total power degradation (%)
    efficiency_factor: float      # Relative efficiency (0-1)

    # Individual mechanism contributions
    radiation_damage: float       # Radiation degradation factor (0-1)
    thermal_damage: float         # Thermal degradation factor (0-1)
    contamination_damage: float   # Contamination degradation factor (0-1)
    aging_damage: float          # Normal aging degradation factor (0-1)

    # Environmental exposure
    total_radiation_dose_rads: float
    total_thermal_cycles: int
    max_temperature_K: float
    min_temperature_K: float
    total_eclipse_hours: float

    # Degradation mechanisms
    mechanisms: List[DegradationMechanism]


@dataclass
class LifetimePrediction:
    """Complete lifetime prediction results"""
    initial_power_watts: float
    final_power_watts: float
    total_degradation_percent: float
    mission_duration_hours: float
    degradation_rate_percent_per_year: float

    # Power statistics
    average_power_watts: float
    minimum_power_watts: float
    maximum_power_watts: float
    total_energy_Wh: float

    # End-of-life predictions
    eol_efficiency_percent: float
    years_to_80_percent: Optional[float]  # Years to reach 80% of initial power
    years_to_50_percent: Optional[float]  # Years to reach 50% of initial power

    # Mechanism breakdown
    mechanism_contributions: Dict[str, float]


class LifetimeDegradationModel:
    """
    Comprehensive lifetime degradation model for solar panels.

    Features:
    - Multi-mechanism degradation integration
    - Time-dependent damage accumulation
    - Environmental condition tracking
    - End-of-life prediction
    - Mechanism contribution analysis
    - Mission scenario optimization
    """

    # Standard degradation rates for different mechanisms
    STANDARD_DEGRADATION_RATES = {
        'radiation': {
            'LEO': 2.5,      # %/year for Low Earth Orbit
            'MEO': 4.0,      # %/year for Medium Earth Orbit
            'GEO': 3.0,      # %/year for Geostationary Orbit
            'SSO': 2.0       # %/year for Sun-Synchronous Orbit
        },
        'thermal': {
            'low_cycles': 0.5,    # %/year for low thermal cycling
            'moderate_cycles': 1.0,  # %/year for moderate cycling
            'high_cycles': 2.0    # %/year for severe cycling
        },
        'contamination': {
            'clean': 0.2,     # %/year for clean environment
            'moderate': 0.5,   # %/year for moderate contamination
            'dirty': 1.0      # %/year for dirty environment
        },
        'aging': {
            'silicon': 0.5,   # %/year normal aging for silicon
            'gaas': 0.3,      # %/year normal aging for GaAs
            'multijunction': 0.4  # %/year normal aging for multi-junction
        }
    }

    def __init__(self, orbit_propagator: OrbitPropagator,
                 radiation_environment: RadiationEnvironment,
                 radiation_damage_model: RadiationDamageModel,
                 thermal_analysis: ThermalAnalysis,
                 thermal_degradation: ThermalDegradation,
                 power_calculator: PowerCalculator,
                 eclipse_calculator: EclipseCalculator):
        """
        Initialize lifetime degradation model

        Args:
            orbit_propagator: Orbital propagator instance
            radiation_environment: Radiation environment model
            radiation_damage_model: Radiation damage model
            thermal_analysis: Thermal analysis module
            thermal_degradation: Thermal degradation model
            power_calculator: Power calculator
            eclipse_calculator: Eclipse calculator
        """
        self.orbit_propagator = orbit_propagator
        self.radiation_env = radiation_environment
        self.radiation_damage = radiation_damage_model
        self.thermal_analysis = thermal_analysis
        self.thermal_degradation = thermal_degradation
        self.power_calc = power_calculator
        self.eclipse_calc = eclipse_calculator

        # Initialize degradation mechanisms
        self.degradation_mechanisms = self._initialize_mechanisms()

        # Mission state tracking
        self.lifetime_history: List[LifetimeState] = []
        self.launch_time: Optional[datetime] = None

    def _initialize_mechanisms(self) -> List[DegradationMechanism]:
        """Initialize degradation mechanisms"""
        return [
            DegradationMechanism(
                name="radiation_damage",
                contribution_percent=0.0,
                rate_per_year=0.0,
                cumulative_damage=0.0,
                activation=True
            ),
            DegradationMechanism(
                name="thermal_cycling",
                contribution_percent=0.0,
                rate_per_year=0.0,
                cumulative_damage=0.0,
                activation=True
            ),
            DegradationMechanism(
                name="surface_contamination",
                contribution_percent=0.0,
                rate_per_year=0.0,
                cumulative_damage=0.0,
                activation=True
            ),
            DegradationMechanism(
                name="normal_aging",
                contribution_percent=0.0,
                rate_per_year=0.0,
                cumulative_damage=0.0,
                activation=True
            )
        ]

    def calculate_combined_degradation_factor(self, time: datetime,
                                            panel_normal: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Calculate combined degradation factor from all mechanisms

        Args:
            time: Current time
            panel_normal: Solar panel normal vector

        Returns:
            Tuple of (combined_factor, individual_factors)
        """
        # Get orbital state
        state = self.orbit_propagator.propagate(time)

        # Radiation damage factor
        radiation_state = self.radiation_damage.damage_state
        radiation_factor = radiation_state.power_efficiency_factor

        # Thermal damage factor
        thermal_state = self.thermal_degradation.degradation_state
        temp_efficiency = self.thermal_degradation.calculate_temperature_dependent_efficiency(
            thermal_analysis.current_temperature if hasattr(thermal_analysis, 'current_temperature') else 298.15
        )
        thermal_factor = temp_efficiency * (1.0 - thermal_state.efficiency_degradation_percent / 100.0)

        # Contamination damage (simplified model)
        mission_hours = (time - self.launch_time).total_seconds() / 3600.0 if self.launch_time else 0
        contamination_rate = 0.001  # 0.1% per 1000 hours
        contamination_factor = max(0.7, 1.0 - contamination_rate * mission_hours / 1000.0)

        # Normal aging (linear model)
        aging_rate = 0.0005  # 0.05% per 1000 hours
        aging_factor = max(0.8, 1.0 - aging_rate * mission_hours / 1000.0)

        # Combined factor (multiplicative model)
        combined_factor = (radiation_factor * thermal_factor *
                         contamination_factor * aging_factor)

        individual_factors = {
            'radiation': radiation_factor,
            'thermal': thermal_factor,
            'contamination': contamination_factor,
            'aging': aging_factor
        }

        return combined_factor, individual_factors

    def simulate_lifetime(self, launch_time: datetime,
                         duration_years: float,
                         time_step_hours: float = 24.0,
                         panel_normal: np.ndarray = None) -> LifetimePrediction:
        """
        Simulate complete mission lifetime degradation

        Args:
            launch_time: Mission launch time
            duration_years: Mission duration in years
            time_step_hours: Time step for simulation
            panel_normal: Solar panel normal vector

        Returns:
            Complete lifetime prediction results
        """
        if panel_normal is None:
            panel_normal = np.array([0, 0, 1])  # Default: Sun-pointing

        self.launch_time = launch_time
        current_time = launch_time
        end_time = launch_time + timedelta(days=duration_years * 365.25)

        # Reset degradation models
        self._reset_degradation_models()

        # Get initial power
        initial_power_output = self._calculate_power_at_time(launch_time, panel_normal)
        initial_power = initial_power_output.power_watts

        # Initialize lifetime tracking
        power_history = []
        degradation_history = []
        max_temp = float('-inf')
        min_temp = float('inf')
        total_eclipse_hours = 0.0
        total_thermal_cycles = 0

        # Main simulation loop
        while current_time <= end_time:
            # Calculate orbital parameters
            state = self.orbit_propagator.propagate(current_time)

            # Calculate radiation exposure
            trapped_fluxes = self.radiation_env.calculate_trapped_particle_flux(state.position, current_time)
            spe_fluxes = self.radiation_env.calculate_solar_particle_event_flux(state.position, current_time)
            gcr_fluxes = self.radiation_env.calculate_gcr_flux(state.position, current_time)
            all_fluxes = trapped_fluxes + spe_fluxes + gcr_fluxes

            radiation_dose = self.radiation_env.calculate_radiation_dose(
                all_fluxes, 0.1, time_step_hours  # 1mm shielding
            )

            # Update radiation damage
            self.radiation_damage.update_damage_state(
                radiation_dose, all_fluxes, time_step_hours
            )

            # Calculate thermal conditions
            eclipse_event = self.eclipse_calc.calculate_eclipse_at_time(current_time)
            irradiance = self.thermal_analysis.calculate_solar_heat_flux(current_time, panel_normal)

            # Solve thermal equation for this time step
            thermal_states = self.thermal_analysis.solve_thermal_equation(
                initial_temp_K=298.15,
                time_span_hours=time_step_hours,
                time_step_hours=min(1.0, time_step_hours),
                panel_normal=panel_normal,
                start_time=current_time
            )

            if thermal_states:
                current_temp = thermal_states[-1].temperature
                max_temp = max(max_temp, current_temp)
                min_temp = min(min_temp, current_temp)

                # Analyze thermal cycles
                thermal_cycles = self.thermal_analysis.analyze_thermal_cycles(thermal_states)
                if thermal_cycles:
                    self.thermal_degradation.update_degradation_state(thermal_cycles, current_temp)
                    total_thermal_cycles += len(thermal_cycles)

            # Track eclipse time
            if eclipse_event.eclipse_type != "none":
                total_eclipse_hours += time_step_hours

            # Calculate current power with degradation
            combined_degradation, individual_factors = self.calculate_combined_degradation_factor(
                current_time, panel_normal
            )

            effective_irradiance = self.SOLAR_CONSTANT  # Simplified
            current_power_output = self.power_calc.calculate_power_output(
                current_time,
                effective_irradiance,
                current_temp if thermal_states else 298.15,
                combined_degradation,
                eclipse_event.max_eclipse_fraction if eclipse_event.eclipse_type != "none" else 0.0
            )

            # Store results
            power_history.append(current_power_output)
            degradation_history.append(combined_degradation)

            # Create lifetime state
            mission_time_hours = (current_time - launch_time).total_seconds() / 3600.0
            lifetime_state = LifetimeState(
                time=current_time,
                mission_time_hours=mission_time_hours,
                initial_power_watts=initial_power,
                current_power_watts=current_power_output.power_watts,
                power_degradation_percent=(1.0 - current_power_output.power_watts / initial_power) * 100.0,
                efficiency_factor=combined_degradation,
                radiation_damage=individual_factors['radiation'],
                thermal_damage=individual_factors['thermal'],
                contamination_damage=individual_factors['contamination'],
                aging_damage=individual_factors['aging'],
                total_radiation_dose_rads=self.radiation_damage.damage_state.ionizing_dose_rads,
                total_thermal_cycles=total_thermal_cycles,
                max_temperature_K=max_temp,
                min_temperature_K=min_temp,
                total_eclipse_hours=total_eclipse_hours,
                mechanisms=self._update_mechanism_contributions(individual_factors, combined_degradation)
            )

            self.lifetime_history.append(lifetime_state)
            current_time += timedelta(hours=time_step_hours)

        # Analyze results
        return self._analyze_lifetime_results(power_history, duration_years)

    def _reset_degradation_models(self):
        """Reset all degradation models to initial state"""
        # Reset radiation damage
        self.radiation_damage.damage_state = type(self.radiation_damage.damage_state)(
            power_efficiency_factor=1.0,
            voltage_degradation=0.0,
            current_degradation=0.0,
            fill_factor_degradation=0.0,
            series_resistance_increase=0.0,
            ddd_accumulated=0.0,
            ionizing_dose_rads=0.0,
            surface_transmission_loss=0.0
        )

        # Reset thermal degradation
        self.thermal_degradation.degradation_state = type(self.thermal_degradation.degradation_state)(
            fatigue_damage_fraction=0.0,
            efficiency_degradation_percent=0.0,
            series_resistance_increase_percent=0.0,
            delamination_risk=0.0,
            solder_joint_damage=0.0,
            cycle_count=0,
            max_temp_swing_experienced=0.0
        )

        # Clear history
        self.lifetime_history = []

    def _calculate_power_at_time(self, time: datetime, panel_normal: np.ndarray) -> PowerOutput:
        """Calculate power output at specific time"""
        # Simplified calculation for initial power
        return self.power_calc.calculate_power_output(
            time,
            self.SOLAR_CONSTANT,
            298.15,  # Standard temperature
            1.0,     # No degradation
            0.0      # No eclipse
        )

    def _update_mechanism_contributions(self, individual_factors: Dict[str, float],
                                      combined_factor: float) -> List[DegradationMechanism]:
        """Update mechanism contributions based on current degradation"""
        mechanisms = self.degradation_mechanisms.copy()

        # Calculate contributions (simplified)
        total_loss = 1.0 - combined_factor
        if total_loss > 0:
            for mechanism in mechanisms:
                if mechanism.name == "radiation_damage":
                    factor = individual_factors['radiation']
                elif mechanism.name == "thermal_cycling":
                    factor = individual_factors['thermal']
                elif mechanism.name == "surface_contamination":
                    factor = individual_factors['contamination']
                elif mechanism.name == "normal_aging":
                    factor = individual_factors['aging']
                else:
                    factor = 1.0

                loss_contribution = (1.0 - factor)
                mechanism.contribution_percent = (loss_contribution / total_loss) * 100.0
                mechanism.cumulative_damage = 1.0 - factor

        return mechanisms

    def _analyze_lifetime_results(self, power_history: List[PowerOutput],
                                duration_years: float) -> LifetimePrediction:
        """Analyze simulation results and create prediction summary"""
        if not power_history:
            raise ValueError("No power data available for analysis")

        powers = [p.power_watts for p in power_history]
        initial_power = powers[0]
        final_power = powers[-1]
        total_degradation = (1.0 - final_power / initial_power) * 100.0

        # Calculate statistics
        avg_power = np.mean(powers)
        min_power = min(powers)
        max_power = max(powers)
        total_energy = self.power_calc.calculate_energy_output(power_history)

        # Calculate degradation rate
        degradation_rate = total_degradation / duration_years  # %/year

        # Predict end-of-life metrics
        final_efficiency = final_power / (self.SOLAR_CONSTANT * self.power_calc.panel_area)
        eol_efficiency_percent = final_efficiency * 100.0

        # Find years to specific degradation levels
        years_to_80 = self._find_years_to_degradation_level(power_history, 20.0)  # 20% degradation
        years_to_50 = self._find_years_to_degradation_level(power_history, 50.0)  # 50% degradation

        # Calculate mechanism contributions
        if self.lifetime_history:
            final_state = self.lifetime_history[-1]
            mechanism_contributions = {
                mech.name: mech.contribution_percent
                for mech in final_state.mechanisms
            }
        else:
            mechanism_contributions = {}

        return LifetimePrediction(
            initial_power_watts=initial_power,
            final_power_watts=final_power,
            total_degradation_percent=total_degradation,
            mission_duration_hours=duration_years * 365.25 * 24.0,
            degradation_rate_percent_per_year=degradation_rate,
            average_power_watts=avg_power,
            minimum_power_watts=min_power,
            maximum_power_watts=max_power,
            total_energy_Wh=total_energy,
            eol_efficiency_percent=eol_efficiency_percent,
            years_to_80_percent=years_to_80,
            years_to_50_percent=years_to_50,
            mechanism_contributions=mechanism_contributions
        )

    def _find_years_to_degradation_level(self, power_history: List[PowerOutput],
                                        degradation_percent: float) -> Optional[float]:
        """Find years to reach specific degradation level"""
        if not power_history or not self.launch_time:
            return None

        initial_power = power_history[0].power_watts
        target_power = initial_power * (1.0 - degradation_percent / 100.0)

        for i, power_output in enumerate(power_history):
            if power_output.power_watts <= target_power:
                time_diff = power_output.time - self.launch_time
                return time_diff.total_days() / 365.25

        return None  # Target degradation not reached

    def compare_scenarios(self, scenarios: Dict[str, Dict],
                         duration_years: float = 10.0) -> Dict[str, LifetimePrediction]:
        """
        Compare multiple mission scenarios

        Args:
            scenarios: Dictionary of scenario configurations
            duration_years: Simulation duration

        Returns:
            Dictionary of predictions for each scenario
        """
        results = {}

        for scenario_name, scenario_config in scenarios.items():
            # Configure models for this scenario
            # (This would involve updating model parameters based on scenario_config)
            # For now, run with current configuration

            # Run simulation
            prediction = self.simulate_lifetime(
                launch_time=datetime.now(),
                duration_years=duration_years
            )

            results[scenario_name] = prediction

        return results

    def get_lifetime_summary(self) -> Dict:
        """
        Get comprehensive lifetime degradation summary

        Returns:
            Dictionary with lifetime analysis
        """
        if not self.lifetime_history:
            return {"status": "No simulation data available"}

        final_state = self.lifetime_history[-1]
        initial_state = self.lifetime_history[0]

        # Calculate mechanism contributions
        mechanism_breakdown = {}
        total_degradation = 0.0

        for mechanism in final_state.mechanisms:
            mechanism_breakdown[mechanism.name] = {
                'contribution_percent': mechanism.contribution_percent,
                'cumulative_damage': mechanism.cumulative_damage,
                'activation': mechanism.activation
            }
            total_degradation += mechanism.cumulative_damage

        return {
            'mission_summary': {
                'launch_time': self.launch_time.isoformat() if self.launch_time else None,
                'mission_duration_hours': final_state.mission_time_hours,
                'mission_duration_days': final_state.mission_time_hours / 24.0,
                'mission_duration_years': final_state.mission_time_hours / (365.25 * 24.0)
            },
            'power_performance': {
                'initial_power_W': initial_state.current_power_watts,
                'final_power_W': final_state.current_power_watts,
                'total_degradation_percent': final_state.power_degradation_percent,
                'average_efficiency_factor': np.mean([state.efficiency_factor for state in self.lifetime_history]),
                'final_efficiency_factor': final_state.efficiency_factor
            },
            'environmental_exposure': {
                'total_radiation_dose_rads': final_state.total_radiation_dose_rads,
                'total_thermal_cycles': final_state.total_thermal_cycles,
                'max_temperature_K': final_state.max_temperature_K,
                'min_temperature_K': final_state.min_temperature_K,
                'temperature_range_K': final_state.max_temperature_K - final_state.min_temperature_K,
                'total_eclipse_hours': final_state.total_eclipse_hours,
                'eclipse_fraction': final_state.total_eclipse_hours / final_state.mission_time_hours
            },
            'degradation_mechanisms': mechanism_breakdown,
            'individual_factors': {
                'radiation_damage': final_state.radiation_damage,
                'thermal_damage': final_state.thermal_damage,
                'contamination_damage': final_state.contamination_damage,
                'aging_damage': final_state.aging_damage
            },
            'recommendations': self._generate_mission_recommendations(final_state)
        }

    def _generate_mission_recommendations(self, final_state: LifetimeState) -> List[str]:
        """Generate mission-specific recommendations"""
        recommendations = []

        # Radiation damage recommendations
        if final_state.radiation_damage < 0.8:
            recommendations.append("Consider additional radiation shielding or more radiation-hardened solar cells")

        # Thermal cycling recommendations
        if final_state.thermal_damage < 0.9:
            recommendations.append("Implement thermal control measures to reduce temperature cycling")

        # High temperature recommendations
        if final_state.max_temperature_K > 373.15:  # 100°C
            recommendations.append("Maximum temperature exceeded 100°C - consider improved thermal management")

        # Eclipse recommendations
        eclipse_fraction = final_state.total_eclipse_hours / final_state.mission_time_hours
        if eclipse_fraction > 0.4:  # More than 40% eclipse time
            recommendations.append("High eclipse fraction - ensure adequate battery storage")

        # Overall performance recommendations
        if final_state.power_degradation_percent > 25:
            recommendations.append("High degradation rate - consider design improvements or shorter mission duration")

        if not recommendations:
            recommendations.append("Mission performance within acceptable parameters")

        return recommendations