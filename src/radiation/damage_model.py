"""
Radiation Damage Model
======================

Converts radiation dose to solar cell degradation calculations.

This module implements the physics-based models for converting radiation
exposure into solar cell performance degradation.

Classes:
    RadiationDamageModel: Main radiation damage calculation class
    SiliconCellModel: Silicon solar cell radiation damage
    MultiJunctionCellModel: Multi-junction solar cell radiation damage
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CellDegradation:
    """Data structure for cell degradation information"""
    efficiency_remaining: float  # Fraction of initial efficiency
    power_output_factor: float   # Fraction of initial power
    degradation_mechanisms: Dict[str, float]  # Breakdown by mechanism
    temperature_coefficient: float  # Temperature dependence

class SiliconCellModel:
    """Radiation damage model for silicon solar cells"""

    def __init__(self):
        """Initialize silicon cell damage model"""
        # Silicon cell parameters from literature
        self.degradation_coeff_A = 0.08  # Displacement damage coefficient
        self.degradation_coeff_B = 1e10  # Damage threshold (1 MeV eq)

        # Temperature annealing parameters
        self.annealing_activation_energy = 0.1  # eV
        self.boltzmann_constant = 8.617e-5  # eV/K

        # Surface darkening parameters
        self.surface_coefficient = 0.02  # Ionizing dose darkening coefficient

    def calculate_displacement_damage(self, ddd_MeV_cm2_g: float,
                                   temperature_K: float) -> float:
        """
        Calculate efficiency degradation from displacement damage

        Args:
            ddd_MeV_cm2_g: Displacement damage dose in MeV cm²/g
            temperature_K: Cell temperature in Kelvin

        Returns:
            Efficiency remaining factor (0 to 1)
        """
        # Displacement Damage Dose (DDD) method
        # P/P₀ = 1 - A × log₁₀(DDD + B)

        if ddd_MeV_cm2_g <= 0:
            return 1.0

        efficiency_factor = 1 - self.degradation_coeff_A * np.log10(
            ddd_MeV_cm2_g + self.degradation_coeff_B
        )

        # Temperature annealing effects
        annealing_factor = self._calculate_annealing_factor(temperature_K)
        efficiency_factor = 1 - (1 - efficiency_factor) * annealing_factor

        return max(0.0, min(1.0, efficiency_factor))

    def _calculate_annealing_factor(self, temperature_K: float) -> float:
        """
        Calculate radiation damage annealing factor

        Args:
            temperature_K: Temperature in Kelvin

        Returns:
            Annealing factor (0 to 1, where 1 = no annealing)
        """
        # Annealing rate decreases at lower temperatures
        annealing_rate = np.exp(-self.annealing_activation_energy /
                                (self.boltzmann_constant * temperature_K))

        # Normalize to room temperature (293K)
        room_temp_rate = np.exp(-self.annealing_activation_energy /
                               (self.boltzmann_constant * 293.0))

        return min(1.0, annealing_rate / room_temp_rate)

    def calculate_surface_darkening(self, ionizing_dose_rads: float) -> float:
        """
        Calculate transmission loss from surface darkening

        Args:
            ionizing_dose_rads: Ionizing radiation dose in rads

        Returns:
            Transmission remaining factor (0 to 1)
        """
        # Transmission loss = C × D^0.5
        if ionizing_dose_rads <= 0:
            return 1.0

        transmission_loss = self.surface_coefficient * np.sqrt(ionizing_dose_rads)
        transmission_factor = 1 - transmission_loss

        return max(0.0, min(1.0, transmission_factor))

class MultiJunctionCellModel:
    """Radiation damage model for multi-junction solar cells"""

    def __init__(self):
        """Initialize multi-junction cell damage model"""
        # Three-junction GaInP/GaAs/Ge cell parameters
        self.subcells = {
            'top': {    # GaInP
                'coeff_A': 0.12,
                'coeff_B': 5e9,
                'bandgap': 1.85  # eV
            },
            'middle': { # GaAs
                'coeff_A': 0.10,
                'coeff_B': 8e9,
                'bandgap': 1.42  # eV
            },
            'bottom': { # Ge
                'coeff_A': 0.06,
                'coeff_B': 1e10,
                'bandgap': 0.66  # eV
            }
        }

    def calculate_subcell_degradation(self, ddd_MeV_cm2_g: float,
                                    subcell: str) -> float:
        """
        Calculate degradation for individual subcell

        Args:
            ddd_MeV_cm2_g: Displacement damage dose
            subcell: 'top', 'middle', or 'bottom'

        Returns:
            Efficiency remaining factor for subcell
        """
        if subcell not in self.subcells:
            raise ValueError(f"Invalid subcell: {subcell}")

        params = self.subcells[subcell]

        if ddd_MeV_cm2_g <= 0:
            return 1.0

        efficiency_factor = 1 - params['coeff_A'] * np.log10(
            ddd_MeV_cm2_g + params['coeff_B']
        )

        return max(0.0, min(1.0, efficiency_factor))

    def calculate_total_degradation(self, ddd_MeV_cm2_g: float) -> float:
        """
        Calculate total multi-junction cell degradation

        Args:
            ddd_MeV_cm2_g: Displacement damage dose

        Returns:
            Total efficiency remaining factor
        """
        # Calculate degradation for each subcell
        top_efficiency = self.calculate_subcell_degradation(ddd_MeV_cm2_g, 'top')
        middle_efficiency = self.calculate_subcell_degradation(ddd_MeV_cm2_g, 'middle')
        bottom_efficiency = self.calculate_subcell_degradation(ddd_MeV_cm2_g, 'bottom')

        # Multi-junction cells are limited by the worst-performing subcell
        total_efficiency = min(top_efficiency, middle_efficiency, bottom_efficiency)

        return total_efficiency

class RadiationDamageModel:
    """Main radiation damage calculation class"""

    def __init__(self, cell_type: str = "silicon"):
        """
        Initialize radiation damage model

        Args:
            cell_type: Type of solar cell ('silicon' or 'multi_junction')
        """
        self.cell_type = cell_type

        if cell_type == "silicon":
            self.cell_model = SiliconCellModel()
        elif cell_type == "multi_junction":
            self.cell_model = MultiJunctionCellModel()
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")

    def calculate_degradation(self, radiation_dose, temperature_K: float = 293.0,
                            time_hours: np.ndarray = None) -> CellDegradation:
        """
        Calculate solar cell degradation from radiation exposure

        Args:
            radiation_dose: RadiationDose object
            temperature_K: Average cell temperature in Kelvin
            time_hours: Time points for timeline calculation

        Returns:
            CellDegradation object with degradation information
        """
        # Get radiation dose values
        total_ionizing = radiation_dose.total_ionizing_dose
        total_displacement = radiation_dose.non_ionizing_dose

        # Calculate degradation mechanisms
        if self.cell_type == "silicon":
            # Displacement damage
            displacement_factor = self.cell_model.calculate_displacement_damage(
                total_displacement, temperature_K
            )

            # Surface darkening
            surface_factor = self.cell_model.calculate_surface_darkening(
                total_ionizing
            )

            # Combined degradation (multiplicative)
            efficiency_remaining = displacement_factor * surface_factor

            degradation_breakdown = {
                'displacement_damage': 1 - displacement_factor,
                'surface_darkening': 1 - surface_factor
            }
        else:
            # Multi-junction cells (primarily displacement damage)
            efficiency_remaining = self.cell_model.calculate_total_degradation(
                total_displacement
            )

            degradation_breakdown = {
                'displacement_damage': 1 - efficiency_remaining,
                'surface_darkening': 0.0  # Minimal for multi-junction
            }

        # Temperature coefficient (simplified)
        temperature_coefficient = -0.0004  # per K for silicon cells
        if self.cell_type == "multi_junction":
            temperature_coefficient = -0.0003  # typical for multi-junction

        # Calculate degradation timeline if time points provided
        degradation_timeline = None
        if time_hours is not None and radiation_dose.dose_timeline is not None:
            degradation_timeline = self._calculate_degradation_timeline(
                radiation_dose.dose_timeline, temperature_K, time_hours
            )

        return CellDegradation(
            efficiency_remaining=efficiency_remaining,
            power_output_factor=efficiency_remaining,  # Simplified: same as efficiency
            degradation_mechanisms=degradation_breakdown,
            temperature_coefficient=temperature_coefficient
        )

    def _calculate_degradation_timeline(self, dose_timeline: np.ndarray,
                                       temperature_K: float,
                                       time_hours: np.ndarray) -> np.ndarray:
        """
        Calculate degradation over time from dose timeline

        Args:
            dose_timeline: Cumulative dose over time
            temperature_K: Cell temperature
            time_hours: Time points

        Returns:
            Array of degradation factors over time
        """
        degradation_timeline = np.zeros(len(dose_timeline))

        for i, dose in enumerate(dose_timeline):
            if self.cell_type == "silicon":
                displacement_factor = self.cell_model.calculate_displacement_damage(
                    dose, temperature_K
                )
                # Simplified: assume proportional ionizing dose
                ionizing_dose = dose * 0.1  # Simplified ratio
                surface_factor = self.cell_model.calculate_surface_darkening(
                    ionizing_dose
                )
                degradation_timeline[i] = displacement_factor * surface_factor
            else:
                degradation_timeline[i] = self.cell_model.calculate_total_degradation(
                    dose
                )

        return degradation_timeline

    def get_degradation_rate(self, current_degradation: float,
                           radiation_flux: float, time_step_hours: float) -> float:
        """
        Calculate instantaneous degradation rate

        Args:
            current_degradation: Current degradation level (0 to 1)
            radiation_flux: Current radiation flux
            time_step_hours: Time step in hours

        Returns:
            Degradation increment for this time step
        """
        # Simplified degradation rate calculation
        # Real implementation would integrate over energy spectrum

        dose_increment = radiation_flux * time_step_hours * 3600  # Convert to seconds

        if self.cell_type == "silicon":
            displacement_factor = self.cell_model.calculate_displacement_damage(
                dose_increment, 293.0  # Room temperature
            )
            degradation_increment = 1 - displacement_factor
        else:
            degradation_increment = 1 - self.cell_model.calculate_total_degradation(
                dose_increment
            )

        return degradation_increment