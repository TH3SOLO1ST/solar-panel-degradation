"""
Radiation Damage Model Module

This module implements radiation-induced damage models for solar cells using the
Displacement Damage Dose (DDD) methodology and surface contamination effects.
It supports different solar cell technologies including Silicon and Multi-junction
cells.

References:
- "Displacement Damage Dose Methodology for Solar Cell Degradation" - NRL
- "Radiation Effects in Space Solar Cells" - NASA
- "Solar Cell Radiation Handbook" - JPL
- "Multi-junction solar cell radiation modeling" - ESA
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import math

from .radiation_environment import RadiationDose, RadiationFlux


@dataclass
class SolarCellProperties:
    """Physical properties of solar cell technology"""
    technology: str          # "silicon", "multi_junction_gaas", "multi_junction_inGaP"
    bandgap_ev: float        # Band gap energy in eV
    absorption_coeff: float  # Absorption coefficient (1/cm)
    diffusion_length: float  # Minority carrier diffusion length (cm)
    junction_depth: float    # Junction depth (cm)
    base_resistivity: float  # Base resistivity (ohm-cm)
    thickness: float         # Cell thickness (cm)


@dataclass
class RadiationDamageCoefficients:
    """Radiation damage coefficients for a specific cell technology"""
    # Displacement damage coefficients
    A_dd: float              # Power degradation coefficient (DDD model)
    B_dd: float              # DDD threshold parameter
    annealing_temp: float    # Annealing temperature (K)
    annealing_energy: float  # Annealing activation energy (eV)

    # Surface contamination coefficients
    C_surface: float         # Surface darkening coefficient
    D_power: float           # Power exponent for surface effects

    # Ionizing dose coefficients
    K_ionizing: float        # Ionizing dose damage coefficient


@dataclass
class RadiationDamageState:
    """Current state of radiation-induced damage"""
    power_efficiency_factor: float    # Relative efficiency (0-1)
    voltage_degradation: float        # Voltage degradation (%)
    current_degradation: float        # Current degradation (%)
    fill_factor_degradation: float    # Fill factor degradation (%)
    series_resistance_increase: float # Series resistance increase (%)
    ddd_accumulated: float            # Accumulated DDD (MeV·cm²/g)
    ionizing_dose_rads: float         # Accumulated ionizing dose (rads)
    surface_transmission_loss: float  # Surface transmission loss (%)


class RadiationDamageModel:
    """
    Comprehensive radiation damage model for solar cells.

    Features:
    - Displacement Damage Dose (DDD) methodology
    - Multi-junction cell modeling
    - Temperature annealing effects
    - Surface contamination modeling
    - Technology-specific damage coefficients
    - Time-dependent degradation recovery
    """

    # Physical constants
    k_B = 8.617333e-5  # Boltzmann constant (eV/K)
    ELECTRON_CHARGE = 1.602e-19  # Coulombs

    # Standard solar cell parameters
    STANDARD_TESTING_TEMP = 298.15  # K (25°C)
    SOLAR_CONSTANT = 1361  # W/m²

    # Pre-defined damage coefficients for common technologies
    TECHNOLOGY_COEFFICIENTS = {
        "silicon": RadiationDamageCoefficients(
            A_dd=0.08,              # From NRL DDD methodology
            B_dd=1e10,              # 1 MeV e⁻ equivalent threshold
            annealing_temp=373.15,  # 100°C annealing temperature
            annealing_energy=0.5,   # 0.5 eV activation energy
            C_surface=0.02,         # Surface darkening coefficient
            D_power=0.5,            # Square root dependence
            K_ionizing=1e-6         # Ionizing dose coefficient
        ),
        "multi_junction_gaas": RadiationDamageCoefficients(
            A_dd=0.10,              # Higher radiation sensitivity
            B_dd=8e9,               # Lower threshold
            annealing_temp=423.15,  # 150°C annealing
            annealing_energy=0.6,   # Higher activation energy
            C_surface=0.015,        # Better radiation resistance
            D_power=0.4,
            K_ionizing=5e-7         # Lower ionizing sensitivity
        ),
        "multi_junction_inGaP": RadiationDamageCoefficients(
            A_dd=0.12,              # Most radiation sensitive top cell
            B_dd=5e9,               # Lowest threshold
            annealing_temp=473.15,  # 200°C annealing
            annealing_energy=0.7,   # Highest activation energy
            C_surface=0.01,         # Best surface radiation resistance
            D_power=0.3,
            K_ionizing=3e-7
        )
    }

    # Solar cell properties
    CELL_PROPERTIES = {
        "silicon": SolarCellProperties(
            technology="silicon",
            bandgap_ev=1.12,
            absorption_coeff=100.0,
            diffusion_length=0.02,
            junction_depth=0.0003,
            base_resistivity=10.0,
            thickness=0.02
        ),
        "multi_junction_gaas": SolarCellProperties(
            technology="multi_junction_gaas",
            bandgap_ev=1.42,        # GaAs middle cell
            absorption_coeff=500.0,
            diffusion_length=0.01,
            junction_depth=0.0001,
            base_resistivity=1.0,
            thickness=0.005
        ),
        "multi_junction_inGaP": SolarCellProperties(
            technology="multi_junction_inGaP",
            bandgap_ev=1.89,        # InGaP top cell
            absorption_coeff=1000.0,
            diffusion_length=0.008,
            junction_depth=0.00005,
            base_resistivity=0.5,
            thickness=0.003
        )
    }

    def __init__(self, cell_technology: str = "silicon",
                 temperature_K: float = STANDARD_TESTING_TEMP,
                 custom_coefficients: Optional[RadiationDamageCoefficients] = None):
        """
        Initialize radiation damage model

        Args:
            cell_technology: Solar cell technology type
            temperature_K: Operating temperature in Kelvin
            custom_coefficients: Optional custom damage coefficients
        """
        self.cell_technology = cell_technology
        self.temperature_K = temperature_K

        # Set damage coefficients
        if custom_coefficients:
            self.coefficients = custom_coefficients
        elif cell_technology in self.TECHNOLOGY_COEFFICIENTS:
            self.coefficients = self.TECHNOLOGY_COEFFICIENTS[cell_technology]
        else:
            raise ValueError(f"Unknown cell technology: {cell_technology}")

        # Get cell properties
        if cell_technology in self.CELL_PROPERTIES:
            self.cell_props = self.CELL_PROPERTIES[cell_technology]
        else:
            raise ValueError(f"No properties defined for technology: {cell_technology}")

        # Initialize damage state
        self.damage_state = RadiationDamageState(
            power_efficiency_factor=1.0,
            voltage_degradation=0.0,
            current_degradation=0.0,
            fill_factor_degradation=0.0,
            series_resistance_increase=0.0,
            ddd_accumulated=0.0,
            ionizing_dose_rads=0.0,
            surface_transmission_loss=0.0
        )

    def calculate_displacement_damage(self, fluxes: List[RadiationFlux],
                                    exposure_time_hours: float) -> float:
        """
        Calculate displacement damage dose from radiation fluxes

        Args:
            fluxes: List of radiation fluxes
            exposure_time_hours: Exposure duration in hours

        Returns:
            Displacement damage dose (MeV·cm²/g)
        """
        total_ddd = 0.0

        for flux in fluxes:
            if flux.particle_type in ["electron", "proton"]:
                # NIEL (Non-Ionizing Energy Loss) coefficients
                if flux.particle_type == "electron":
                    # NIEL for electrons in silicon/solar cell material
                    if flux.energy_mev < 0.1:
                        niel = 0.01  # MeV·cm²/g
                    elif flux.energy_mev < 1.0:
                        niel = 0.01 * (flux.energy_mev / 0.1)
                    else:
                        niel = 0.1 * (flux.energy_mev / 1.0)**-0.5
                else:  # proton
                    # NIEL for protons (higher damage potential)
                    if flux.energy_mev < 1.0:
                        niel = 0.1
                    elif flux.energy_mev < 10.0:
                        niel = 0.1 * (flux.energy_mev / 1.0)**0.8
                    else:
                        niel = 0.6 * (flux.energy_mev / 10.0)**-0.2

                # Calculate DDD contribution
                # DDD = ∫ Φ(E) × NIEL(E) dE
                # Discrete approximation: DDD ≈ Σ Φ(E) × NIEL(E) × ΔE × Δt
                delta_E = flux.energy_range_mev[1] - flux.energy_range_mev[0]
                ddd_contribution = (flux.flux_particles * niel * delta_E *
                                  exposure_time_hours * 3600)  # Convert to seconds

                total_ddd += ddd_contribution

        return total_ddd

    def calculate_power_degradation_from_ddd(self, ddd: float) -> float:
        """
        Calculate power degradation factor using DDD methodology

        Args:
            ddd: Displacement damage dose (MeV·cm²/g)

        Returns:
            Power efficiency factor (0-1, where 1 = no degradation)
        """
        # DDD methodology: P/P₀ = 1 - A × log₁₀(DDD + B)
        if ddd <= 0:
            return 1.0

        degradation_factor = self.coefficients.A_dd * np.log10(ddd + self.coefficients.B_dd)
        efficiency_factor = max(0.0, 1.0 - degradation_factor)

        return efficiency_factor

    def calculate_surface_contamination_damage(self, ionizing_dose_rads: float) -> float:
        """
        Calculate surface contamination/transmission loss

        Args:
            ionizing_dose_rads: Accumulated ionizing dose in rads

        Returns:
            Transmission loss factor (0-1, where 1 = no loss)
        """
        if ionizing_dose_rads <= 0:
            return 1.0

        # Surface darkening: Transmission loss = C × D^D_power
        transmission_loss = self.coefficients.C_surface * (ionizing_dose_rads ** self.coefficients.D_power)
        transmission_factor = max(0.0, 1.0 - transmission_loss)

        return transmission_factor

    def apply_temperature_annealing(self, time_hours: float, temperature_K: float) -> float:
        """
        Calculate annealing recovery factor

        Args:
            time_hours: Annealing time duration
            temperature_K: Annealing temperature

        Returns:
            Recovery factor (0-1, where 1 = full recovery)
        """
        if temperature_K <= self.coefficients.annealing_temp:
            return 0.0  # No annealing below threshold

        # Arrhenius-type annealing: Recovery = 1 - exp(-t/τ)
        # τ = τ₀ × exp(Ea/kT)
        tau_0 = 1e6  # seconds (pre-exponential factor)
        tau = tau_0 * np.exp(self.coefficients.annealing_energy / (self.k_B * temperature_K))

        time_seconds = time_hours * 3600
        recovery = 1.0 - np.exp(-time_seconds / tau)

        return min(1.0, recovery)

    def update_damage_state(self, radiation_dose: RadiationDose,
                          fluxes: List[RadiationFlux],
                          exposure_time_hours: float,
                          apply_annealing: bool = True) -> RadiationDamageState:
        """
        Update damage state based on new radiation exposure

        Args:
            radiation_dose: Accumulated radiation dose
            fluxes: Radiation fluxes during exposure
            exposure_time_hours: Exposure duration
            apply_annealing: Apply temperature annealing effects

        Returns:
            Updated damage state
        """
        # Calculate new DDD accumulation
        new_ddd = self.calculate_displacement_damage(fluxes, exposure_time_hours)

        # Update total accumulated DDD
        self.damage_state.ddd_accumulated += new_ddd

        # Update ionizing dose
        self.damage_state.ionizing_dose_rads += radiation_dose.dose_rads

        # Calculate power degradation from DDD
        ddd_efficiency = self.calculate_power_degradation_from_ddd(self.damage_state.ddd_accumulated)

        # Calculate surface contamination damage
        surface_transmission = self.calculate_surface_contamination_damage(self.damage_state.ionizing_dose_rads)

        # Apply annealing if specified
        annealing_factor = 1.0
        if apply_annealing:
            annealing_recovery = self.apply_temperature_annealing(exposure_time_hours, self.temperature_K)
            annealing_factor = 1.0 - annealing_recovery * 0.3  # Max 30% recovery

        # Combined efficiency factor
        self.damage_state.power_efficiency_factor = ddd_efficiency * surface_transmission * annealing_factor

        # Calculate individual parameter degradations
        self._calculate_parameter_degradations()

        return self.damage_state

    def _calculate_parameter_degradations(self):
        """Calculate degradations for individual cell parameters"""
        base_efficiency_loss = 1.0 - self.damage_state.power_efficiency_factor

        # Distribute degradation among parameters (simplified model)
        self.damage_state.voltage_degradation = base_efficiency_loss * 0.3  # 30% from voltage
        self.damage_state.current_degradation = base_efficiency_loss * 0.4  # 40% from current
        self.damage_state.fill_factor_degradation = base_efficiency_loss * 0.3  # 30% from fill factor

        # Series resistance increases with displacement damage
        self.damage_state.series_resistance_increase = self.damage_state.ddd_accumulated * 1e-8

        # Surface transmission loss
        self.damage_state.surface_transmission_loss = (
            1.0 - self.calculate_surface_contamination_damage(self.damage_state.ionizing_dose_rads)
        ) * 100.0  # Convert to percentage

    def predict_lifetime_degradation(self, radiation_environment_model,
                                   orbit_propagator, start_time: datetime,
                                   duration_years: float, time_step_hours: float = 24.0) -> List[RadiationDamageState]:
        """
        Predict lifetime degradation based on orbital environment

        Args:
            radiation_environment_model: Radiation environment model
            orbit_propagator: Orbital propagator for position calculation
            start_time: Mission start time
            duration_years: Mission duration in years
            time_step_hours: Time step for calculations

        Returns:
            List of damage states over time
        """
        from datetime import timedelta

        damage_timeline = []
        current_time = start_time
        end_time = start_time + timedelta(days=duration_years * 365.25)

        # Reset damage state
        self.damage_state = RadiationDamageState(
            power_efficiency_factor=1.0,
            voltage_degradation=0.0,
            current_degradation=0.0,
            fill_factor_degradation=0.0,
            series_resistance_increase=0.0,
            ddd_accumulated=0.0,
            ionizing_dose_rads=0.0,
            surface_transmission_loss=0.0
        )

        while current_time <= end_time:
            # Get satellite position
            state = orbit_propagator.propagate(current_time)

            # Calculate radiation fluxes at this position
            trapped_fluxes = radiation_environment_model.calculate_trapped_particle_flux(
                state.position, current_time
            )
            spe_fluxes = radiation_environment_model.calculate_solar_particle_event_flux(
                state.position, current_time
            )
            gcr_fluxes = radiation_environment_model.calculate_gcr_flux(
                state.position, current_time
            )

            all_fluxes = trapped_fluxes + spe_fluxes + gcr_fluxes

            # Calculate radiation dose for this time step
            dose = radiation_environment_model.calculate_radiation_dose(
                all_fluxes, 0.1, time_step_hours  # 1mm shielding
            )

            # Update damage state
            self.update_damage_state(dose, all_fluxes, time_step_hours)

            # Store current state (deep copy)
            current_state = RadiationDamageState(
                power_efficiency_factor=self.damage_state.power_efficiency_factor,
                voltage_degradation=self.damage_state.voltage_degradation,
                current_degradation=self.damage_state.current_degradation,
                fill_factor_degradation=self.damage_state.fill_factor_degradation,
                series_resistance_increase=self.damage_state.series_resistance_increase,
                ddd_accumulated=self.damage_state.ddd_accumulated,
                ionizing_dose_rads=self.damage_state.ionizing_dose_rads,
                surface_transmission_loss=self.damage_state.surface_transmission_loss
            )
            damage_timeline.append(current_state)

            current_time += timedelta(hours=time_step_hours)

        return damage_timeline

    def get_cell_parameters_at_state(self, initial_params: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate current cell parameters based on damage state

        Args:
            initial_params: Initial cell parameters before radiation damage

        Returns:
            Current cell parameters after degradation
        """
        current_params = initial_params.copy()

        # Apply voltage degradation
        if 'voc' in current_params:
            current_params['voc'] *= (1.0 - self.damage_state.voltage_degradation / 100.0)

        # Apply current degradation
        if 'isc' in current_params:
            current_params['isc'] *= (1.0 - self.damage_state.current_degradation / 100.0)

        # Apply fill factor degradation
        if 'ff' in current_params:
            current_params['ff'] *= (1.0 - self.damage_state.fill_factor_degradation / 100.0)

        # Apply series resistance increase
        if 'rs' in current_params:
            current_params['rs'] *= (1.0 + self.damage_state.series_resistance_increase / 100.0)

        # Apply efficiency degradation
        if 'efficiency' in current_params:
            current_params['efficiency'] *= self.damage_state.power_efficiency_factor

        return current_params

    def get_damage_summary(self) -> Dict:
        """
        Get comprehensive damage summary

        Returns:
            Dictionary with damage analysis
        """
        return {
            'cell_technology': self.cell_technology,
            'temperature_K': self.temperature_K,
            'damage_state': {
                'power_efficiency_factor': self.damage_state.power_efficiency_factor,
                'power_degradation_percent': (1.0 - self.damage_state.power_efficiency_factor) * 100.0,
                'voltage_degradation_percent': self.damage_state.voltage_degradation,
                'current_degradation_percent': self.damage_state.current_degradation,
                'fill_factor_degradation_percent': self.damage_state.fill_factor_degradation,
                'series_resistance_increase_percent': self.damage_state.series_resistance_increase,
                'ddd_accumulated_mev_cm2_g': self.damage_state.ddd_accumulated,
                'ionizing_dose_rads': self.damage_state.ionizing_dose_rads,
                'surface_transmission_loss_percent': self.damage_state.surface_transmission_loss
            },
            'damage_coefficients': {
                'A_dd': self.coefficients.A_dd,
                'B_dd': self.coefficients.B_dd,
                'annealing_temp_K': self.coefficients.annealing_temp,
                'C_surface': self.coefficients.C_surface,
                'D_power': self.coefficients.D_power
            }
        }