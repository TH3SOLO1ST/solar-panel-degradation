"""
Power Calculator Module

This module calculates instantaneous solar panel power output including
environmental corrections, degradation effects, and I-V curve modeling.
It implements equivalent circuit models and temperature-dependent performance.

References:
- "Solar Cell Device Physics" - Stephen Fonash
- "Solar Cells: Operating Principles, Technology, and System Applications" - Green
- "Photovoltaic Systems Engineering" - Messenger & Ventre
- NASA Solar Cell Performance Handbook
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import math

from ..orbital.orbit_propagator import OrbitPropagator, OrbitalState
from ..orbital.eclipse_calculator import EclipseCalculator
from ..radiation.damage_model import RadiationDamageModel
from ..thermal.thermal_analysis import ThermalAnalysis, ThermalState


@dataclass
class SolarCellParameters:
    """Electrical parameters of solar cell at standard conditions"""
    short_circuit_current: float    # Isc (A/m²)
    open_circuit_voltage: float     # Voc (V)
    max_power_current: float        # Imp (A/m²)
    max_power_voltage: float        # Vmp (V)
    fill_factor: float              # FF (0-1)
    efficiency: float               # η (0-1)
    series_resistance: float        # Rs (Ω·m²)
    shunt_resistance: float         # Rsh (Ω·m²)
    ideality_factor: float          # n (diode ideality factor)
    saturation_current: float       # I₀ (A/m²)


@dataclass
class PowerOutput:
    """Power output data at a specific time"""
    time: datetime
    power_watts: float              # Total power output (W)
    power_density_wm2: float        # Power per unit area (W/m²)
    current_amps: float             # Current output (A)
    voltage_volts: float            # Voltage output (V)
    efficiency: float               # Current efficiency (0-1)
    irradiance_wm2: float           # Solar irradiance (W/m²)
    temperature_K: float            # Cell temperature (K)
    degradation_factor: float       # Overall degradation factor (0-1)
    eclipse_fraction: float         # Eclipse shadow fraction (0-1)


@dataclass
class IVCurve:
    """I-V curve data for solar panel"""
    voltages: np.ndarray            # Voltage points (V)
    currents: np.ndarray            # Current points (A/m²)
    powers: np.ndarray              # Power points (W/m²)
    voc: float                      # Open circuit voltage (V)
    isc: float                      # Short circuit current (A/m²)
    vmp: float                      # Voltage at max power (V)
    imp: float                      # Current at max power (A/m²)
    pmp: float                      # Maximum power (W/m²)
    ff: float                       # Fill factor (0-1)


class PowerCalculator:
    """
    Comprehensive solar panel power calculator.

    Features:
    - I-V curve modeling using equivalent circuit
    - Temperature-dependent performance
    - Radiation degradation effects
    - Eclipse and shadow effects
    - Multi-junction cell modeling
    - Real-time power prediction
    """

    # Physical constants
    k_B = 1.381e-23              # Boltzmann constant (J/K)
    q = 1.602e-19                # Elementary charge (C)
    SOLAR_CONSTANT = 1361        # W/m² at 1 AU

    # Standard test conditions
    STC_TEMPERATURE = 298.15     # K (25°C)
    STC_IRRADIANCE = 1000        # W/m²
    AIR_MASS = 1.5              # AM1.5 solar spectrum

    # Default solar cell parameters (Silicon)
    DEFAULT_CELL_PARAMS = SolarCellParameters(
        short_circuit_current=400,      # A/m² (40 mA/cm²)
        open_circuit_voltage=0.62,      # V
        max_power_current=380,          # A/m² (38 mA/cm²)
        max_power_voltage=0.52,         # V
        fill_factor=0.82,               # 82%
        efficiency=0.20,                # 20%
        series_resistance=0.01,         # Ω·m²
        shunt_resistance=1000,          # Ω·m²
        ideality_factor=1.2,            # Typical for silicon
        saturation_current=1e-10        # A/m²
    )

    def __init__(self, panel_area_m2: float,
                 cell_parameters: Optional[SolarCellParameters] = None,
                 cell_type: str = "silicon"):
        """
        Initialize power calculator

        Args:
            panel_area_m2: Solar panel area in square meters
            cell_parameters: Solar cell electrical parameters
            cell_type: Type of solar cell technology
        """
        self.panel_area = panel_area_m2
        self.cell_type = cell_type

        # Set cell parameters
        if cell_parameters:
            self.cell_params = cell_parameters
        else:
            self.cell_params = self.DEFAULT_CELL_PARAMS

        # Initialize power output history
        self.power_history: List[PowerOutput] = []

    def calculate_thermal_voltage(self, temperature_K: float) -> float:
        """
        Calculate thermal voltage Vt = kT/q

        Args:
            temperature_K: Temperature in Kelvin

        Returns:
            Thermal voltage in Volts
        """
        return self.k_B * temperature_K / self.q

    def calculate_saturation_current(self, temperature_K: float) -> float:
        """
        Calculate temperature-dependent saturation current

        Args:
            temperature_K: Temperature in Kelvin

        Returns:
            Saturation current (A/m²)
        """
        # I₀(T) = I₀(Tref) × (T/Tref)³ × exp(-Eg/q × (1/T - 1/Tref)/Vt)
        T_ref = self.STC_TEMPERATURE
        Eg = 1.12  # Band gap of silicon in eV
        Vt = self.calculate_thermal_voltage(temperature_K)
        Vt_ref = self.calculate_thermal_voltage(T_ref)

        I0_T = (self.cell_params.saturation_current *
               (temperature_K / T_ref)**3 *
               np.exp(-Eg * (1/temperature_K - 1/T_ref) / Vt))

        return I0_T

    def calculate_bandgap_temp_coefficient(self, temperature_K: float) -> float:
        """
        Calculate temperature-dependent band gap

        Args:
            temperature_K: Temperature in Kelvin

        Returns:
            Band gap energy in eV
        """
        # Varshni equation for silicon band gap
        Eg_0 = 1.17    # Band gap at 0K
        alpha = 4.73e-4  # eV/K
        beta = 636       # K

        Eg_T = Eg_0 - (alpha * temperature_K**2) / (temperature_K + beta)
        return Eg_T

    def calculate_photocurrent(self, irradiance_wm2: float,
                             temperature_K: float) -> float:
        """
        Calculate photocurrent based on irradiance and temperature

        Args:
            irradiance_wm2: Solar irradiance (W/m²)
            temperature_K: Cell temperature (K)

        Returns:
            Photocurrent (A/m²)
        """
        # Linear scaling with irradiance
        I_ph_STC = self.cell_params.short_circuit_current
        I_ph = I_ph_STC * (irradiance_wm2 / self.STC_IRRADIANCE)

        # Temperature coefficient for current (positive)
        alpha_Isc = 0.0005  # 0.05%/K for silicon
        delta_T = temperature_K - self.STC_TEMPERATURE

        I_ph_temp = I_ph * (1 + alpha_Isc * delta_T)

        return I_ph_temp

    def calculate_single_diode_current(self, voltage: float,
                                     photocurrent: float,
                                     saturation_current: float,
                                     thermal_voltage: float,
                                     series_resistance: float,
                                     shunt_resistance: float) -> float:
        """
        Calculate current using single-diode model

        Args:
            voltage: Voltage across cell (V)
            photocurrent: Photocurrent (A/m²)
            saturation_current: Diode saturation current (A/m²)
            thermal_voltage: Thermal voltage (V)
            series_resistance: Series resistance (Ω·m²)
            shunt_resistance: Shunt resistance (Ω·m²)

        Returns:
            Current (A/m²)
        """
        # Single-diode equation: I = Iph - I₀ × exp((V + I×Rs)/(n×Vt)) - (V + I×Rs)/Rsh
        # This requires iterative solution, use Lambert W function approximation

        n = self.cell_params.ideality_factor

        # Simplified analytical solution (ignoring Rs and Rsh for initial estimate)
        I_approx = photocurrent - saturation_current * (np.exp(voltage / (n * thermal_voltage)) - 1)

        # Include series resistance effect (first-order correction)
        if series_resistance > 0:
            I_corrected = I_approx * (1 - series_resistance * I_approx / voltage) if voltage != 0 else I_approx
        else:
            I_corrected = I_approx

        # Include shunt resistance effect
        if shunt_resistance > 0:
            I_shunt = voltage / shunt_resistance
            I_corrected -= I_shunt

        return max(0, I_corrected)

    def calculate_iv_curve(self, irradiance_wm2: float,
                          temperature_K: float,
                          degradation_factor: float = 1.0) -> IVCurve:
        """
        Calculate complete I-V curve

        Args:
            irradiance_wm2: Solar irradiance (W/m²)
            temperature_K: Cell temperature (K)
            degradation_factor: Overall degradation factor (0-1)

        Returns:
            Complete I-V curve data
        """
        # Calculate temperature-dependent parameters
        Vt = self.calculate_thermal_voltage(temperature_K)
        I0 = self.calculate_saturation_current(temperature_K)
        I_ph = self.calculate_photocurrent(irradiance_wm2, temperature_K)

        # Apply degradation to parameters
        I_ph *= degradation_factor
        I0 /= degradation_factor  # Saturation current increases with degradation
        Rs = self.cell_params.series_resistance * (2 - degradation_factor)  # Rs increases with degradation
        Rsh = self.cell_params.shunt_resistance * degradation_factor  # Rsh decreases with degradation

        # Generate voltage points
        V_max = self.cell_params.open_circuit_voltage * 1.2  # Slightly beyond Voc
        voltages = np.linspace(0, V_max, 200)

        # Calculate current for each voltage
        currents = np.array([
            self.calculate_single_diode_current(V, I_ph, I0, Vt, Rs, Rsh)
            for V in voltages
        ])

        # Calculate power
        powers = voltages * currents

        # Find key points
        # Open circuit voltage (where current = 0)
        voc_idx = np.where(currents <= 0)[0]
        voc = voltages[voc_idx[0]] if len(voc_idx) > 0 else voltages[-1]

        # Short circuit current (at V = 0)
        isc = currents[0]

        # Maximum power point
        max_power_idx = np.argmax(powers)
        vmp = voltages[max_power_idx]
        imp = currents[max_power_idx]
        pmp = powers[max_power_idx]

        # Fill factor
        ff = pmp / (voc * isc) if (voc * isc) > 0 else 0

        return IVCurve(
            voltages=voltages,
            currents=currents,
            powers=powers,
            voc=voc,
            isc=isc,
            vmp=vmp,
            imp=imp,
            pmp=pmp,
            ff=ff
        )

    def calculate_power_output(self, time: datetime,
                             irradiance_wm2: float,
                             temperature_K: float,
                             degradation_factor: float = 1.0,
                             eclipse_fraction: float = 0.0) -> PowerOutput:
        """
        Calculate instantaneous power output

        Args:
            time: Time for calculation
            irradiance_wm2: Solar irradiance (W/m²)
            temperature_K: Cell temperature (K)
            degradation_factor: Overall degradation factor (0-1)
            eclipse_fraction: Eclipse shadow fraction (0-1)

        Returns:
            Power output data
        """
        # Apply eclipse reduction
        effective_irradiance = irradiance_wm2 * (1.0 - eclipse_fraction)

        if effective_irradiance <= 0:
            # No power in complete eclipse
            return PowerOutput(
                time=time,
                power_watts=0.0,
                power_density_wm2=0.0,
                current_amps=0.0,
                voltage_volts=0.0,
                efficiency=0.0,
                irradiance_wm2=irradiance_wm2,
                temperature_K=temperature_K,
                degradation_factor=degradation_factor,
                eclipse_fraction=eclipse_fraction
            )

        # Calculate I-V curve
        iv_curve = self.calculate_iv_curve(effective_irradiance, temperature_K, degradation_factor)

        # Maximum power point
        power_density = iv_curve.pmp
        total_power = power_density * self.panel_area

        # Operating voltage and current at max power
        operating_voltage = iv_curve.vmp
        operating_current = iv_curve.imp * self.panel_area

        # Actual efficiency at current conditions
        efficiency = (total_power / (effective_irradiance * self.panel_area)
                     if effective_irradiance > 0 else 0.0)

        return PowerOutput(
            time=time,
            power_watts=total_power,
            power_density_wm2=power_density,
            current_amps=operating_current,
            voltage_volts=operating_voltage,
            efficiency=efficiency,
            irradiance_wm2=irradiance_wm2,
            temperature_K=temperature_K,
            degradation_factor=degradation_factor,
            eclipse_fraction=eclipse_fraction
        )

    def calculate_power_series(self, time_series: List[datetime],
                             irradiance_series: List[float],
                             temperature_series: List[float],
                             degradation_factor: float = 1.0,
                             eclipse_fractions: Optional[List[float]] = None) -> List[PowerOutput]:
        """
        Calculate power output for time series

        Args:
            time_series: List of time points
            irradiance_series: List of irradiance values (W/m²)
            temperature_series: List of temperatures (K)
            degradation_factor: Overall degradation factor
            eclipse_fractions: Optional eclipse fractions

        Returns:
            List of power output data
        """
        if len(time_series) != len(irradiance_series) or len(time_series) != len(temperature_series):
            raise ValueError("Input arrays must have the same length")

        if eclipse_fractions is None:
            eclipse_fractions = [0.0] * len(time_series)

        power_series = []
        for i, time in enumerate(time_series):
            power_output = self.calculate_power_output(
                time,
                irradiance_series[i],
                temperature_series[i],
                degradation_factor,
                eclipse_fractions[i]
            )
            power_series.append(power_output)

        self.power_history.extend(power_series)
        return power_series

    def calculate_energy_output(self, power_series: List[PowerOutput]) -> float:
        """
        Calculate total energy output from power series

        Args:
            power_series: List of power output data

        Returns:
            Total energy output in Watt-hours
        """
        if len(power_series) < 2:
            return 0.0

        total_energy = 0.0
        for i in range(1, len(power_series)):
            dt = (power_series[i].time - power_series[i-1].time).total_seconds() / 3600.0  # hours
            avg_power = (power_series[i].power_watts + power_series[i-1].power_watts) / 2.0
            total_energy += avg_power * dt

        return total_energy

    def apply_degradation_to_parameters(self, radiation_damage_model: RadiationDamageModel) -> SolarCellParameters:
        """
        Apply degradation effects to cell parameters

        Args:
            radiation_damage_model: Radiation damage model with current state

        Returns:
            Degraded cell parameters
        """
        degraded_params = SolarCellParameters(
            short_circuit_current=self.cell_params.short_circuit_current,
            open_circuit_voltage=self.cell_params.open_circuit_voltage,
            max_power_current=self.cell_params.max_power_current,
            max_power_voltage=self.cell_params.max_power_voltage,
            fill_factor=self.cell_params.fill_factor,
            efficiency=self.cell_params.efficiency,
            series_resistance=self.cell_params.series_resistance,
            shunt_resistance=self.cell_params.shunt_resistance,
            ideality_factor=self.cell_params.ideality_factor,
            saturation_current=self.cell_params.saturation_current
        )

        # Apply radiation damage
        damage_state = radiation_damage_model.damage_state
        efficiency_factor = damage_state.power_efficiency_factor

        # Reduce current and voltage
        degraded_params.short_circuit_current *= efficiency_factor
        degraded_params.open_circuit_voltage *= efficiency_factor
        degraded_params.max_power_current *= efficiency_factor
        degraded_params.max_power_voltage *= efficiency_factor
        degraded_params.efficiency *= efficiency_factor

        # Increase series resistance
        degraded_params.series_resistance *= (1.0 + damage_state.series_resistance_increase / 100.0)

        # Reduce fill factor
        degraded_params.fill_factor *= efficiency_factor

        return degraded_params

    def get_power_statistics(self, power_series: List[PowerOutput]) -> Dict:
        """
        Calculate comprehensive power statistics

        Args:
            power_series: List of power output data

        Returns:
            Dictionary with power statistics
        """
        if not power_series:
            return {}

        powers = [p.power_watts for p in power_series]
        efficiencies = [p.efficiency for p in power_series if p.efficiency > 0]
        temperatures = [p.temperature_K for p in power_series]

        # Time-based statistics
        total_duration = (power_series[-1].time - power_series[0].time).total_seconds() / 3600.0
        total_energy = self.calculate_energy_output(power_series)
        avg_power = np.mean(powers)
        peak_power = max(powers)
        min_power = min(powers)

        # Efficiency statistics
        avg_efficiency = np.mean(efficiencies) if efficiencies else 0.0
        peak_efficiency = max(efficiencies) if efficiencies else 0.0

        # Temperature statistics
        avg_temp = np.mean(temperatures)
        max_temp = max(temperatures)
        min_temp = min(temperatures)

        # Power quality metrics
        power_variance = np.var(powers)
        power_cv = np.std(powers) / avg_power if avg_power > 0 else 0.0

        # Eclipse statistics
        eclipse_times = sum(1 for p in power_series if p.eclipse_fraction > 0.5)
        eclipse_fraction = eclipse_times / len(power_series)

        return {
            'performance': {
                'peak_power_W': peak_power,
                'average_power_W': avg_power,
                'minimum_power_W': min_power,
                'total_energy_Wh': total_energy,
                'energy_per_day_Wh': total_energy * 24.0 / total_duration if total_duration > 0 else 0.0,
                'power_variance': power_variance,
                'power_coefficient_of_variation': power_cv
            },
            'efficiency': {
                'peak_efficiency': peak_efficiency,
                'average_efficiency': avg_efficiency,
                'efficiency_ratio': avg_efficiency / self.cell_params.efficiency if self.cell_params.efficiency > 0 else 0.0
            },
            'thermal': {
                'average_temperature_K': avg_temp,
                'maximum_temperature_K': max_temp,
                'minimum_temperature_K': min_temp,
                'temperature_range_K': max_temp - min_temp,
                'average_temperature_C': avg_temp - 273.15
            },
            'eclipse': {
                'eclipse_time_fraction': eclipse_fraction,
                'eclipse_hours': eclipse_fraction * total_duration
            },
            'panel_info': {
                'area_m2': self.panel_area,
                'cell_type': self.cell_type,
                'rated_efficiency': self.cell_params.efficiency
            }
        }

    def get_power_summary_at_time(self, time: datetime,
                                irradiance_wm2: float,
                                temperature_K: float,
                                degradation_factor: float = 1.0,
                                eclipse_fraction: float = 0.0) -> Dict:
        """
        Get detailed power summary at specific time

        Args:
            time: Time for calculation
            irradiance_wm2: Solar irradiance (W/m²)
            temperature_K: Cell temperature (K)
            degradation_factor: Overall degradation factor
            eclipse_fraction: Eclipse shadow fraction

        Returns:
            Detailed power analysis dictionary
        """
        # Calculate power output
        power_output = self.calculate_power_output(
            time, irradiance_wm2, temperature_K, degradation_factor, eclipse_fraction
        )

        # Calculate I-V curve
        iv_curve = self.calculate_iv_curve(
            irradiance_wm2 * (1.0 - eclipse_fraction), temperature_K, degradation_factor
        )

        return {
            'time': time.isoformat(),
            'conditions': {
                'irradiance_W_m2': irradiance_wm2,
                'effective_irradiance_W_m2': irradiance_wm2 * (1.0 - eclipse_fraction),
                'temperature_K': temperature_K,
                'temperature_C': temperature_K - 273.15,
                'eclipse_fraction': eclipse_fraction,
                'degradation_factor': degradation_factor
            },
            'power_output': {
                'total_power_W': power_output.power_watts,
                'power_density_W_m2': power_output.power_density_wm2,
                'voltage_V': power_output.voltage_volts,
                'current_A': power_output.current_amps,
                'efficiency': power_output.efficiency,
                'efficiency_percent': power_output.efficiency * 100.0
            },
            'iv_curve': {
                'open_circuit_voltage_V': iv_curve.voc,
                'short_circuit_current_A_m2': iv_curve.isc,
                'max_power_voltage_V': iv_curve.vmp,
                'max_power_current_A_m2': iv_curve.imp,
                'max_power_W_m2': iv_curve.pmp,
                'fill_factor': iv_curve.ff,
                'fill_factor_percent': iv_curve.ff * 100.0
            },
            'performance_metrics': {
                'power_per_rated_area': power_output.power_watts / self.panel_area,
                'efficiency_of_rated': power_output.efficiency / self.cell_params.efficiency,
                'temperature_loss_percent': max(0, (self.cell_params.efficiency - power_output.efficiency) / self.cell_params.efficiency * 100.0),
                'degradation_loss_percent': (1.0 - degradation_factor) * 100.0,
                'eclipse_loss_percent': eclipse_fraction * 100.0
            }
        }