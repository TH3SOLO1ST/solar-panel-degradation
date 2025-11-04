"""
Thermal Analysis Module

This module calculates orbital temperature profiles, thermal cycling effects,
and heat transfer for solar panels in space. It handles solar heating, radiative
cooling, eclipse cooling, and thermal stress analysis.

References:
- "Spacecraft Thermal Control Handbook" by Gilmore
- "Fundamentals of Heat and Mass Transfer" by Incropera & DeWitt
- NASA Thermal Engineering Handbook
- ESA Thermal Control Guidelines
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from ..orbital.orbit_propagator import OrbitPropagator, OrbitalState
from ..orbital.eclipse_calculator import EclipseCalculator

try:
    from scipy.integrate import solve_ivp
except ImportError:
    raise ImportError("SciPy required for thermal analysis. Install with: pip install scipy")


@dataclass
class ThermalProperties:
    """Thermal properties of solar panel materials"""
    specific_heat: float        # J/(kg·K)
    thermal_conductivity: float  # W/(m·K)
    density: float              # kg/m³
    emissivity: float           # Surface emissivity (0-1)
    absorptivity: float         # Solar absorptivity (0-1)
    thickness: float            # Panel thickness (m)


@dataclass
class ThermalState:
    """Thermal state of solar panel at a given time"""
    time: datetime
    temperature: float          # Panel temperature (K)
    heat_flux_solar: float      # Solar heat flux (W/m²)
    heat_flux_albedo: float     # Earth albedo heat flux (W/m²)
    heat_flux_earth_ir: float   # Earth IR heat flux (W/m²)
    heat_flux_radiated: float   # Radiated heat flux (W/m²)
    net_heat_flux: float        # Net heat flux (W/m²)
    eclipse_status: bool        # True if in eclipse
    temperature_gradient: float # Temperature gradient (K/m)


@dataclass
class ThermalCycle:
    """Thermal cycle event information"""
    start_time: datetime
    end_time: datetime
    min_temperature: float      # Minimum temperature (K)
    max_temperature: float      # Maximum temperature (K)
    temperature_swing: float    # Temperature swing ΔT (K)
    cycle_duration: float       # Cycle duration (hours)
    heating_rate: float         # Heating rate (K/hour)
    cooling_rate: float         # Cooling rate (K/hour)


class ThermalAnalysis:
    """
    Comprehensive thermal analysis for solar panels in orbit.

    Features:
    - Solar heating calculation
    - Radiative cooling modeling
    - Eclipse thermal effects
    - Earth albedo and IR radiation
    - Thermal cycling analysis
    - Temperature gradient calculation
    - Phase change material integration
    """

    # Physical constants
    STEFAN_BOLTZMANN = 5.67e-8      # W/(m²·K⁴)
    SOLAR_CONSTANT = 1361           # W/m² at 1 AU
    EARTH_RADIUS = 6378.137         # km
    SPACE_TEMP = 3.0                # K (deep space temperature)
    SUN_ANGULAR_RADIUS = 0.266      # degrees

    # Earth thermal properties
    EARTH_ALBEDO = 0.3               # Average Earth albedo
    EARTH_IR_INTENSITY = 237         # W/m² (Earth IR radiation)
    EARTH_EMISSIVITY = 0.95          # Earth emissivity

    # Default solar panel properties
    DEFAULT_THERMAL_PROPERTIES = ThermalProperties(
        specific_heat=900,           # J/(kg·K) - typical for silicon solar cells
        thermal_conductivity=150,    # W/(m·K) - aluminum substrate
        density=2700,                # kg/m³ - aluminum
        emissivity=0.85,             # Surface emissivity
        absorptivity=0.92,           # Solar absorptivity
        thickness=0.005              # 5mm thickness
    )

    def __init__(self, orbit_propagator: OrbitPropagator,
                 eclipse_calculator: EclipseCalculator,
                 thermal_properties: Optional[ThermalProperties] = None,
                 use_earth_radiation: bool = True):
        """
        Initialize thermal analysis

        Args:
            orbit_propagator: Configured orbit propagator
            eclipse_calculator: Configured eclipse calculator
            thermal_properties: Solar panel thermal properties
            use_earth_radiation: Include Earth albedo and IR radiation
        """
        self.propagator = orbit_propagator
        self.eclipse_calc = eclipse_calculator
        self.use_earth_radiation = use_earth_radiation

        # Set thermal properties
        if thermal_properties:
            self.thermal_props = thermal_properties
        else:
            self.thermal_props = self.DEFAULT_THERMAL_PROPERTIES

        # Initialize thermal state
        self.thermal_history: List[ThermalState] = []
        self.thermal_cycles: List[ThermalCycle] = []

        # Thermal model parameters
        self.panel_mass_per_area = (self.thermal_props.density *
                                   self.thermal_props.thickness)  # kg/m²

    def calculate_solar_heat_flux(self, time: datetime, panel_normal: np.ndarray) -> float:
        """
        Calculate solar heat flux on solar panel

        Args:
            time: Time for calculation
            panel_normal: Normal vector of solar panel (unit vector)

        Returns:
            Solar heat flux (W/m²)
        """
        # Check if satellite is in eclipse
        eclipse_event = self.eclipse_calc.calculate_eclipse_at_time(time)
        if eclipse_event.eclipse_type != "none":
            return 0.0  # No solar flux in eclipse

        # Get satellite position
        state = self.propagator.propagate(time)

        # Calculate solar incidence angle
        sun_vector = self.eclipse_calc._get_sun_position(time) - state.position
        sun_vector = sun_vector / np.linalg.norm(sun_vector)

        # Calculate angle between panel normal and sun direction
        cos_angle = np.dot(panel_normal, sun_vector)
        cos_angle = max(0.0, cos_angle)  # Only positive flux (one-sided panel)

        # Account for distance from Sun (varies with Earth's orbit)
        sun_distance_km = np.linalg.norm(self.eclipse_calc._get_sun_position(time))
        distance_factor = (self.eclipse_calc.EARTH_SUN_DISTANCE / sun_distance_km) ** 2

        # Calculate solar heat flux
        solar_flux = (self.SOLAR_CONSTANT * distance_factor *
                     self.thermal_props.absorptivity *
                     cos_angle *
                     (1.0 - eclipse_event.max_eclipse_fraction))  # Partial eclipse reduction

        return solar_flux

    def calculate_earth_heat_fluxes(self, time: datetime) -> Tuple[float, float]:
        """
        Calculate Earth albedo and IR heat fluxes

        Args:
            time: Time for calculation

        Returns:
            Tuple of (albedo_flux, ir_flux) in W/m²
        """
        if not self.use_earth_radiation:
            return 0.0, 0.0

        # Get satellite position
        state = self.propagator.propagate(time)
        altitude_km = state.altitude

        # Calculate Earth view factor (simplified)
        earth_angular_radius = np.arcsin(self.EARTH_RADIUS / (self.EARTH_RADIUS + altitude_km))
        view_factor = 0.5 * (1 - np.cos(earth_angular_radius))

        # Albedo flux (reflected solar radiation)
        # Simplified model - assumes uniform Earth illumination
        albedo_flux = (self.SOLAR_CONSTANT * self.EARTH_ALBEDO *
                      self.thermal_props.absorptivity * view_factor)

        # Earth IR flux (thermal radiation from Earth)
        earth_ir_flux = (self.EARTH_IR_INTENSITY * self.EARTH_EMISSIVITY *
                        self.thermal_props.emissivity * view_factor)

        return albedo_flux, earth_ir_flux

    def calculate_radiative_cooling(self, temperature_K: float) -> float:
        """
        Calculate radiative heat loss from panel

        Args:
            temperature_K: Panel temperature in Kelvin

        Returns:
            Radiative heat flux (W/m²) - positive means heat loss
        """
        # Stefan-Boltzmann radiation
        radiated_flux = (self.thermal_props.emissivity *
                        self.STEFAN_BOLTZMANN *
                        (temperature_K**4 - self.SPACE_TEMP**4))

        return radiated_flux

    def solve_thermal_equation(self, initial_temp_K: float, time_span_hours: float,
                             time_step_hours: float, panel_normal: np.ndarray,
                             start_time: datetime) -> List[ThermalState]:
        """
        Solve thermal differential equation over time span

        Args:
            initial_temp_K: Initial temperature (K)
            time_span_hours: Duration to solve (hours)
            time_step_hours: Time step for solution (hours)
            panel_normal: Panel normal vector
            start_time: Start time for analysis

        Returns:
            List of ThermalState objects over time
        """
        def thermal_derivative(t, y):
            """Thermal differential equation: m*c*dT/dt = Q_in - Q_out"""
            temperature = y[0]
            current_time = start_time + timedelta(hours=t)

            # Calculate heat fluxes
            solar_flux = self.calculate_solar_heat_flux(current_time, panel_normal)
            albedo_flux, ir_flux = self.calculate_earth_heat_fluxes(current_time)
            radiated_flux = self.calculate_radiative_cooling(temperature)

            # Net heat flux
            net_flux = solar_flux + albedo_flux + ir_flux - radiated_flux

            # Temperature derivative: dT/dt = Q_net / (m*c)
            dT_dt = net_flux / (self.panel_mass_per_area * self.thermal_props.specific_heat)

            return [dT_dt]

        # Time points
        t_span = (0, time_span_hours)
        t_eval = np.arange(0, time_span_hours + time_step_hours, time_step_hours)

        # Solve ODE
        solution = solve_ivp(
            thermal_derivative,
            t_span,
            [initial_temp_K],
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )

        # Process results
        thermal_states = []
        for i, t in enumerate(solution.t):
            current_time = start_time + timedelta(hours=t)
            temperature = solution.y[0, i]

            # Calculate heat fluxes for this state
            solar_flux = self.calculate_solar_heat_flux(current_time, panel_normal)
            albedo_flux, ir_flux = self.calculate_earth_heat_fluxes(current_time)
            radiated_flux = self.calculate_radiative_cooling(temperature)
            net_flux = solar_flux + albedo_flux + ir_flux - radiated_flux

            # Check eclipse status
            eclipse_event = self.eclipse_calc.calculate_eclipse_at_time(current_time)
            eclipse_status = eclipse_event.eclipse_type != "none"

            # Estimate temperature gradient (simplified - assume linear through thickness)
            temp_gradient = (net_flux / self.thermal_props.thermal_conductivity)

            state = ThermalState(
                time=current_time,
                temperature=temperature,
                heat_flux_solar=solar_flux,
                heat_flux_albedo=albedo_flux,
                heat_flux_earth_ir=ir_flux,
                heat_flux_radiated=radiated_flux,
                net_heat_flux=net_flux,
                eclipse_status=eclipse_status,
                temperature_gradient=temp_gradient
            )
            thermal_states.append(state)

        return thermal_states

    def analyze_thermal_cycles(self, thermal_states: List[ThermalState],
                             min_temp_swing_K: float = 10.0) -> List[ThermalCycle]:
        """
        Analyze thermal cycles from thermal state history

        Args:
            thermal_states: List of thermal states over time
            min_temp_swing_K: Minimum temperature swing to count as cycle

        Returns:
            List of ThermalCycle objects
        """
        if len(thermal_states) < 2:
            return []

        cycles = []
        current_cycle_start = None
        current_min_temp = float('inf')
        current_max_temp = float('-inf')

        for i, state in enumerate(thermal_states):
            temp = state.temperature

            # Update min/max temps
            current_min_temp = min(current_min_temp, temp)
            current_max_temp = max(current_max_temp, temp)

            # Detect cycle transitions
            # Start of heating (cooling to heating transition)
            if i > 0 and thermal_states[i-1].temperature > temp and current_cycle_start is None:
                current_cycle_start = state.time
                current_min_temp = temp
                current_max_temp = temp

            # End of cycle (heating to cooling transition with significant temperature swing)
            elif (i > 0 and thermal_states[i-1].temperature < temp and
                  current_cycle_start is not None and
                  (current_max_temp - current_min_temp) > min_temp_swing_K):

                # Calculate heating and cooling rates
                cycle_states = [s for s in thermal_states
                              if current_cycle_start <= s.time <= state.time]

                if len(cycle_states) > 1:
                    # Heating rate (max positive slope)
                    heating_rates = []
                    cooling_rates = []

                    for j in range(1, len(cycle_states)):
                        dt = (cycle_states[j].time - cycle_states[j-1].time).total_seconds() / 3600.0
                        dT = cycle_states[j].temperature - cycle_states[j-1].temperature

                        if dT > 0:
                            heating_rates.append(dT / dt)
                        else:
                            cooling_rates.append(abs(dT / dt))

                    heating_rate = max(heating_rates) if heating_rates else 0.0
                    cooling_rate = max(cooling_rates) if cooling_rates else 0.0
                else:
                    heating_rate = 0.0
                    cooling_rate = 0.0

                # Create thermal cycle
                cycle = ThermalCycle(
                    start_time=current_cycle_start,
                    end_time=state.time,
                    min_temperature=current_min_temp,
                    max_temperature=current_max_temp,
                    temperature_swing=current_max_temp - current_min_temp,
                    cycle_duration=(state.time - current_cycle_start).total_seconds() / 3600.0,
                    heating_rate=heating_rate,
                    cooling_rate=cooling_rate
                )
                cycles.append(cycle)

                # Reset for next cycle
                current_cycle_start = None
                current_min_temp = float('inf')
                current_max_temp = float('-inf')

        return cycles

    def calculate_steady_state_temperature(self, time: datetime,
                                         panel_normal: np.ndarray,
                                         initial_guess_K: float = 300.0) -> float:
        """
        Calculate steady-state temperature at given conditions

        Args:
            time: Time for calculation
            panel_normal: Panel normal vector
            initial_guess_K: Initial temperature guess for iteration

        Returns:
            Steady-state temperature (K)
        """
        def heat_balance_error(T):
            """Heat balance equation error"""
            solar_flux = self.calculate_solar_heat_flux(time, panel_normal)
            albedo_flux, ir_flux = self.calculate_earth_heat_fluxes(time)
            radiated_flux = self.calculate_radiative_cooling(T)

            net_flux = solar_flux + albedo_flux + ir_flux - radiated_flux
            return net_flux

        # Use Newton-Raphson iteration to find steady-state temperature
        T = initial_guess_K
        tolerance = 0.1  # K
        max_iterations = 50

        for _ in range(max_iterations):
            # Calculate error and derivative
            error = heat_balance_error(T)

            # Numerical derivative
            dT = 0.1
            d_error = (heat_balance_error(T + dT) - heat_balance_error(T - dT)) / (2 * dT)

            if abs(d_error) < 1e-10:
                break

            # Newton-Raphson update
            T_new = T - error / d_error

            if abs(T_new - T) < tolerance:
                return T_new

            T = max(100.0, min(500.0, T_new))  # Keep temperature in reasonable range

        return T

    def get_thermal_statistics(self, thermal_states: List[ThermalState]) -> Dict:
        """
        Calculate comprehensive thermal statistics

        Args:
            thermal_states: List of thermal states

        Returns:
            Dictionary with thermal statistics
        """
        if not thermal_states:
            return {}

        temperatures = [state.temperature for state in thermal_states]
        eclipse_times = [state.time for state in thermal_states if state.eclipse_status]
        eclipse_temps = [state.temperature for state in thermal_states if state.eclipse_status]
        sunlit_temps = [state.temperature for state in thermal_states if not state.eclipse_status]

        # Basic statistics
        stats = {
            'temperature': {
                'min_K': min(temperatures),
                'max_K': max(temperatures),
                'mean_K': np.mean(temperatures),
                'std_K': np.std(temperatures),
                'range_K': max(temperatures) - min(temperatures)
            },
            'eclipse': {
                'time_fraction': len(eclipse_times) / len(thermal_states),
                'min_temp_K': min(eclipse_temps) if eclipse_temps else None,
                'max_temp_K': max(eclipse_temps) if eclipse_temps else None,
                'mean_temp_K': np.mean(eclipse_temps) if eclipse_temps else None
            },
            'sunlit': {
                'min_temp_K': min(sunlit_temps) if sunlit_temps else None,
                'max_temp_K': max(sunlit_temps) if sunlit_temps else None,
                'mean_temp_K': np.mean(sunlit_temps) if sunlit_temps else None
            }
        }

        # Heat flux statistics
        solar_fluxes = [state.heat_flux_solar for state in thermal_states]
        albedo_fluxes = [state.heat_flux_albedo for state in thermal_states]
        ir_fluxes = [state.heat_flux_earth_ir for state in thermal_states]
        radiated_fluxes = [state.heat_flux_radiated for state in thermal_states]
        net_fluxes = [state.net_heat_flux for state in thermal_states]

        stats['heat_fluxes'] = {
            'solar': {
                'mean_W_m2': np.mean(solar_fluxes),
                'max_W_m2': max(solar_fluxes),
                'min_W_m2': min(solar_fluxes)
            },
            'albedo': {
                'mean_W_m2': np.mean(albedo_fluxes),
                'max_W_m2': max(albedo_fluxes),
                'min_W_m2': min(albedo_fluxes)
            },
            'earth_ir': {
                'mean_W_m2': np.mean(ir_fluxes),
                'max_W_m2': max(ir_fluxes),
                'min_W_m2': min(ir_fluxes)
            },
            'radiated': {
                'mean_W_m2': np.mean(radiated_fluxes),
                'max_W_m2': max(radiated_fluxes),
                'min_W_m2': min(radiated_fluxes)
            },
            'net': {
                'mean_W_m2': np.mean(net_fluxes),
                'max_W_m2': max(net_fluxes),
                'min_W_m2': min(net_fluxes)
            }
        }

        # Convert to Celsius for user convenience
        stats['temperature']['min_C'] = stats['temperature']['min_K'] - 273.15
        stats['temperature']['max_C'] = stats['temperature']['max_K'] - 273.15
        stats['temperature']['mean_C'] = stats['temperature']['mean_K'] - 273.15

        if eclipse_temps:
            stats['eclipse']['min_temp_C'] = stats['eclipse']['min_temp_K'] - 273.15
            stats['eclipse']['max_temp_C'] = stats['eclipse']['max_temp_K'] - 273.15
            stats['eclipse']['mean_temp_C'] = stats['eclipse']['mean_temp_K'] - 273.15

        if sunlit_temps:
            stats['sunlit']['min_temp_C'] = stats['sunlit']['min_temp_K'] - 273.15
            stats['sunlit']['max_temp_C'] = stats['sunlit']['max_temp_K'] - 273.15
            stats['sunlit']['mean_temp_C'] = stats['sunlit']['mean_temp_K'] - 273.15

        return stats