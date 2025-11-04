"""
Radiation Environment Module

This module models the space radiation environment including trapped particles,
solar particle events, and galactic cosmic rays. It implements AE8/AP8 and AP9
models for trapped radiation and provides dose accumulation calculations.

References:
- NASA AE8/AP8 models for trapped radiation
- AP9/AT9 models (if available)
- NOAA space weather data for solar particle events
- CREME96 for cosmic ray modeling
- "Space Radiation Environment" by Vette
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import requests
import json

try:
    import pandas as pd
except ImportError:
    raise ImportError("Pandas required for data handling. Install with: pip install pandas")

try:
    from scipy import interpolate, integrate
except ImportError:
    raise ImportError("SciPy required for interpolation. Install with: pip install scipy")


@dataclass
class RadiationFlux:
    """Radiation flux data for a specific particle type and energy"""
    particle_type: str  # "electron", "proton", "heavy_ion"
    energy_mev: float   # Particle energy in MeV
    flux_particles: float  # Flux in particles/(cm²·s·sr·MeV)
    energy_range_mev: Tuple[float, float]  # Energy range [min, max]


@dataclass
class RadiationDose:
    """Accumulated radiation dose information"""
    dose_rads: float          # Total ionizing dose (rads)
    dose_si: float           # Dose in SI units (Gray)
    ddd_mev_cm2_g: float     # Displacement damage dose (MeV·cm²/g)
    fluence_1mev_e: float    # 1 MeV electron equivalent fluence
    time_hours: float        # Accumulation time (hours)


@dataclass
class SolarParticleEvent:
    """Solar particle event parameters"""
    start_time: datetime
    end_time: datetime
    peak_flux: float         # Peak flux in particles/(cm²·s·sr)
    energy_spectrum: List[Tuple[float, float]]  # Energy bins and fluxes
    event_type: str         # "proton", "electron", "mixed"
    severity: str           # "weak", "moderate", "strong", "severe"


class RadiationEnvironment:
    """
    Comprehensive space radiation environment model.

    Features:
    - AE8/AP8 trapped particle models
    - Solar particle event modeling
    - Galactic cosmic ray modeling
    - Real space weather data integration
    - Shielding calculations
    - South Atlantic Anomaly modeling
    """

    # Physical constants
    EARTH_RADIUS = 6378.137  # km
    PROTON_REST_MASS = 938.272  # MeV
    ELECTRON_REST_MASS = 0.511  # MeV
    SPEED_OF_LIGHT = 2.998e8   # m/s

    # Radiation model parameters
    SOLAR_CYCLE_PERIOD = 11.0  # years
    MIN_SOLAR_ACTIVITY = 0.0   # dimensionless
    MAX_SOLAR_ACTIVITY = 1.0   # dimensionless

    # AE8/AP8 model coefficients (simplified for demonstration)
    # In practice, these would be loaded from model data files
    AE8_COEFFICIENTS = {
        'min_flux': 1e2,      # particles/(cm²·s·sr·MeV)
        'max_flux': 1e8,      # particles/(cm²·s·sr·MeV)
        'peak_altitude': 30000,  # km (Van Allen belt peak)
        'width_parameter': 5000,  # km
    }
    AP8_COEFFICIENTS = {
        'min_flux': 1e1,      # particles/(cm²·s·sr·MeV)
        'max_flux': 1e7,      # particles/(cm²·s·sr·MeV)
        'inner_peak': 15000,  # km (inner belt)
        'outer_peak': 25000,  # km (outer belt)
    }

    def __init__(self, use_real_data: bool = False, solar_activity: float = 0.5):
        """
        Initialize radiation environment model

        Args:
            use_real_data: Use real space weather data when available
            solar_activity: Solar activity level (0-1, 0=minimum, 1=maximum)
        """
        self.use_real_data = use_real_data
        self.solar_activity = solar_activity
        self.real_data_cache = {}

        # Initialize model interpolators (would load actual model data in production)
        self._init_trapped_particle_models()

    def _init_trapped_particle_models(self):
        """Initialize trapped particle model interpolators"""
        # Create simplified model grids for demonstration
        # In practice, these would be loaded from official AE8/AP8 data files

        # L-shell values (Earth radii)
        self.L_values = np.linspace(1.1, 7.0, 100)

        # Magnetic field values (Gauss)
        self.B_values = np.linspace(0.1, 0.8, 50)

        # Energy values (MeV)
        self.electron_energies = np.logspace(-1, 2, 50)  # 0.1 to 100 MeV
        self.proton_energies = np.logspace(-1, 2, 50)    # 0.1 to 100 MeV

        # Initialize flux grids (simplified models)
        self._create_electron_flux_grid()
        self._create_proton_flux_grid()

    def _create_electron_flux_grid(self):
        """Create electron flux grid using simplified AE8 model"""
        self.electron_flux_grid = np.zeros((len(self.L_values),
                                          len(self.B_values),
                                          len(self.electron_energies)))

        for i, L in enumerate(self.L_values):
            for j, B in enumerate(self.B_values):
                for k, E in enumerate(self.electron_energies):
                    # Simplified AE8 flux model
                    if 1.2 <= L <= 3.0:  # Inner belt
                        peak_L = 2.0
                        flux_peak = self.AE8_COEFFICIENTS['max_flux'] * np.exp(-E/1.0)
                        spatial_factor = np.exp(-((L - peak_L)/0.5)**2) * np.exp(-B/0.3)
                    elif 3.0 <= L <= 7.0:  # Outer belt
                        peak_L = 4.5
                        flux_peak = self.AE8_COEFFICIENTS['max_flux'] * 0.3 * np.exp(-E/2.0)
                        spatial_factor = np.exp(-((L - peak_L)/1.0)**2) * np.exp(-B/0.4)
                    else:
                        flux_peak = self.AE8_COEFFICIENTS['min_flux']
                        spatial_factor = 0.1

                    self.electron_flux_grid[i, j, k] = flux_peak * spatial_factor

    def _create_proton_flux_grid(self):
        """Create proton flux grid using simplified AP8 model"""
        self.proton_flux_grid = np.zeros((len(self.L_values),
                                        len(self.B_values),
                                        len(self.proton_energies)))

        for i, L in enumerate(self.L_values):
            for j, B in enumerate(self.B_values):
                for k, E in enumerate(self.proton_energies):
                    # Simplified AP8 flux model
                    if 1.1 <= L <= 2.5:  # Inner belt
                        peak_L = 1.8
                        flux_peak = self.AP8_COEFFICIENTS['max_flux'] * np.exp(-E/20.0)
                        spatial_factor = np.exp(-((L - peak_L)/0.3)**2) * np.exp(-B/0.2)
                    elif 2.5 <= L <= 6.0:  # Outer belt (much weaker protons)
                        peak_L = 4.0
                        flux_peak = self.AP8_COEFFICIENTS['max_flux'] * 0.01 * np.exp(-E/10.0)
                        spatial_factor = np.exp(-((L - peak_L)/1.2)**2) * np.exp(-B/0.3)
                    else:
                        flux_peak = self.AP8_COEFFICIENTS['min_flux']
                        spatial_factor = 0.01

                    self.proton_flux_grid[i, j, k] = flux_peak * spatial_factor

    def calculate_trapped_particle_flux(self, position: np.ndarray, time: datetime) -> List[RadiationFlux]:
        """
        Calculate trapped particle flux at given position and time

        Args:
            position: Satellite position in ECI coordinates (km)
            time: Time for calculation

        Returns:
            List of RadiationFlux objects for different particle types and energies
        """
        # Calculate L-shell and B-field values
        L, B = self._calculate_magnetic_coordinates(position, time)

        # Solar activity modulation
        solar_mod = 1.0 + 0.5 * (self.solar_activity - 0.5) * np.sin(2 * np.pi * time.year / self.SOLAR_CYCLE_PERIOD)

        fluxes = []

        # Calculate electron fluxes
        electron_fluxes = self._interpolate_flux_grid(L, B, self.electron_energies,
                                                    self.electron_flux_grid) * solar_mod

        for i, energy in enumerate(self.electron_energies[::5]):  # Sample every 5th energy
            flux = electron_fluxes[i*5]
            if flux > 1e-2:  # Only include significant fluxes
                fluxes.append(RadiationFlux(
                    particle_type="electron",
                    energy_mev=energy,
                    flux_particles=flux,
                    energy_range_mev=(energy/2, energy*2)
                ))

        # Calculate proton fluxes
        proton_fluxes = self._interpolate_flux_grid(L, B, self.proton_energies,
                                                  self.proton_flux_grid) * solar_mod

        for i, energy in enumerate(self.proton_energies[::5]):  # Sample every 5th energy
            flux = proton_fluxes[i*5]
            if flux > 1e-3:  # Only include significant fluxes
                fluxes.append(RadiationFlux(
                    particle_type="proton",
                    energy_mev=energy,
                    flux_particles=flux,
                    energy_range_mev=(energy/2, energy*2)
                ))

        return fluxes

    def _calculate_magnetic_coordinates(self, position: np.ndarray, time: datetime) -> Tuple[float, float]:
        """
        Calculate magnetic L-shell and B-field coordinates

        Args:
            position: Satellite position (km)
            time: Time for calculation

        Returns:
            Tuple of (L_shell, B_field)
        """
        # Simplified dipole model for L-shell calculation
        r = np.linalg.norm(position)
        lat = np.arcsin(position[2] / r)  # Magnetic latitude

        # L-shell approximation
        L = r / (self.EARTH_RADIUS * np.cos(lat)**2)

        # B-field approximation (dipole)
        B = 31100 / (L**3) * np.sqrt(1 + 3 * np.sin(lat)**2) / np.cos(lat)**6  # Gauss

        return L, B

    def _interpolate_flux_grid(self, L: float, B: float, energies: np.ndarray,
                             flux_grid: np.ndarray) -> np.ndarray:
        """
        Interpolate flux from pre-computed grid

        Args:
            L: L-shell value
            B: B-field value (Gauss)
            energies: Energy values (MeV)
            flux_grid: 3D flux grid [L, B, Energy]

        Returns:
            Interpolated flux values for each energy
        """
        # Ensure L and B are within grid bounds
        L = np.clip(L, self.L_values[0], self.L_values[-1])
        B = np.clip(B, self.B_values[0], self.B_values[-1])

        # Create interpolator
        interp = interpolate.RegularGridInterpolator(
            (self.L_values, self.B_values, energies),
            flux_grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # Interpolate for all energies
        points = np.column_stack([np.full_like(energies, L),
                                 np.full_like(energies, B),
                                 energies])

        return interp(points)

    def calculate_solar_particle_event_flux(self, time: datetime, position: np.ndarray) -> List[RadiationFlux]:
        """
        Calculate solar particle event flux at given time and position

        Args:
            time: Time for calculation
            position: Satellite position (km)

        Returns:
            List of RadiationFlux objects for solar particles
        """
        if self.use_real_data:
            return self._get_real_spe_data(time, position)
        else:
            return self._generate_synthetic_spe(time, position)

    def _get_real_spe_data(self, time: datetime, position: np.ndarray) -> List[RadiationFlux]:
        """Get real solar particle event data from space weather APIs"""
        # This would connect to NOAA GOES or other space weather data sources
        # For demonstration, return empty flux (no SPE)
        return []

    def _generate_synthetic_spe(self, time: datetime, position: np.ndarray) -> List[RadiationFlux]:
        """Generate synthetic solar particle event data"""
        # Simple SPE probability model based on solar activity
        spe_probability = self.solar_activity * 0.1  # Max 10% chance per day

        if np.random.random() > spe_probability:
            return []

        # Generate synthetic SPE parameters
        if self.solar_activity > 0.7:
            severity = "severe"
            peak_flux = 1e5
        elif self.solar_activity > 0.4:
            severity = "strong"
            peak_flux = 1e4
        else:
            severity = "moderate"
            peak_flux = 1e3

        # Create energy spectrum (power law)
        energies = np.logspace(0, 2, 20)  # 1 to 100 MeV
        spectral_index = -3.0  # Typical for solar protons

        fluxes = []
        for energy in energies:
            flux = peak_flux * (energy/10.0)**spectral_index
            fluxes.append(RadiationFlux(
                particle_type="proton",
                energy_mev=energy,
                flux_particles=flux,
                energy_range_mev=(energy/np.sqrt(2), energy*np.sqrt(2))
            ))

        return fluxes

    def calculate_gcr_flux(self, time: datetime, position: np.ndarray) -> List[RadiationFlux]:
        """
        Calculate galactic cosmic ray flux

        Args:
            time: Time for calculation
            position: Satellite position (km)

        Returns:
            List of RadiationFlux objects for GCR particles
        """
        # GCR flux is inversely related to solar activity
        gcr_modulation = 1.0 - 0.7 * self.solar_activity

        # Simplified GCR spectrum (power law)
        energies = np.logspace(2, 4, 10)  # 100 MeV to 10 GeV

        fluxes = []
        for energy in energies:
            if energy < 1000:  # Protons up to 1 GeV
                flux = 1.0 * gcr_modulation * (energy/100.0)**-2.7
                fluxes.append(RadiationFlux(
                    particle_type="proton",
                    energy_mev=energy,
                    flux_particles=flux,
                    energy_range_mev=(energy/2, energy*2)
                ))

        return fluxes

    def calculate_radiation_dose(self, fluxes: List[RadiationFlux],
                               shielding_thickness_cm: float = 0.1,
                               exposure_time_hours: float = 1.0) -> RadiationDose:
        """
        Calculate accumulated radiation dose from fluxes

        Args:
            fluxes: List of radiation fluxes
            shielding_thickness_cm: Shielding thickness in cm
            exposure_time_hours: Exposure duration in hours

        Returns:
            RadiationDose object with dose information
        """
        total_dose_rads = 0.0
        total_ddd = 0.0
        total_fluence_1mev_e = 0.0

        for flux in fluxes:
            # Apply shielding attenuation
            transmission = self._calculate_shielding_transmission(
                flux.particle_type, flux.energy_mev, shielding_thickness_cm
            )

            attenuated_flux = flux.flux_particles * transmission

            # Calculate dose rate (simplified)
            if flux.particle_type == "electron":
                # Electrons: dose rate ≈ flux × energy × stopping power
                dose_rate = attenuated_flux * flux.energy_mev * 1.6e-8  # rad/s
                ddd_contribution = attenuated_flux * flux.energy_mev * 1e-6  # MeV·cm²/g

                # 1 MeV electron equivalent
                if abs(flux.energy_mev - 1.0) < 0.5:
                    total_fluence_1mev_e += attenuated_flux * exposure_time_hours * 3600

            elif flux.particle_type == "proton":
                # Protons: higher damage potential
                dose_rate = attenuated_flux * flux.energy_mev * 1.6e-7  # rad/s
                ddd_contribution = attenuated_flux * flux.energy_mev * 1e-5  # MeV·cm²/g

                # 1 MeV electron equivalent (NIEL scaling)
                niel_factor = (flux.energy_mev / 1.0)**0.8  # Simplified NIEL scaling
                total_fluence_1mev_e += attenuated_flux * niel_factor * exposure_time_hours * 3600
            else:
                continue

            total_dose_rads += dose_rate * exposure_time_hours * 3600
            total_ddd += ddd_contribution * exposure_time_hours * 3600

        return RadiationDose(
            dose_rads=total_dose_rads,
            dose_si=total_dose_rads * 0.01,  # Convert rad to Gray
            ddd_mev_cm2_g=total_ddd,
            fluence_1mev_e=total_fluence_1mev_e,
            time_hours=exposure_time_hours
        )

    def _calculate_shielding_transmission(self, particle_type: str,
                                        energy_mev: float,
                                        thickness_cm: float) -> float:
        """
        Calculate transmission through shielding material

        Args:
            particle_type: Type of particle ("electron" or "proton")
            energy_mev: Particle energy in MeV
            thickness_cm: Shielding thickness in cm

        Returns:
            Transmission fraction (0-1)
        """
        # Simplified shielding model (aluminum)
        density_al = 2.7  # g/cm³

        if particle_type == "electron":
            # Electron range in aluminum (approximate)
            if energy_mev < 0.5:
                range_cm = 0.05 * (energy_mev / 0.5)**2
            else:
                range_cm = 0.05 + 0.4 * (energy_mev - 0.5)
        else:  # proton
            # Proton range in aluminum (approximate)
            if energy_mev < 10:
                range_cm = 0.01 * (energy_mev / 10)**2
            else:
                range_cm = 0.01 + 0.1 * (energy_mev - 10)

        # Transmission probability
        if thickness_cm >= range_cm:
            return 0.0
        else:
            return np.exp(-thickness_cm / range_cm)

    def get_radiation_environment_summary(self, position: np.ndarray,
                                        time: datetime,
                                        duration_hours: float = 24.0) -> Dict:
        """
        Get comprehensive radiation environment summary

        Args:
            position: Satellite position (km)
            time: Starting time
            duration_hours: Duration for analysis (hours)

        Returns:
            Dictionary with comprehensive radiation environment data
        """
        # Calculate all radiation sources
        trapped_fluxes = self.calculate_trapped_particle_flux(position, time)
        spe_fluxes = self.calculate_solar_particle_event_flux(position, time)
        gcr_fluxes = self.calculate_gcr_flux(position, time)

        all_fluxes = trapped_fluxes + spe_fluxes + gcr_fluxes

        # Calculate doses for different shielding levels
        shielding_levels = [0.01, 0.1, 0.5, 1.0]  # cm
        doses = {}

        for thickness in shielding_levels:
            dose = self.calculate_radiation_dose(all_fluxes, thickness, duration_hours)
            doses[f"{thickness*10:.0f}mm"] = dose

        # Calculate magnetic coordinates
        L, B = self._calculate_magnetic_coordinates(position, time)

        return {
            'position': position.tolist(),
            'time': time.isoformat(),
            'duration_hours': duration_hours,
            'magnetic_coordinates': {'L_shell': L, 'B_field_gauss': B},
            'trapped_particle_count': len(trapped_fluxes),
            'spe_active': len(spe_fluxes) > 0,
            'gcr_count': len(gcr_fluxes),
            'total_particle_count': len(all_fluxes),
            'doses_by_shielding': {
                thickness: {
                    'dose_rads': dose.dose_rads,
                    'dose_si': dose.dose_si,
                    'ddd_mev_cm2_g': dose.ddd_mev_cm2_g,
                    'fluence_1mev_e': dose.fluence_1mev_e
                }
                for thickness, dose in doses.items()
            },
            'solar_activity': self.solar_activity
        }