"""
Radiation Environment
=====================

Models the space radiation environment including trapped particles,
solar particle events, and galactic cosmic rays.

This module provides comprehensive radiation environment modeling for
solar panel degradation calculations.

Classes:
    RadiationEnvironment: Main radiation environment class
    TrappedParticleModel: Trapped particle flux modeling
    SolarEventModel: Solar particle event modeling
    CosmicRayModel: Galactic cosmic ray modeling
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RadiationDose:
    """Data structure for radiation dose information"""
    total_ionizing_dose: float  # rads
    non_ionizing_dose: float    # MeV cm²/g (displacement damage)
    electron_fluence: float     # electrons/cm²
    proton_fluence: float       # protons/cm²
    time_points: np.ndarray     # time points in hours
    dose_timeline: np.ndarray   # cumulative dose over time

class TrappedParticleModel:
    """Models trapped radiation belts (Van Allen belts)"""

    def __init__(self, model_type: str = "AP8"):
        """
        Initialize trapped particle model

        Args:
            model_type: Radiation model type ("AP8", "AE8", "AP9")
        """
        self.model_type = model_type
        self.earth_radius = 6371.0  # km

        # Simplified model parameters (would use actual AE8/AP8 data in production)
        self._initialize_model_parameters()

    def _initialize_model_parameters(self):
        """Initialize simplified radiation belt parameters"""
        # Inner radiation belt (protons)
        self.inner_belt = {
            'L_min': 1.2,
            'L_max': 2.0,
            'peak_flux': 1e4,  # protons/cm²/s
            'peak_energy': 10,  # MeV
            'width': 0.3
        }

        # Outer radiation belt (electrons)
        self.outer_belt = {
            'L_min': 3.0,
            'L_max': 7.0,
            'peak_flux': 1e8,  # electrons/cm²/s
            'peak_energy': 0.5,  # MeV
            'width': 1.0
        }

        # South Atlantic Anomaly enhancement
        self.saa_factor = 2.0  # Flux enhancement factor

    def calculate_l_shell(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate McIlwain L-shell parameter for positions

        Args:
            positions: Nx3 array of positions in km

        Returns:
            Array of L-shell values
        """
        # Simplified L-shell calculation
        # L ≈ r / cos²(λ) where r is radial distance and λ is magnetic latitude
        r = np.linalg.norm(positions, axis=1)
        L = r / self.earth_radius
        return L

    def get_trapped_flux(self, positions: np.ndarray, time_hours: np.ndarray,
                        particle_type: str = 'proton') -> np.ndarray:
        """
        Get trapped particle flux for given positions

        Args:
            positions: Nx3 array of positions
            time_hours: Array of time points
            particle_type: 'proton' or 'electron'

        Returns:
            Array of flux values in particles/cm²/s
        """
        L_values = self.calculate_l_shell(positions)
        flux = np.zeros(len(L_values))

        if particle_type == 'proton':
            # Inner belt model
            belt = self.inner_belt
            for i, L in enumerate(L_values):
                if belt['L_min'] <= L <= belt['L_max']:
                    # Gaussian-like profile
                    flux[i] = belt['peak_flux'] * np.exp(-((L - 1.6) / belt['width'])**2)
        elif particle_type == 'electron':
            # Outer belt model
            belt = self.outer_belt
            for i, L in enumerate(L_values):
                if belt['L_min'] <= L <= belt['L_max']:
                    # Broader profile for electrons
                    flux[i] = belt['peak_flux'] * np.exp(-((L - 4.5) / belt['width'])**2)

        return flux

class SolarEventModel:
    """Models solar particle events (solar flares, CMEs)"""

    def __init__(self):
        """Initialize solar event model"""
        # Historical solar event statistics (simplified)
        self.event_rate = 0.5  # events per year on average
        self.event_duration_avg = 24  # hours
        self.event_fluence_avg = 1e9  # protons/cm²
        self.event_energy_avg = 10  # MeV

    def generate_solar_events(self, start_time: datetime, duration_days: float) -> List[Dict]:
        """
        Generate synthetic solar particle events

        Args:
            start_time: Start time for simulation
            duration_days: Duration in days

        Returns:
            List of solar event dictionaries
        """
        events = []
        n_events = np.random.poisson(self.event_rate * duration_days / 365.0)

        for _ in range(n_events):
            # Random event time
            event_time = start_time + timedelta(
                days=np.random.uniform(0, duration_days)
            )

            # Random event properties
            duration = np.random.exponential(self.event_duration_avg)
            fluence = np.random.exponential(self.event_fluence_avg)
            energy = np.random.gamma(2, self.event_energy_avg / 2)

            events.append({
                'start_time': event_time,
                'duration_hours': duration,
                'fluence': fluence,
                'energy_MeV': energy,
                'peak_flux': fluence / (duration * 3600)  # particles/cm²/s
            })

        return events

    def get_event_flux(self, time_hours: np.ndarray, events: List[Dict]) -> np.ndarray:
        """
        Calculate solar event flux over time

        Args:
            time_hours: Array of time points in hours
            events: List of solar event dictionaries

        Returns:
            Array of flux values in particles/cm²/s
        """
        flux = np.zeros(len(time_hours))

        for event in events:
            event_start = (event['start_time'] - datetime(2000, 1, 1)).total_seconds() / 3600
            event_end = event_start + event['duration_hours']

            # Find time points during event
            mask = (time_hours >= event_start) & (time_hours <= event_end)

            # Simple rectangular profile (would be more realistic with time evolution)
            flux[mask] += event['peak_flux']

        return flux

class CosmicRayModel:
    """Models galactic cosmic ray radiation"""

    def __init__(self):
        """Initialize cosmic ray model"""
        # GCR flux parameters (simplified)
        self.base_flux = 1e2  # particles/cm²/s (high energy)
        self.solar_modulation = True  # Include solar cycle effects

    def get_gcr_flux(self, time_hours: np.ndarray) -> np.ndarray:
        """
        Calculate galactic cosmic ray flux

        Args:
            time_hours: Array of time points in hours

        Returns:
            Array of GCR flux values in particles/cm²/s
        """
        flux = np.ones(len(time_hours)) * self.base_flux

        if self.solar_modulation:
            # Simple solar cycle modulation (11-year cycle)
            solar_phase = 2 * np.pi * time_hours / (11 * 365 * 24)
            modulation = 1 - 0.3 * np.sin(solar_phase)
            flux *= modulation

        return flux

class RadiationEnvironment:
    """Main radiation environment class"""

    def __init__(self, model_type: str = "AP8"):
        """
        Initialize radiation environment

        Args:
            model_type: Radiation model type
        """
        self.trapped_model = TrappedParticleModel(model_type)
        self.solar_model = SolarEventModel()
        self.gcr_model = CosmicRayModel()

    def calculate_radiation_dose(self, positions: np.ndarray, time_hours: np.ndarray,
                                shielding_thickness_mm: float = 1.0) -> RadiationDose:
        """
        Calculate cumulative radiation dose for satellite trajectory

        Args:
            positions: Nx3 array of satellite positions
            time_hours: Array of time points
            shielding_thickness_mm: Thickness of shielding material in mm

        Returns:
            RadiationDose object with dose information
        """
        # Get trapped particle fluxes
        proton_flux = self.trapped_model.get_trapped_flux(positions, time_hours, 'proton')
        electron_flux = self.trapped_model.get_trapped_flux(positions, time_hours, 'electron')

        # Generate solar events
        start_time = datetime(2000, 1, 1)  # Reference time
        duration_days = time_hours[-1] / 24.0
        solar_events = self.solar_model.generate_solar_events(start_time, duration_days)

        # Get solar event flux
        solar_flux = self.solar_model.get_event_flux(time_hours, solar_events)

        # Get GCR flux
        gcr_flux = self.gcr_model.get_gcr_flux(time_hours)

        # Calculate cumulative fluences
        time_step_hours = time_hours[1] - time_hours[0] if len(time_hours) > 1 else 1.0
        time_step_seconds = time_step_hours * 3600

        total_proton_fluence = np.sum((proton_flux + solar_flux) * time_step_seconds)
        total_electron_fluence = np.sum(electron_flux * time_step_seconds)

        # Calculate doses
        # Simplified dose calculation (would use more complex NIEL calculations)
        ionizing_dose = self._calculate_ionizing_dose(
            total_proton_fluence, total_electron_fluence, shielding_thickness_mm
        )
        non_ionizing_dose = self._calculate_displacement_damage(
            total_proton_fluence, shielding_thickness_mm
        )

        # Create timeline of cumulative dose
        cumulative_dose = np.zeros(len(time_hours))
        for i in range(1, len(time_hours)):
            dose_increment = self._calculate_step_dose(
                proton_flux[i], electron_flux[i], solar_flux[i], gcr_flux[i],
                time_step_seconds, shielding_thickness_mm
            )
            cumulative_dose[i] = cumulative_dose[i-1] + dose_increment

        return RadiationDose(
            total_ionizing_dose=ionizing_dose,
            non_ionizing_dose=non_ionizing_dose,
            electron_fluence=total_electron_fluence,
            proton_fluence=total_proton_fluence,
            time_points=time_hours,
            dose_timeline=cumulative_dose
        )

    def _calculate_ionizing_dose(self, proton_fluence: float, electron_fluence: float,
                               shielding_mm: float) -> float:
        """
        Calculate ionizing radiation dose

        Args:
            proton_fluence: Proton fluence in protons/cm²
            electron_fluence: Electron fluence in electrons/cm²
            shielding_mm: Shielding thickness in mm

        Returns:
            Ionizing dose in rads
        """
        # Simplified calculation (would use actual stopping power data)
        proton_dose_coefficient = 1e-4  # rads per proton/cm² (1 MeV equivalent)
        electron_dose_coefficient = 1e-7  # rads per electron/cm²

        # Shielding attenuation (exponential)
        shielding_factor = np.exp(-shielding_mm / 10.0)  # 10 mm attenuation length

        dose = (proton_fluence * proton_dose_coefficient +
                electron_fluence * electron_dose_coefficient) * shielding_factor

        return dose

    def _calculate_displacement_damage(self, proton_fluence: float,
                                     shielding_mm: float) -> float:
        """
        Calculate non-ionizing energy loss (displacement damage)

        Args:
            proton_fluence: Proton fluence in protons/cm²
            shielding_mm: Shielding thickness in mm

        Returns:
            Displacement damage dose in MeV cm²/g
        """
        # NIEL coefficient for 1 MeV protons in silicon
        niel_coefficient = 4e-3  # MeV cm²/g per proton/cm²

        # Shielding attenuation
        shielding_factor = np.exp(-shielding_mm / 15.0)  # 15 mm attenuation length

        damage_dose = proton_fluence * niel_coefficient * shielding_factor

        return damage_dose

    def _calculate_step_dose(self, proton_flux: float, electron_flux: float,
                           solar_flux: float, gcr_flux: float,
                           time_step_seconds: float, shielding_mm: float) -> float:
        """
        Calculate dose for a single time step

        Args:
            proton_flux: Proton flux in particles/cm²/s
            electron_flux: Electron flux in particles/cm²/s
            solar_flux: Solar event flux in particles/cm²/s
            gcr_flux: GCR flux in particles/cm²/s
            time_step_seconds: Time step in seconds
            shielding_mm: Shielding thickness in mm

        Returns:
            Dose increment in rads
        """
        total_proton_flux = proton_flux + solar_flux + gcr_flux
        total_electron_flux = electron_flux

        # Fluences for this time step
        proton_fluence = total_proton_flux * time_step_seconds
        electron_fluence = total_electron_flux * time_step_seconds

        return self._calculate_ionizing_dose(proton_fluence, electron_fluence, shielding_mm)