"""
Orbit Propagator Module

This module implements satellite orbit propagation using SGP4 for LEO/MEO orbits
and simplified Keplerian propagation for GEO orbits. It calculates satellite
positions, velocities, and provides orbital parameters for degradation analysis.

References:
- SGP4 algorithm: Hoots, Roehrich, "Models for Atmosphere"
- Skyfield library documentation
- Vallado, "Fundamentals of Astrodynamics and Applications"
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

try:
    from skyfield.api import load, EarthSatellite, wgs84
    from skyfield.timelib import Timescale
except ImportError:
    raise ImportError("Skyfield library required. Install with: pip install skyfield")


@dataclass
class OrbitalElements:
    """Keplerian orbital elements"""
    semi_major_axis: float  # km
    eccentricity: float
    inclination: float  # radians
    raan: float  # Right Ascension of Ascending Node (radians)
    arg_perigee: float  # Argument of Perigee (radians)
    mean_anomaly: float  # Mean Anomaly (radians)
    epoch: datetime

    def __post_init__(self):
        """Validate orbital elements"""
        if self.eccentricity < 0 or self.eccentricity >= 1:
            raise ValueError(f"Eccentricity must be in range [0, 1), got {self.eccentricity}")
        if self.semi_major_axis < 6378.137:  # Earth radius in km
            raise ValueError(f"Orbit altitude too low: {self.semi_major_axis} km")


@dataclass
class OrbitalState:
    """Satellite orbital state at a given time"""
    time: datetime
    position: np.ndarray  # km [x, y, z] in ECI frame
    velocity: np.ndarray  # km/s [vx, vy, vz] in ECI frame
    altitude: float  # km above Earth surface
    latitude: float  # degrees
    longitude: float  # degrees
    velocity_magnitude: float  # km/s


class OrbitPropagator:
    """
    Satellite orbit propagator supporting multiple orbit types and propagation methods.

    Features:
    - SGP4 propagation for LEO/MEO orbits
    - Keplerian propagation for GEO/circular orbits
    - High-precision position and velocity calculation
    - Support for multiple coordinate systems
    """

    # Physical constants
    EARTH_RADIUS = 6378.137  # km
    MU_EARTH = 398600.4418  # km^3/s^2 (Earth gravitational parameter)
    EARTH_FLATTENING = 1/298.257223563
    J2 = 1.08262668e-3  # Earth's J2 perturbation coefficient

    def __init__(self, use_sgp4: bool = True):
        """
        Initialize orbit propagator

        Args:
            use_sgp4: Use SGP4 algorithm when available (recommended for accuracy)
        """
        self.use_sgp4 = use_sgp4
        self.ts = load.timescale()
        self._cached_sgp4_sat: Optional[EarthSatellite] = None
        self._cached_elements: Optional[OrbitalElements] = None

    def set_orbit_from_elements(self, elements: OrbitalElements) -> None:
        """
        Set orbit using Keplerian elements

        Args:
            elements: Orbital elements defining the orbit
        """
        self._cached_elements = elements

        if self.use_sgp4:
            self._setup_sgp4_satellite(elements)

    def set_orbit_from_tle(self, tle_line1: str, tle_line2: str) -> None:
        """
        Set orbit using Two-Line Element (TLE) format

        Args:
            tle_line1: First line of TLE
            tle_line2: Second line of TLE
        """
        if not self.use_sgp4:
            raise ValueError("TLE support requires SGP4 propagation method")

        self._cached_sgp4_sat = EarthSatellite(tle_line1, tle_line2, 'SAT')

    def _setup_sgp4_satellite(self, elements: OrbitalElements) -> None:
        """Setup SGP4 satellite from orbital elements"""
        # Convert orbital elements to mean orbital elements for SGP4
        period = 2 * np.pi * np.sqrt(elements.semi_major_axis**3 / self.MU_EARTH)
        mean_motion = 86400.0 / period  # revolutions per day

        # Estimate inclination in degrees and RAAN in degrees
        inc_deg = np.degrees(elements.inclination)
        raan_deg = np.degrees(elements.raan)
        ecc = elements.eccentricity
        arg_per_deg = np.degrees(elements.arg_perigee)
        mean_anom_deg = np.degrees(elements.mean_anomaly)

        # Create satellite with epoch
        t = self.ts.from_datetime(elements.epoch)
        self._cached_sgp4_sat = EarthSatellite(
            '1 00000U 23001.00000000  .00000000  00000+0  00000+0 0  0000',
            f'2 00000 {inc_deg:.4f} {raan_deg:.4f} {ecc:.7f} {arg_per_deg:.4f} {mean_anom_deg:.4f} {mean_motion:.8f}00000',
            'SATELLITE',
            t
        )

    def propagate(self, time: datetime) -> OrbitalState:
        """
        Propagate orbit to specified time

        Args:
            time: Target time for propagation

        Returns:
            OrbitalState at specified time
        """
        if self.use_sgp4 and self._cached_sgp4_sat:
            return self._propagate_sgp4(time)
        elif self._cached_elements:
            return self._propagate_keplerian(time)
        else:
            raise ValueError("No orbit defined. Call set_orbit_from_elements or set_orbit_from_tle first")

    def _propagate_sgp4(self, time: datetime) -> OrbitalState:
        """Propagate using SGP4 algorithm"""
        t = self.ts.from_datetime(time)
        position, velocity = self._cached_sgp4_sat.at(t).position.km, self._cached_sgp4_sat.at(t).velocity.km_per_s

        # Calculate geodetic coordinates
        geodetic = wgs84.subpoint(self._cached_sgp4_sat.at(t))

        # Calculate altitude and velocity magnitude
        altitude = np.linalg.norm(position) - self.EARTH_RADIUS
        velocity_magnitude = np.linalg.norm(velocity)

        return OrbitalState(
            time=time,
            position=position,
            velocity=velocity,
            altitude=altitude,
            latitude=geodetic.latitude.degrees,
            longitude=geodetic.longitude.degrees,
            velocity_magnitude=velocity_magnitude
        )

    def _propagate_keplerian(self, time: datetime) -> OrbitalState:
        """Propagate using simplified Keplerian elements"""
        if not self._cached_elements:
            raise ValueError("Orbital elements not defined")

        elements = self._cached_elements

        # Time since epoch in seconds
        dt = (time - elements.epoch).total_seconds()

        # Mean motion (rad/s)
        n = np.sqrt(self.MU_EARTH / elements.semi_major_axis**3)

        # Mean anomaly at time t
        M = elements.mean_anomaly + n * dt

        # Solve Kepler's equation for eccentric anomaly (Newton-Raphson)
        E = M
        for _ in range(10):  # Usually converges quickly
            E = E - (E - elements.eccentricity * np.sin(E) - M) / (1 - elements.eccentricity * np.cos(E))

        # True anomaly
        nu = 2 * np.arctan2(
            np.sqrt(1 + elements.eccentricity) * np.sin(E/2),
            np.sqrt(1 - elements.eccentricity) * np.cos(E/2)
        )

        # Distance from Earth center
        r = elements.semi_major_axis * (1 - elements.eccentricity * np.cos(E))

        # Position in orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)

        # Velocity in orbital plane
        h = np.sqrt(self.MU_EARTH * elements.semi_major_axis * (1 - elements.eccentricity**2))
        vx_orb = -h * np.sin(nu) / r
        vy_orb = h * (elements.eccentricity + np.cos(nu)) / r

        # Rotation matrices to transform to ECI frame
        cos_omega = np.cos(elements.arg_perigee)
        sin_omega = np.sin(elements.arg_perigee)
        cos_i = np.cos(elements.inclination)
        sin_i = np.sin(elements.inclination)
        cos_raan = np.cos(elements.raan)
        sin_raan = np.sin(elements.raan)

        # Transform position to ECI frame
        x = (cos_omega * cos_raan - sin_omega * cos_i * sin_raan) * x_orb + \
            (-sin_omega * cos_raan - cos_omega * cos_i * sin_raan) * y_orb
        y = (cos_omega * sin_raan + sin_omega * cos_i * cos_raan) * x_orb + \
            (-sin_omega * sin_raan + cos_omega * cos_i * cos_raan) * y_orb
        z = sin_omega * sin_i * x_orb + cos_omega * sin_i * y_orb

        # Transform velocity to ECI frame
        vx = (cos_omega * cos_raan - sin_omega * cos_i * sin_raan) * vx_orb + \
             (-sin_omega * cos_raan - cos_omega * cos_i * sin_raan) * vy_orb
        vy = (cos_omega * sin_raan + sin_omega * cos_i * cos_raan) * vx_orb + \
             (-sin_omega * sin_raan + cos_omega * cos_i * cos_raan) * vy_orb
        vz = sin_omega * sin_i * vx_orb + cos_omega * sin_i * vy_orb

        position = np.array([x, y, z])
        velocity = np.array([vx, vy, vz])

        # Calculate geodetic coordinates (simplified)
        altitude = np.linalg.norm(position) - self.EARTH_RADIUS
        velocity_magnitude = np.linalg.norm(velocity)

        # Simple latitude/longitude calculation
        r_xy = np.sqrt(x**2 + y**2)
        latitude = np.degrees(np.arctan2(z, r_xy))
        longitude = np.degrees(np.arctan2(y, x))

        return OrbitalState(
            time=time,
            position=position,
            velocity=velocity,
            altitude=altitude,
            latitude=latitude,
            longitude=longitude,
            velocity_magnitude=velocity_magnitude
        )

    def get_orbital_period(self) -> float:
        """
        Calculate orbital period in seconds

        Returns:
            Orbital period in seconds
        """
        if self._cached_elements:
            a = self._cached_elements.semi_major_axis
            return 2 * np.pi * np.sqrt(a**3 / self.MU_EARTH)
        elif self._cached_sgp4_sat:
            # Estimate from SGP4 satellite
            a = (self.MU_EARTH / ((2 * np.pi * self._cached_sgp4_sat.no / 86400)**2))**(1/3)
            return 2 * np.pi * np.sqrt(a**3 / self.MU_EARTH)
        else:
            raise ValueError("No orbit defined")

    def propagate_batch(self, times: List[datetime]) -> List[OrbitalState]:
        """
        Propagate orbit for multiple times (batch processing)

        Args:
            times: List of target times

        Returns:
            List of OrbitalState objects
        """
        return [self.propagate(time) for time in times]

    def create_time_series(self, start_time: datetime, duration: timedelta,
                          time_step: timedelta) -> List[OrbitalState]:
        """
        Create time series of orbital states

        Args:
            start_time: Start time for propagation
            duration: Total duration to propagate
            time_step: Time step between states

        Returns:
            List of OrbitalState objects covering the time range
        """
        times = []
        current_time = start_time
        while current_time <= start_time + duration:
            times.append(current_time)
            current_time += time_step

        return self.propagate_batch(times)

    @staticmethod
    def create_leo_orbit(altitude_km: float, inclination_deg: float,
                        epoch: datetime) -> OrbitalElements:
        """
        Create typical LEO orbital elements

        Args:
            altitude_km: Altitude above Earth surface in km
            inclination_deg: Orbital inclination in degrees
            epoch: Reference epoch

        Returns:
            OrbitalElements for LEO orbit
        """
        a = OrbitPropagator.EARTH_RADIUS + altitude_km
        return OrbitalElements(
            semi_major_axis=a,
            eccentricity=0.001,  # Nearly circular
            inclination=np.radians(inclination_deg),
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0,
            epoch=epoch
        )

    @staticmethod
    def create_geo_orbit(epoch: datetime) -> OrbitalElements:
        """
        Create GEO orbital elements

        Args:
            epoch: Reference epoch

        Returns:
            OrbitalElements for GEO orbit
        """
        # GEO altitude: approximately 35786 km above equator
        a = 42164.0  # km (Earth radius + GEO altitude)
        return OrbitalElements(
            semi_major_axis=a,
            eccentricity=0.0,  # Circular
            inclination=0.0,  # Equatorial
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0,
            epoch=epoch
        )

    @staticmethod
    def create_molniya_orbit(epoch: datetime, raan_deg: float = 0.0,
                           arg_perigee_deg: float = 270.0) -> OrbitalElements:
        """
        Create Molniya orbit elements (highly eccentric, 12-hour period)

        Args:
            epoch: Reference epoch
            raan_deg: RAAN in degrees
            arg_perigee_deg: Argument of perigee in degrees

        Returns:
            OrbitalElements for Molniya orbit
        """
        a = 26600.0  # km (approximately 12-hour period)
        return OrbitalElements(
            semi_major_axis=a,
            eccentricity=0.74,  # High eccentricity
            inclination=np.radians(63.4),  # Critical inclination
            raan=np.radians(raan_deg),
            arg_perigee=np.radians(arg_perigee_deg),
            mean_anomaly=0.0,
            epoch=epoch
        )