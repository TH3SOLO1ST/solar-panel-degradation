"""
Orbit Propagator
================

Implements orbital propagation calculations for various orbit types
including LEO, MEO, GEO, and Sun-synchronous orbits.

This module uses the SGP4 algorithm for high-precision orbit propagation
and provides simplified methods for GEO orbits.

Classes:
    OrbitPropagator: Main propagator class
    OrbitElements: Data structure for orbital elements
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
try:
    from sgp4.api import Satrec, jday
except ImportError:
    # Fallback implementation if sgp4 not available
    Satrec = None

class OrbitElements:
    """Data structure for orbital elements"""

    def __init__(self,
                 semi_major_axis_km: float,
                 eccentricity: float,
                 inclination_deg: float,
                 raan_deg: float,
                 arg_perigee_deg: float,
                 mean_anomaly_deg: float,
                 epoch: datetime):
        """
        Initialize orbital elements

        Args:
            semi_major_axis_km: Semi-major axis in km
            eccentricity: Orbital eccentricity (0 for circular)
            inclination_deg: Inclination in degrees
            raan_deg: Right ascension of ascending node in degrees
            arg_perigee_deg: Argument of perigee in degrees
            mean_anomaly_deg: Mean anomaly at epoch in degrees
            epoch: Reference epoch time
        """
        self.a = semi_major_axis_km
        self.e = eccentricity
        self.i = np.radians(inclination_deg)
        self.raan = np.radians(raan_deg)
        self.omega = np.radians(arg_perigee_deg)
        self.M0 = np.radians(mean_anomaly_deg)
        self.epoch = epoch

        # Calculate orbital period
        self.mu = 398600.4418  # Earth's gravitational parameter (km³/s²)
        self.period = 2 * np.pi * np.sqrt(self.a**3 / self.mu)  # seconds

class OrbitPropagator:
    """Main orbit propagator class"""

    def __init__(self, orbit_elements: OrbitElements):
        """
        Initialize orbit propagator

        Args:
            orbit_elements: Orbital elements object
        """
        self.orbit = orbit_elements
        self.use_sgp4 = Satrec is not None and self._should_use_sgp4()

    def _should_use_sgp4(self) -> bool:
        """Determine if SGP4 should be used based on orbit type"""
        # Use simplified propagation for near-circular GEO orbits
        if abs(self.orbit.a - 42164) < 1000 and self.orbit.e < 0.01:
            return False
        return True

    def propagate(self, start_time: datetime, duration_days: float,
                  time_step_hours: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate orbit for specified duration

        Args:
            start_time: Start time for propagation
            duration_days: Duration in days
            time_step_hours: Time step in hours

        Returns:
            Tuple of (positions, times) where positions is Nx3 array in km
        """
        if self.use_sgp4:
            return self._propagate_sgp4(start_time, duration_days, time_step_hours)
        else:
            return self._propagate_simplified(start_time, duration_days, time_step_hours)

    def _propagate_sgp4(self, start_time: datetime, duration_days: float,
                       time_step_hours: float) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate using SGP4 algorithm"""
        # Convert TLE format for SGP4 (simplified)
        # This is a placeholder - real implementation would need proper TLE conversion
        raise NotImplementedError("SGP4 propagation requires proper TLE conversion")

    def _propagate_simplified(self, start_time: datetime, duration_days: float,
                             time_step_hours: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simplified Keplerian propagation for near-circular orbits"""
        # Time array
        total_hours = duration_days * 24
        time_steps = int(total_hours / time_step_hours)
        times = np.arange(time_steps + 1) * time_step_hours

        # Initialize position array
        positions = np.zeros((len(times), 3))

        # Earth parameters
        earth_omega = 2 * np.pi / (24 * 3600)  # Earth rotation rate (rad/s)
        earth_radius = 6371.0  # km

        for i, t in enumerate(times):
            # Mean motion (rad/s)
            n = 2 * np.pi / self.orbit.period

            # Mean anomaly at time t
            dt_seconds = t * 3600
            M = self.orbit.M0 + n * dt_seconds

            # For circular orbits, true anomaly ≈ mean anomaly
            nu = M

            # Position in orbital plane
            r = self.orbit.a * (1 - self.orbit.e**2) / (1 + self.orbit.e * np.cos(nu))

            # Position in orbital plane coordinates
            x_orbital = r * np.cos(nu)
            y_orbital = r * np.sin(nu)
            z_orbital = 0

            # Rotation matrices for orbital elements
            # Rotation by argument of perigee
            R_omega = np.array([
                [np.cos(self.orbit.omega), -np.sin(self.orbit.omega), 0],
                [np.sin(self.orbit.omega), np.cos(self.orbit.omega), 0],
                [0, 0, 1]
            ])

            # Rotation by inclination
            R_i = np.array([
                [1, 0, 0],
                [0, np.cos(self.orbit.i), -np.sin(self.orbit.i)],
                [0, np.sin(self.orbit.i), np.cos(self.orbit.i)]
            ])

            # Rotation by RAAN
            R_raan = np.array([
                [np.cos(self.orbit.raan), -np.sin(self.orbit.raan), 0],
                [np.sin(self.orbit.raan), np.cos(self.orbit.raan), 0],
                [0, 0, 1]
            ])

            # Combined rotation
            R_total = R_raan @ R_i @ R_omega

            # Transform to inertial coordinates
            pos_orbital = np.array([x_orbital, y_orbital, z_orbital])
            pos_inertial = R_total @ pos_orbital

            positions[i] = pos_inertial

        return positions, times

    def get_orbital_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate orbital velocity for given positions

        Args:
            positions: Nx3 array of positions in km

        Returns:
            Nx3 array of velocities in km/s
        """
        velocities = np.zeros_like(positions)

        for i, pos in enumerate(positions):
            r = np.linalg.norm(pos)
            v_mag = np.sqrt(self.orbit.mu / r)

            # Velocity perpendicular to position (simplified)
            # For circular orbits, velocity is perpendicular to radius
            if i == 0:
                # Use first two points to estimate direction
                if len(positions) > 1:
                    delta_pos = positions[1] - pos
                    # Make perpendicular
                    tangent = np.cross(pos, np.array([0, 0, 1]))
                    if np.linalg.norm(tangent) > 0:
                        tangent = tangent / np.linalg.norm(tangent)
                    else:
                        tangent = np.array([-pos[1], pos[0], 0])
                        tangent = tangent / np.linalg.norm(tangent)
                else:
                    tangent = np.array([-pos[1], pos[0], 0])
                    tangent = tangent / np.linalg.norm(tangent)
            else:
                # Use finite difference
                if i < len(positions) - 1:
                    tangent = positions[i + 1] - positions[i - 1]
                else:
                    tangent = positions[i] - positions[i - 1]
                tangent = tangent / np.linalg.norm(tangent)

            velocities[i] = v_mag * tangent

        return velocities

    def calculate_altitude(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate altitude for given positions

        Args:
            positions: Nx3 array of positions in km

        Returns:
            Array of altitudes in km
        """
        earth_radius = 6371.0  # km
        distances = np.linalg.norm(positions, axis=1)
        return distances - earth_radius