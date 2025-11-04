"""
Eclipse Calculator
==================

Calculates eclipse periods and solar exposure for satellites in orbit.

This module implements geometric calculations to determine when a satellite
is in Earth's shadow (umbra or penumbra) and provides statistics on
eclipse duration and frequency.

Classes:
    EclipseCalculator: Main class for eclipse calculations
    EclipsePeriod: Data structure for eclipse period information
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class EclipsePeriod:
    """Data structure for eclipse period information"""
    start_time: float  # hours from simulation start
    end_time: float    # hours from simulation start
    duration: float    # hours
    eclipse_type: str  # 'umbra' or 'penumbra'
    max_depth: float   # 0.0 to 1.0 (1.0 = total eclipse)

class EclipseCalculator:
    """Calculator for eclipse periods and solar exposure"""

    def __init__(self, earth_radius_km: float = 6371.0, sun_radius_km: float = 696340.0):
        """
        Initialize eclipse calculator

        Args:
            earth_radius_km: Earth radius in km
            sun_radius_km: Sun radius in km
        """
        self.earth_radius = earth_radius_km
        self.sun_radius = sun_radius_km
        self.au_distance = 149597870.7  # km (1 Astronomical Unit)

    def calculate_eclipse_periods(self, positions: np.ndarray, times: np.ndarray,
                                 sun_position: np.ndarray) -> List[EclipsePeriod]:
        """
        Calculate eclipse periods for satellite positions

        Args:
            positions: Nx3 array of satellite positions in km
            times: Array of corresponding times in hours
            sun_position: 3D array of sun position in km

        Returns:
            List of EclipsePeriod objects
        """
        eclipse_periods = []
        in_eclipse = False
        eclipse_start = 0
        current_type = None

        for i, (pos, t) in enumerate(zip(positions, times)):
            eclipse_state = self._check_eclipse_state(pos, sun_position)

            if not in_eclipse and eclipse_state['in_eclipse']:
                # Eclipse starts
                in_eclipse = True
                eclipse_start = t
                current_type = eclipse_state['type']
                max_depth = eclipse_state['depth']

            elif in_eclipse and not eclipse_state['in_eclipse']:
                # Eclipse ends
                in_eclipse = False
                duration = t - eclipse_start
                if duration > 0:
                    eclipse_periods.append(EclipsePeriod(
                        start_time=eclipse_start,
                        end_time=t,
                        duration=duration,
                        eclipse_type=current_type,
                        max_depth=max_depth
                    ))

            elif in_eclipse:
                # Update max depth during eclipse
                max_depth = max(max_depth, eclipse_state['depth'])

        return eclipse_periods

    def _check_eclipse_state(self, satellite_pos: np.ndarray, sun_pos: np.ndarray) -> Dict:
        """
        Check if satellite is in eclipse

        Args:
            satellite_pos: 3D satellite position in km
            sun_pos: 3D sun position in km

        Returns:
            Dictionary with eclipse state information
        """
        # Simplified eclipse calculation
        # Check if satellite is in Earth's shadow

        # Vector from Earth center to satellite
        sat_vector = satellite_pos
        sat_distance = np.linalg.norm(sat_vector)

        # Vector from satellite to sun
        sun_vector = sun_pos - satellite_pos
        sun_distance = np.linalg.norm(sun_vector)

        # Vector from Earth center to sun
        earth_sun_vector = sun_pos
        earth_sun_distance = np.linalg.norm(earth_sun_vector)

        # Calculate angular radius of Earth as seen from satellite
        earth_angular_radius = np.arcsin(self.earth_radius / sat_distance)

        # Calculate angular radius of sun as seen from satellite
        sun_angular_radius = np.arcsin(self.sun_radius / sun_distance)

        # Calculate angular separation between satellite and sun as seen from Earth
        # Using dot product to find angle between sat_vector and sun_vector
        cos_angle = np.dot(sat_vector, sun_pos) / (sat_distance * earth_sun_distance)
        angular_separation = np.arccos(np.clip(cos_angle, -1, 1))

        # Check eclipse conditions
        if angular_separation < earth_angular_radius - sun_angular_radius:
            # Total eclipse (umbra)
            return {
                'in_eclipse': True,
                'type': 'umbra',
                'depth': 1.0
            }
        elif angular_separation < earth_angular_radius + sun_angular_radius:
            # Partial eclipse (penumbra)
            depth = (earth_angular_radius + sun_angular_radius - angular_separation) / (2 * sun_angular_radius)
            return {
                'in_eclipse': True,
                'type': 'penumbra',
                'depth': min(1.0, max(0.0, depth))
            }
        else:
            # No eclipse
            return {
                'in_eclipse': False,
                'type': None,
                'depth': 0.0
            }

    def calculate_solar_exposure(self, positions: np.ndarray, times: np.ndarray,
                                sun_position: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate solar exposure time

        Args:
            positions: Nx3 array of satellite positions
            times: Array of times in hours
            sun_position: Sun position vector

        Returns:
            Tuple of (solar_exposure_array, total_exposure_percentage)
        """
        n_samples = len(positions)
        in_sunlight = np.zeros(n_samples, dtype=bool)

        for i, pos in enumerate(positions):
            eclipse_state = self._check_eclipse_state(pos, sun_position)
            in_sunlight[i] = not eclipse_state['in_eclipse']

        # Calculate exposure percentage
        exposure_percentage = np.sum(in_sunlight) / n_samples * 100

        return in_sunlight, exposure_percentage

    def get_eclipse_statistics(self, eclipse_periods: List[EclipsePeriod]) -> Dict:
        """
        Calculate eclipse statistics

        Args:
            eclipse_periods: List of EclipsePeriod objects

        Returns:
            Dictionary with eclipse statistics
        """
        if not eclipse_periods:
            return {
                'total_eclipses': 0,
                'total_eclipse_time': 0.0,
                'average_eclipse_duration': 0.0,
                'longest_eclipse': 0.0,
                'shortest_eclipse': 0.0,
                'eclipse_frequency': 0.0
            }

        total_time = sum(p.duration for p in eclipse_periods)
        avg_duration = total_time / len(eclipse_periods)
        longest = max(p.duration for p in eclipse_periods)
        shortest = min(p.duration for p in eclipse_periods)

        # Calculate frequency (eclipses per day)
        if eclipse_periods:
            time_span = eclipse_periods[-1].end_time - eclipse_periods[0].start_time
            frequency = len(eclipse_periods) / (time_span / 24) if time_span > 0 else 0
        else:
            frequency = 0

        return {
            'total_eclipses': len(eclipse_periods),
            'total_eclipse_time': total_time,  # hours
            'average_eclipse_duration': avg_duration,  # hours
            'longest_eclipse': longest,  # hours
            'shortest_eclipse': shortest,  # hours
            'eclipse_frequency': frequency  # eclipses per day
        }

    def calculate_solar_incident_angle(self, positions: np.ndarray, sun_position: np.ndarray,
                                     panel_normal: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
        """
        Calculate solar incident angle for solar panel

        Args:
            positions: Nx3 array of satellite positions
            sun_position: Sun position vector
            panel_normal: Normal vector of solar panel (default: z-axis)

        Returns:
            Array of incident angles in radians
        """
        incident_angles = np.zeros(len(positions))

        for i, pos in enumerate(positions):
            # Vector from satellite to sun
            sun_vector = sun_position - pos
            sun_vector = sun_vector / np.linalg.norm(sun_vector)

            # Calculate incident angle
            cos_angle = np.dot(panel_normal, sun_vector)
            cos_angle = np.clip(cos_angle, -1, 1)
            incident_angles[i] = np.arccos(cos_angle)

        return incident_angles