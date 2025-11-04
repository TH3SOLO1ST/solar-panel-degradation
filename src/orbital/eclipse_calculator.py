"""
Eclipse Calculator Module

This module calculates eclipse periods, solar exposure times, and shadow geometry
for satellites in orbit. It handles both umbral (total) and penumbral (partial)
eclipse conditions with high-precision calculations.

References:
- "Fundamentals of Astrodynamics and Applications" by Vallado
- "Space Mission Analysis and Design" by Larson & Wertz
- NASA SP-43: Eclipse prediction algorithms
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from .orbit_propagator import OrbitPropagator, OrbitalState

try:
    from skyfield.api import load, Earth, Sun
    from skyfield.timelib import Timescale
except ImportError:
    raise ImportError("Skyfield library required. Install with: pip install skyfield")


@dataclass
class EclipseEvent:
    """Data class for eclipse event information"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    eclipse_type: str = "none"  # "none", "penumbra", "umbra"
    duration_minutes: float = 0.0
    max_eclipse_fraction: float = 0.0  # Fraction of sun obscured (0-1)
    entry_angle: float = 0.0  # Entry angle relative to velocity vector (degrees)
    exit_angle: float = 0.0   # Exit angle relative to velocity vector (degrees)


@dataclass
class EclipseStatistics:
    """Statistics for eclipse analysis over a time period"""
    total_eclipse_time: float  # Total time in eclipse (hours)
    eclipse_fraction: float    # Fraction of time in eclipse (0-1)
    num_eclipse_events: int    # Number of eclipse periods
    avg_eclipse_duration: float  # Average eclipse duration (minutes)
    max_eclipse_duration: float   # Maximum eclipse duration (minutes)
    penumbra_only_time: float     # Time in penumbra only (hours)
    umbra_time: float            # Time in umbra (hours)


class EclipseCalculator:
    """
    High-precision eclipse calculator for satellite orbits.

    Features:
    - Umbral and penumbral shadow geometry
    - Eclipse entry/exit angle calculation
    - Eclipse duration statistics
    - Support for highly elliptical orbits
    - Atmospheric refraction effects
    """

    # Physical constants
    EARTH_RADIUS = 6378.137  # km
    SUN_RADIUS = 696340.0     # km
    AU_DISTANCE = 149597870.7  # km (1 Astronomical Unit)
    EARTH_SUN_DISTANCE = 149597870.7  # km (average)

    # Atmospheric refraction correction (degrees)
    ATMOSPHERIC_REFRACTION = 0.5667  # degrees at horizon

    def __init__(self, orbit_propagator: OrbitPropagator, use_atmosphere: bool = True):
        """
        Initialize eclipse calculator

        Args:
            orbit_propagator: Configured orbit propagator instance
            use_atmosphere: Include atmospheric refraction effects
        """
        self.propagator = orbit_propagator
        self.use_atmosphere = use_atmosphere
        self.ts = load.timescale()
        self.earth = Earth
        self.sun = Sun

        # Cache for performance
        self._sun_position_cache: Dict[datetime, np.ndarray] = {}

    def calculate_eclipse_at_time(self, time: datetime) -> EclipseEvent:
        """
        Determine eclipse status at a specific time

        Args:
            time: Time to check eclipse status

        Returns:
            EclipseEvent with eclipse information
        """
        # Get satellite position
        state = self.propagator.propagate(time)
        sat_pos = state.position

        # Get sun position
        sun_pos = self._get_sun_position(time)

        # Calculate eclipse geometry
        eclipse_fraction, eclipse_type = self._calculate_eclipse_geometry(sat_pos, sun_pos)

        if eclipse_fraction > 0:
            # In eclipse
            if eclipse_type == "umbra":
                max_eclipse_fraction = 1.0
            else:
                max_eclipse_fraction = eclipse_fraction

            return EclipseEvent(
                eclipse_type=eclipse_type,
                duration_minutes=0.0,
                max_eclipse_fraction=max_eclipse_fraction
            )
        else:
            return EclipseEvent(eclipse_type="none")

    def calculate_eclipse_periods(self, start_time: datetime, duration: timedelta,
                                 time_step: timedelta) -> List[EclipseEvent]:
        """
        Calculate all eclipse periods within a time range

        Args:
            start_time: Start of analysis period
            duration: Duration to analyze
            time_step: Time resolution for eclipse detection

        Returns:
            List of EclipseEvent objects representing each eclipse period
        """
        times = []
        current_time = start_time
        end_time = start_time + duration

        while current_time <= end_time:
            times.append(current_time)
            current_time += time_step

        # Calculate eclipse status at each time point
        eclipse_states = []
        for time in times:
            event = self.calculate_eclipse_at_time(time)
            eclipse_states.append((time, event))

        # Group consecutive eclipse periods
        eclipse_events = []
        current_eclipse = None

        for time, event in eclipse_states:
            if event.eclipse_type != "none":
                if current_eclipse is None:
                    # Start new eclipse period
                    current_eclipse = EclipseEvent(
                        start_time=time,
                        eclipse_type=event.eclipse_type,
                        max_eclipse_fraction=event.max_eclipse_fraction
                    )
                else:
                    # Update current eclipse
                    current_eclipse.max_eclipse_fraction = max(
                        current_eclipse.max_eclipse_fraction, event.max_eclipse_fraction
                    )
                    # Update eclipse type if we enter umbra
                    if event.eclipse_type == "umbra":
                        current_eclipse.eclipse_type = "umbra"
            else:
                if current_eclipse is not None:
                    # End current eclipse period
                    current_eclipse.end_time = time
                    current_eclipse.duration_minutes = (
                        current_eclipse.end_time - current_eclipse.start_time
                    ).total_seconds() / 60.0
                    eclipse_events.append(current_eclipse)
                    current_eclipse = None

        # Handle eclipse that extends beyond end time
        if current_eclipse is not None:
            current_eclipse.end_time = end_time
            current_eclipse.duration_minutes = (
                current_eclipse.end_time - current_eclipse.start_time
            ).total_seconds() / 60.0
            eclipse_events.append(current_eclipse)

        return eclipse_events

    def calculate_eclipse_statistics(self, start_time: datetime, duration: timedelta,
                                   time_step: timedelta) -> EclipseStatistics:
        """
        Calculate comprehensive eclipse statistics for a time period

        Args:
            start_time: Start of analysis period
            duration: Duration to analyze
            time_step: Time resolution for analysis

        Returns:
            EclipseStatistics object with comprehensive eclipse data
        """
        eclipse_events = self.calculate_eclipse_periods(start_time, duration, time_step)

        # Calculate total eclipse time
        total_eclipse_minutes = sum(event.duration_minutes for event in eclipse_events)
        total_time_hours = duration.total_seconds() / 3600.0
        total_eclipse_hours = total_eclipse_minutes / 60.0

        # Calculate time in different eclipse types
        umbra_minutes = sum(
            event.duration_minutes for event in eclipse_events
            if event.eclipse_type == "umbra"
        )
        penumbra_minutes = total_eclipse_minutes - umbra_minutes

        # Calculate statistics
        num_events = len(eclipse_events)
        avg_duration = total_eclipse_minutes / num_events if num_events > 0 else 0.0
        max_duration = max(
            (event.duration_minutes for event in eclipse_events),
            default=0.0
        )
        eclipse_fraction = total_eclipse_hours / total_time_hours if total_time_hours > 0 else 0.0

        return EclipseStatistics(
            total_eclipse_time=total_eclipse_hours,
            eclipse_fraction=eclipse_fraction,
            num_eclipse_events=num_events,
            avg_eclipse_duration=avg_duration,
            max_eclipse_duration=max_duration,
            penumbra_only_time=penumbra_minutes / 60.0,
            umbra_time=umbra_minutes / 60.0
        )

    def _calculate_eclipse_geometry(self, sat_pos: np.ndarray,
                                  sun_pos: np.ndarray) -> Tuple[float, str]:
        """
        Calculate eclipse geometry and fraction

        Args:
            sat_pos: Satellite position vector (km)
            sun_pos: Sun position vector (km)

        Returns:
            Tuple of (eclipse_fraction, eclipse_type)
        """
        # Vector from Earth to satellite
        r_sat = sat_pos
        r_sat_mag = np.linalg.norm(r_sat)

        # Vector from Earth to Sun
        r_sun = sun_pos
        r_sun_mag = np.linalg.norm(r_sun)

        # Vector from satellite to Sun
        r_sat_to_sun = r_sun - r_sat

        # Calculate angular distances
        cos_earth_sat_sun = np.dot(r_sat, r_sat_to_sun) / (r_sat_mag * np.linalg.norm(r_sat_to_sun))
        angle_earth_sat_sun = np.arccos(np.clip(cos_earth_sat_sun, -1.0, 1.0))

        # Earth and Sun angular radii as seen from satellite
        earth_angular_radius = np.arcsin(self.EARTH_RADIUS / r_sat_mag)
        sun_angular_radius = np.arcsin(self.SUN_RADIUS / np.linalg.norm(r_sat_to_sun))

        # Atmospheric refraction correction
        if self.use_atmosphere:
            earth_angular_radius += np.radians(self.ATMOSPHERIC_REFRACTION)

        # Determine eclipse type
        if angle_earth_sat_sun + sun_angular_radius < earth_angular_radius:
            # Total eclipse (umbra)
            eclipse_fraction = 1.0
            eclipse_type = "umbra"
        elif abs(angle_earth_sat_sun - earth_angular_radius) < sun_angular_radius:
            # Partial eclipse (penumbra)
            # Calculate overlap fraction
            if earth_angular_radius >= sun_angular_radius:
                # Earth appears larger than Sun
                overlap = (sun_angular_radius + earth_angular_radius - angle_earth_sat_sun) / (2 * sun_angular_radius)
                eclipse_fraction = max(0.0, min(1.0, overlap))
            else:
                # Sun appears larger than Earth
                overlap = (sun_angular_radius + earth_angular_radius - angle_earth_sat_sun) / (2 * earth_angular_radius)
                eclipse_fraction = max(0.0, min(1.0, overlap))
            eclipse_type = "penumbra"
        else:
            # No eclipse
            eclipse_fraction = 0.0
            eclipse_type = "none"

        return eclipse_fraction, eclipse_type

    def _get_sun_position(self, time: datetime) -> np.ndarray:
        """
        Get Sun position vector at given time (with caching)

        Args:
            time: Time to get Sun position

        Returns:
            Sun position vector in ECI coordinates (km)
        """
        if time in self._sun_position_cache:
            return self._sun_position_cache[time]

        t = self.ts.from_datetime(time)
        astrometric = self.sun.at(t).observe(self.earth)
        position = astrometric.position.km  # This gives Earth relative to Sun
        sun_pos = -position  # Convert to Sun relative to Earth

        # Cache the result
        self._sun_position_cache[time] = sun_pos
        return sun_pos

    def calculate_solar_incidence_angle(self, time: datetime,
                                      panel_normal: np.ndarray) -> float:
        """
        Calculate solar incidence angle for a solar panel

        Args:
            time: Time for calculation
            panel_normal: Normal vector of solar panel (unit vector)

        Returns:
            Solar incidence angle in radians (0 = direct sunlight, π/2 = edge-on)
        """
        # Get satellite and sun positions
        state = self.propagator.propagate(time)
        sun_pos = self._get_sun_position(time)

        # Vector from satellite to Sun
        sun_vector = sun_pos - state.position
        sun_vector = sun_vector / np.linalg.norm(sun_vector)  # Normalize

        # Calculate angle between panel normal and sun vector
        cos_angle = np.dot(panel_normal, sun_vector)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Ensure angle is between 0 and π/2 (solar panels work from one side)
        angle = np.arccos(abs(cos_angle))
        return angle

    def predict_next_eclipse(self, current_time: datetime,
                           search_duration: timedelta = timedelta(hours=24)) -> Optional[EclipseEvent]:
        """
        Predict the next eclipse after current_time

        Args:
            current_time: Time to start searching from
            search_duration: Maximum time to search ahead

        Returns:
            Next EclipseEvent or None if no eclipse found in search period
        """
        time_step = timedelta(minutes=1)  # Fine resolution for accurate prediction
        end_time = current_time + search_duration

        # Search for eclipse start
        search_time = current_time
        while search_time <= end_time:
            event = self.calculate_eclipse_at_time(search_time)
            if event.eclipse_type != "none":
                # Found eclipse start, now find the full eclipse period
                eclipse_events = self.calculate_eclipse_periods(
                    search_time,
                    timedelta(hours=6),  # Assume max 6-hour eclipse
                    timedelta(minutes=1)
                )
                return eclipse_events[0] if eclipse_events else None
            search_time += time_step

        return None

    def get_eclipse_map_data(self, start_time: datetime, duration: timedelta,
                           time_step: timedelta) -> Dict[str, List]:
        """
        Get eclipse data suitable for plotting eclipse maps

        Args:
            start_time: Start time for analysis
            duration: Duration to analyze
            time_step: Time resolution

        Returns:
            Dictionary with time series data for plotting
        """
        times = []
        eclipse_fractions = []
        eclipse_types = []
        altitudes = []
        latitudes = []
        longitudes = []

        current_time = start_time
        end_time = start_time + duration

        while current_time <= end_time:
            state = self.propagator.propagate(current_time)
            event = self.calculate_eclipse_at_time(current_time)

            times.append(current_time)
            eclipse_fractions.append(event.max_eclipse_fraction)
            eclipse_types.append(event.eclipse_type)
            altitudes.append(state.altitude)
            latitudes.append(state.latitude)
            longitudes.append(state.longitude)

            current_time += time_step

        return {
            'times': times,
            'eclipse_fractions': eclipse_fractions,
            'eclipse_types': eclipse_types,
            'altitudes': altitudes,
            'latitudes': latitudes,
            'longitudes': longitudes
        }

    def clear_cache(self) -> None:
        """Clear internal position caches"""
        self._sun_position_cache.clear()