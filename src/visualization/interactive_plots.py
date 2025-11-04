"""
Interactive Plots Module

This module creates interactive visualizations for solar panel degradation analysis
using Plotly. It provides comprehensive plotting capabilities for lifetime power
trends, degradation breakdowns, orbital parameters, and environmental conditions.

References:
- Plotly Python documentation
- "Data Visualization with Plotly" - O'Reilly
- NASA Visualization Standards for Space Data
- ESA Data Visualization Guidelines
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import math

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
except ImportError:
    raise ImportError("Plotly required for interactive visualizations. Install with: pip install plotly")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    # Fallback if matplotlib not available
    pass

from ..degradation.lifetime_model import LifetimeState, LifetimePrediction
from ..degradation.power_calculator import PowerOutput
from ..thermal.thermal_analysis import ThermalState
from ..orbital.orbit_propagator import OrbitalState


class InteractivePlots:
    """
    Interactive visualization toolkit for solar panel degradation analysis.

    Features:
    - Lifetime power trend plotting
    - Degradation mechanism breakdown
    - Orbital parameter visualization
    - Environmental condition plots
    - Multi-scenario comparison
    - Export capabilities
    - Customizable themes and styling
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize interactive plots

        Args:
            theme: Plotly theme for styling
        """
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set1
        self.degradation_colors = {
            'radiation': '#FF6B6B',
            'thermal': '#4ECDC4',
            'contamination': '#45B7D1',
            'aging': '#96CEB4',
            'combined': '#FFA07A'
        }

    def plot_lifetime_power_trend(self, lifetime_states: List[LifetimeState],
                                 title: str = "Solar Panel Lifetime Power Trend") -> go.Figure:
        """
        Create interactive lifetime power trend plot

        Args:
            lifetime_states: List of lifetime states over mission
            title: Plot title

        Returns:
            Plotly figure object
        """
        if not lifetime_states:
            return go.Figure()

        # Extract data
        times = [state.time for state in lifetime_states]
        mission_hours = [state.mission_time_hours for state in lifetime_states]
        powers = [state.current_power_watts for state in lifetime_states]
        degradations = [state.power_degradation_percent for state in lifetime_states]
        efficiency_factors = [state.efficiency_factor for state in lifetime_states]

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Power Output', 'Power Degradation', 'Efficiency Factor'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )

        # Power output plot
        fig.add_trace(
            go.Scatter(
                x=times,
                y=powers,
                mode='lines+markers',
                name='Power Output',
                line=dict(color='royalblue', width=2),
                hovertemplate='Time: %{x}<br>Power: %{y:.2f} W<extra></extra>'
            ),
            row=1, col=1
        )

        # Add initial power reference line
        if lifetime_states:
            initial_power = lifetime_states[0].current_power_watts
            fig.add_hline(
                y=initial_power,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Initial: {initial_power:.1f} W",
                row=1, col=1
            )

        # Degradation plot
        fig.add_trace(
            go.Scatter(
                x=times,
                y=degradations,
                mode='lines+markers',
                name='Power Degradation',
                line=dict(color='red', width=2),
                hovertemplate='Time: %{x}<br>Degradation: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )

        # Efficiency factor plot
        fig.add_trace(
            go.Scatter(
                x=times,
                y=efficiency_factors,
                mode='lines+markers',
                name='Efficiency Factor',
                line=dict(color='green', width=2),
                hovertemplate='Time: %{x}<br>Efficiency: %{y:.4f}<extra></extra>'
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
            title=title,
            height=900,
            template=self.theme,
            showlegend=False
        )

        # Update axes
        fig.update_xaxes(title_text="Mission Time", row=3, col=1)
        fig.update_yaxes(title_text="Power (W)", row=1, col=1)
        fig.update_yaxes(title_text="Degradation (%)", row=2, col=1)
        fig.update_yaxes(title_text="Efficiency Factor", row=3, col=1)

        return fig

    def plot_degradation_breakdown(self, lifetime_states: List[LifetimeState],
                                 title: str = "Degradation Mechanism Breakdown") -> go.Figure:
        """
        Create degradation mechanism breakdown plot

        Args:
            lifetime_states: List of lifetime states
            title: Plot title

        Returns:
            Plotly figure object
        """
        if not lifetime_states:
            return go.Figure()

        # Extract mechanism contributions over time
        times = [state.time for state in lifetime_states]

        mechanism_data = {}
        for state in lifetime_states:
            for mechanism in state.mechanisms:
                if mechanism.name not in mechanism_data:
                    mechanism_data[mechanism.name] = []
                mechanism_data[mechanism.name].append(mechanism.contribution_percent)

        # Create stacked area plot
        fig = go.Figure()

        for mechanism_name, contributions in mechanism_data.items():
            if mechanism_name in self.degradation_colors:
                color = self.degradation_colors[mechanism_name]
            else:
                color = self.color_palette[len(mechanism_data) % len(self.color_palette)]

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=contributions,
                    mode='lines',
                    name=mechanism_name.replace('_', ' ').title(),
                    line=dict(width=0),
                    stackgroup='one',
                    fillcolor=color,
                    hovertemplate=f'<b>{mechanism_name.replace("_", " ").title()}</b><br>' +
                                 'Time: %{x}<br>Contribution: %{y:.1f}%<extra></extra>'
                )
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Mission Time",
            yaxis_title="Contribution to Degradation (%)",
            template=self.theme,
            height=600,
            hovermode='x unified'
        )

        return fig

    def plot_environmental_conditions(self, lifetime_states: List[LifetimeState],
                                    title: str = "Environmental Conditions") -> go.Figure:
        """
        Create environmental conditions plot

        Args:
            lifetime_states: List of lifetime states
            title: Plot title

        Returns:
            Plotly figure object
        """
        if not lifetime_states:
            return go.Figure()

        # Extract environmental data
        times = [state.time for state in lifetime_states]
        radiation_dose = [state.total_radiation_dose_rads for state in lifetime_states]
        thermal_cycles = [state.total_thermal_cycles for state in lifetime_states]
        max_temps = [state.max_temperature_K for state in lifetime_states]
        min_temps = [state.min_temperature_K for state in lifetime_states]
        eclipse_hours = [state.total_eclipse_hours for state in lifetime_states]

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Radiation Dose', 'Thermal Cycles',
                'Temperature Range', 'Cumulative Eclipse Time',
                'Temperature Evolution', 'Environmental Overview'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "scatter3d"}]
            ]
        )

        # Radiation dose
        fig.add_trace(
            go.Scatter(
                x=times,
                y=radiation_dose,
                mode='lines+markers',
                name='Radiation Dose',
                line=dict(color='orange', width=2),
                hovertemplate='Time: %{x}<br>Dose: %{y:.2e} rads<extra></extra>'
            ),
            row=1, col=1
        )

        # Thermal cycles
        fig.add_trace(
            go.Scatter(
                x=times,
                y=thermal_cycles,
                mode='lines+markers',
                name='Thermal Cycles',
                line=dict(color='red', width=2),
                hovertemplate='Time: %{x}<br>Cycles: %{y}<0f><extra></extra>'
            ),
            row=1, col=2
        )

        # Temperature range
        fig.add_trace(
            go.Scatter(
                x=times,
                y=max_temps,
                mode='lines',
                name='Max Temperature',
                line=dict(color='red', width=2),
                hovertemplate='Time: %{x}<br>Max Temp: %{y:.1f} K<extra></extra>'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=times,
                y=min_temps,
                mode='lines',
                name='Min Temperature',
                line=dict(color='blue', width=2),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)',
                hovertemplate='Time: %{x}<br>Min Temp: %{y:.1f} K<extra></extra>'
            ),
            row=2, col=1
        )

        # Eclipse time
        fig.add_trace(
            go.Scatter(
                x=times,
                y=eclipse_hours,
                mode='lines+markers',
                name='Eclipse Hours',
                line=dict(color='purple', width=2),
                hovertemplate='Time: %{x}<br>Eclipse: %{y:.1f} hours<extra></extra>'
            ),
            row=2, col=2
        )

        # Combined environmental overview
        mission_hours = [(state.time - lifetime_states[0].time).total_seconds() / 3600.0 for state in lifetime_states]

        fig.add_trace(
            go.Scatter3d(
                x=mission_hours,
                y=radiation_dose,
                z=thermal_cycles,
                mode='markers',
                marker=dict(
                    size=5,
                    color=max_temps,
                    colorscale='Viridis',
                    colorbar=dict(title="Max Temp (K)", x=1.02)
                ),
                name='Environmental State',
                hovertemplate='Mission Time: %{x:.1f} hrs<br>' +
                             'Radiation: %{y:.2e} rads<br>' +
                             'Thermal Cycles: %{z:.0f}<br>' +
                             'Max Temp: %{marker.color:.1f} K<extra></extra>'
            ),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            title=title,
            height=1200,
            template=self.theme,
            showlegend=False
        )

        # Update axes
        fig.update_xaxes(title_text="Mission Time", row=3, col=2)
        fig.update_yaxes(title_text="Mission Hours", row=3, col=2)
        fig.update_zaxis(title_text="Thermal Cycles", row=3, col=2)

        return fig

    def plot_power_performance_metrics(self, power_outputs: List[PowerOutput],
                                     title: str = "Power Performance Metrics") -> go.Figure:
        """
        Create detailed power performance metrics plot

        Args:
            power_outputs: List of power output data
            title: Plot title

        Returns:
            Plotly figure object
        """
        if not power_outputs:
            return go.Figure()

        # Extract data
        times = [p.time for p in power_outputs]
        powers = [p.power_watts for p in power_outputs]
        voltages = [p.voltage_volts for p in power_outputs]
        currents = [p.current_amps for p in power_outputs]
        efficiencies = [p.efficiency * 100 for p in power_outputs]
        temperatures = [p.temperature_K - 273.15 for p in power_outputs]
        degradations = [p.degradation_factor for p in power_outputs]

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Power Output', 'Voltage & Current',
                'Efficiency', 'Temperature',
                'Degradation Factor', 'Performance Overview'
            )
        )

        # Power output
        fig.add_trace(
            go.Scatter(
                x=times,
                y=powers,
                mode='lines+markers',
                name='Power',
                line=dict(color='blue', width=2),
                hovertemplate='Time: %{x}<br>Power: %{y:.2f} W<extra></extra>'
            ),
            row=1, col=1
        )

        # Voltage and current (dual axis)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=voltages,
                mode='lines',
                name='Voltage',
                line=dict(color='red', width=2),
                hovertemplate='Time: %{x}<br>Voltage: %{y:.2f} V<extra></extra>'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=times,
                y=currents,
                mode='lines',
                name='Current',
                line=dict(color='green', width=2),
                yaxis='y2',
                hovertemplate='Time: %{x}<br>Current: %{y:.2f} A<extra></extra>'
            ),
            row=1, col=2
        )

        # Efficiency
        fig.add_trace(
            go.Scatter(
                x=times,
                y=efficiencies,
                mode='lines+markers',
                name='Efficiency',
                line=dict(color='purple', width=2),
                hovertemplate='Time: %{x}<br>Efficiency: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )

        # Temperature
        fig.add_trace(
            go.Scatter(
                x=times,
                y=temperatures,
                mode='lines+markers',
                name='Temperature',
                line=dict(color='orange', width=2),
                hovertemplate='Time: %{x}<br>Temperature: %{y:.1f}°C<extra></extra>'
            ),
            row=2, col=2
        )

        # Degradation factor
        fig.add_trace(
            go.Scatter(
                x=times,
                y=degradations,
                mode='lines+markers',
                name='Degradation Factor',
                line=dict(color='gray', width=2),
                hovertemplate='Time: %{x}<br>Degradation: %{y:.4f}<extra></extra>'
            ),
            row=3, col=1
        )

        # Performance overview (normalized values)
        normalized_powers = [p / max(powers) * 100 for p in powers]
        normalized_efficiencies = [e / max(efficiencies) * 100 for e in efficiencies]

        fig.add_trace(
            go.Scatter(
                x=times,
                y=normalized_powers,
                mode='lines',
                name='Normalized Power',
                line=dict(color='blue', width=2, dash='dash'),
                hovertemplate='Time: %{x}<br>Power: %{y:.1f}%<extra></extra>'
            ),
            row=3, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=times,
                y=normalized_efficiencies,
                mode='lines',
                name='Normalized Efficiency',
                line=dict(color='purple', width=2, dash='dash'),
                hovertemplate='Time: %{x}<br>Efficiency: %{y:.1f}%<extra></extra>'
            ),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            title=title,
            height=1200,
            template=self.theme,
            showlegend=False
        )

        # Add second y-axis for current plot
        fig.update_yaxes(title_text="Current (A)", secondary_y=True, row=1, col=2)

        # Update other axes
        fig.update_xaxes(title_text="Time", row=3, col=2)
        fig.update_yaxes(title_text="Power (W)", row=1, col=1)
        fig.update_yaxes(title_text="Voltage (V)", row=1, col=2)
        fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)
        fig.update_yaxes(title_text="Degradation Factor", row=3, col=1)
        fig.update_yaxes(title_text="Normalized Value (%)", row=3, col=2)

        return fig

    def compare_scenarios(self, scenarios: Dict[str, List[LifetimeState]],
                         title: str = "Scenario Comparison") -> go.Figure:
        """
        Compare multiple mission scenarios

        Args:
            scenarios: Dictionary of scenario name to lifetime states
            title: Plot title

        Returns:
            Plotly figure object
        """
        if not scenarios:
            return go.Figure()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Power Output Comparison', 'Degradation Comparison',
                          'Efficiency Comparison', 'Performance Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        colors = self.color_palette[:len(scenarios)]

        for i, (scenario_name, states) in enumerate(scenarios.items()):
            if not states:
                continue

            color = colors[i % len(colors)]
            times = [state.time for state in states]
            powers = [state.current_power_watts for state in states]
            degradations = [state.power_degradation_percent for state in states]
            efficiencies = [state.efficiency_factor * 100 for state in states]

            # Power output
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=powers,
                    mode='lines+markers',
                    name=scenario_name,
                    line=dict(color=color, width=2),
                    hovertemplate=f'<b>{scenario_name}</b><br>' +
                                 'Time: %{x}<br>Power: %{y:.2f} W<extra></extra>'
                ),
                row=1, col=1
            )

            # Degradation
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=degradations,
                    mode='lines',
                    name=f'{scenario_name} - Degradation',
                    line=dict(color=color, width=2, dash='dash'),
                    showlegend=False,
                    hovertemplate=f'<b>{scenario_name}</b><br>' +
                                 'Time: %{x}<br>Degradation: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=2
            )

            # Efficiency
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=efficiencies,
                    mode='lines',
                    name=f'{scenario_name} - Efficiency',
                    line=dict(color=color, width=2, dash='dot'),
                    showlegend=False,
                    hovertemplate=f'<b>{scenario_name}</b><br>' +
                                 'Time: %{x}<br>Efficiency: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )

        # Performance metrics summary
        scenario_metrics = []
        for scenario_name, states in scenarios.items():
            if states:
                initial_power = states[0].current_power_watts
                final_power = states[-1].current_power_watts
                total_degradation = states[-1].power_degradation_percent
                avg_efficiency = np.mean([state.efficiency_factor for state in states]) * 100

                scenario_metrics.append({
                    'Scenario': scenario_name,
                    'Initial Power (W)': initial_power,
                    'Final Power (W)': final_power,
                    'Total Degradation (%)': total_degradation,
                    'Average Efficiency (%)': avg_efficiency
                })

        if scenario_metrics:
            metrics_df = pd.DataFrame(scenario_metrics)

            # Add bar chart for final power comparison
            fig.add_trace(
                go.Bar(
                    x=metrics_df['Scenario'],
                    y=metrics_df['Final Power (W)'],
                    name='Final Power',
                    marker_color=colors[:len(scenarios)],
                    hovertemplate='Scenario: %{x}<br>Final Power: %{y:.2f} W<extra></extra>'
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title=title,
            height=1000,
            template=self.theme,
            showlegend=True
        )

        # Update axes
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Scenario", row=2, col=2)
        fig.update_yaxes(title_text="Power (W)", row=1, col=1)
        fig.update_yaxes(title_text="Degradation (%)", row=1, col=2)
        fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
        fig.update_yaxes(title_text="Final Power (W)", row=2, col=2)

        return fig

    def create_dashboard(self, lifetime_states: List[LifetimeState],
                        power_outputs: List[PowerOutput],
                        thermal_states: List[ThermalState],
                        title: str = "Solar Panel Degradation Dashboard") -> go.Figure:
        """
        Create comprehensive dashboard with multiple visualizations

        Args:
            lifetime_states: Lifetime degradation states
            power_outputs: Power output data
            thermal_states: Thermal analysis states
            title: Dashboard title

        Returns:
            Plotly figure object with multiple subplots
        """
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Power Trend', 'Degradation Breakdown', 'Efficiency',
                'Temperature Profile', 'Radiation Exposure', 'Eclipse Events',
                'Current-Voltage Characteristics', 'Performance Metrics', 'Environmental Summary'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"type": "scatter3d"}]
            ]
        )

        # Add various plots to the dashboard
        # (Implementation would include all the individual plot types)
        # This is a placeholder for the complete dashboard

        # Update layout
        fig.update_layout(
            title=title,
            height=1600,
            template=self.theme,
            showlegend=False
        )

        return fig

    def export_plot(self, fig: go.Figure, filename: str,
                   format: str = "html", width: int = 1200, height: int = 800):
        """
        Export plot to various formats

        Args:
            fig: Plotly figure to export
            filename: Output filename
            format: Export format ("html", "png", "svg", "pdf")
            width: Image width in pixels
            height: Image height in pixels
        """
        if format.lower() == "html":
            fig.write_html(filename, include_plotlyjs='cdn')
        elif format.lower() == "png":
            fig.write_image(filename, width=width, height=height, format="png")
        elif format.lower() == "svg":
            fig.write_image(filename, width=width, height=height, format="svg")
        elif format.lower() == "pdf":
            fig.write_image(filename, width=width, height=height, format="pdf")
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def create_animation(self, data_over_time: List[Dict],
                       title: str = "Degradation Animation") -> go.Figure:
        """
        Create animated visualization showing degradation progression

        Args:
            data_over_time: List of data dictionaries for each time step
            title: Animation title

        Returns:
            Plotly figure with animation
        """
        fig = go.Figure()

        # Create frames for animation
        frames = []
        for i, data in enumerate(data_over_time):
            frame = go.Frame(
                data=[go.Scatter(
                    x=data.get('x', []),
                    y=data.get('y', []),
                    mode='markers+lines',
                    name=f'Time Step {i}',
                    marker=dict(size=8, color=data.get('colors', 'blue'))
                )],
                name=f'Step {i}'
            )
            frames.append(frame)

        fig.frames = frames

        # Add animation controls
        fig.update_layout(
            title=title,
            template=self.theme,
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play", method="animate",
                             args=[None, {"frame": {"duration": 500, "redraw": True},
                                          "fromcurrent": True}]),
                        dict(label="Pause", method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}])]
            )],
            sliders=[dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue={"prefix": "Time Step: "},
                pad={"b": 10, "t": 50},
                len=0.9,
                x=0.05,
                y=0,
                steps=[dict(args=[[f"Step {k}"], {"frame": {"duration": 0, "redraw": True},
                                                    "mode": "immediate"}],
                           label=str(k), method="animate") for k in range(len(data_over_time))]
            )]
        )

        return fig