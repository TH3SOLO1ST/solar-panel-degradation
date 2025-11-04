"""
Interactive Plots
=================

Creates interactive visualizations using Plotly for solar panel degradation
analysis results.

This module provides a comprehensive set of plots for analyzing
solar panel performance over mission lifetime.

Classes:
    InteractivePlots: Main visualization class
    PlotStyler: Plot styling and configuration
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from ..degradation.lifetime_model import LifetimeResults

class PlotStyler:
    """Provides consistent styling for all plots"""

    def __init__(self):
        """Initialize plot styler"""
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'tertiary': '#2ca02c',
            'quaternary': '#d62728',
            'quinary': '#9467bd',
            'senary': '#8c564b',
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'danger': '#d62728'
        }

        self.template = 'plotly_white'
        self.font = dict(family="Arial, sans-serif", size=12, color="#1a1a1a")

    def apply_layout(self, fig, title: str = "", xaxis_title: str = "",
                    yaxis_title: str = ""):
        """Apply consistent layout to figure"""
        fig.update_layout(
            template=self.template,
            title=dict(text=title, font=dict(size=16, **self.font)),
            xaxis=dict(title=xaxis_title, **self.font),
            yaxis=dict(title=yaxis_title, **self.font),
            font=self.font,
            showlegend=True,
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='white'
        )
        return fig

class InteractivePlots:
    """Main interactive visualization class"""

    def __init__(self):
        """Initialize interactive plots"""
        self.styler = PlotStyler()

    def create_lifetime_power_plot(self, results: LifetimeResults) -> go.Figure:
        """
        Create lifetime power output plot

        Args:
            results: LifetimeResults object

        Returns:
            Plotly figure object
        """
        # Convert time to years for better readability
        time_years = results.time_hours / (365.25 * 24)
        power_kw = results.power_output / 1000.0

        fig = go.Figure()

        # Main power output line
        fig.add_trace(go.Scatter(
            x=time_years,
            y=power_kw,
            mode='lines',
            name='Power Output',
            line=dict(color=self.styler.color_palette['primary'], width=2),
            hovertemplate='<b>Year %{x:.2f}</b><br>' +
                         'Power: %{y:.2f} kW<br>' +
                         '<extra></extra>'
        ))

        # Initial power reference line
        initial_power = power_kw[0]
        fig.add_hline(
            y=initial_power,
            line_dash="dash",
            line_color=self.styler.color_palette['secondary'],
            annotation_text=f"Initial: {initial_power:.2f} kW"
        )

        # degradation zones
        final_power = power_kw[-1]
        degradation_pct = (1 - final_power/initial_power) * 100

        fig.add_annotation(
            x=time_years[-1],
            y=final_power,
            text=f"Final: {final_power:.2f} kW<br>Degradation: {degradation_pct:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=self.styler.color_palette['danger']
        )

        self.styler.apply_layout(
            fig,
            title="Solar Panel Power Output Over Mission Lifetime",
            xaxis_title="Mission Time (years)",
            yaxis_title="Power Output (kW)"
        )

        return fig

    def create_efficiency_plot(self, results: LifetimeResults) -> go.Figure:
        """
        Create efficiency degradation plot

        Args:
            results: LifetimeResults object

        Returns:
            Plotly figure object
        """
        time_years = results.time_hours / (365.25 * 24)
        efficiency_pct = results.efficiency * 100

        fig = go.Figure()

        # Efficiency line
        fig.add_trace(go.Scatter(
            x=time_years,
            y=efficiency_pct,
            mode='lines',
            name='Efficiency',
            line=dict(color=self.styler.color_palette['tertiary'], width=2),
            fill='tonexty',
            fillcolor=f'rgba(44, 160, 44, 0.1)',
            hovertemplate='<b>Year %{x:.2f}</b><br>' +
                         'Efficiency: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))

        # Mark degradation milestones
        initial_eff = efficiency_pct[0]
        degradation_levels = [0.95, 0.90, 0.85, 0.80]  # 5%, 10%, 15%, 20% degradation

        for level in degradation_levels:
            target_eff = initial_eff * level
            idx = np.where(efficiency_pct <= target_eff)[0]
            if len(idx) > 0:
                first_idx = idx[0]
                fig.add_trace(go.Scatter(
                    x=[time_years[first_idx]],
                    y=[efficiency_pct[first_idx]],
                    mode='markers',
                    name=f'{(1-level)*100:.0f}% Degradation',
                    marker=dict(
                        size=8,
                        color=self.styler.color_palette['warning'],
                        symbol='diamond'
                    ),
                    showlegend=(level == 0.95)  # Only show first in legend
                ))

        self.styler.apply_layout(
            fig,
            title="Solar Cell Efficiency Degradation",
            xaxis_title="Mission Time (years)",
            yaxis_title="Efficiency (%)"
        )

        return fig

    def create_degradation_breakdown_plot(self, results: LifetimeResults) -> go.Figure:
        """
        Create degradation breakdown by mechanism

        Args:
            results: LifetimeResults object

        Returns:
            Plotly figure object
        """
        breakdown = results.degradation_breakdown

        # Prepare data for pie chart
        mechanisms = list(breakdown.keys())
        values = list(breakdown.values())
        labels = [mechanism.replace('_', ' ').title() for mechanism in mechanisms]

        # Filter out zero values
        non_zero_data = [(label, value) for label, value in zip(labels, values) if value > 0]
        if not non_zero_data:
            # No degradation data
            fig = go.Figure()
            self.styler.apply_layout(fig, title="No Significant Degradation Detected")
            return fig

        labels, values = zip(*non_zero_data)

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=[
                self.styler.color_palette['quaternary'],
                self.styler.color_palette['quinary'],
                self.styler.color_palette['senary'],
                self.styler.color_palette['warning']
            ][:len(labels)],
            hovertemplate='<b>%{label}</b><br>' +
                         'Contribution: %{value:.1f}%<br>' +
                         '<extra></extra>'
        )])

        self.styler.apply_layout(
            fig,
            title="Degradation Sources Breakdown",
            xaxis_title="",
            yaxis_title=""
        )

        return fig

    def create_temperature_profile_plot(self, results: LifetimeResults) -> go.Figure:
        """
        Create temperature profile plot

        Args:
            results: LifetimeResults object

        Returns:
            Plotly figure object
        """
        time_days = results.time_hours / 24.0
        temperatures_c = results.environmental_conditions['temperatures'] - 273.15

        fig = go.Figure()

        # Temperature line
        fig.add_trace(go.Scatter(
            x=time_days,
            y=temperatures_c,
            mode='lines',
            name='Temperature',
            line=dict(color=self.styler.color_palette['secondary'], width=1),
            opacity=0.7,
            hovertemplate='<b>Day %{x:.1f}</b><br>' +
                         'Temperature: %{y:.1f}°C<br>' +
                         '<extra></extra>'
        ))

        # Add statistical bands
        temp_mean = np.mean(temperatures_c)
        temp_std = np.std(temperatures_c)

        fig.add_hline(
            y=temp_mean,
            line_dash="dash",
            line_color=self.styler.color_palette['primary'],
            annotation_text=f"Mean: {temp_mean:.1f}°C"
        )

        fig.add_hrect(
            y0=temp_mean - temp_std,
            y1=temp_mean + temp_std,
            fillcolor="rgba(31, 119, 180, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="±1σ"
        )

        # Mark extreme temperatures
        max_temp_idx = np.argmax(temperatures_c)
        min_temp_idx = np.argmin(temperatures_c)

        fig.add_trace(go.Scatter(
            x=[time_days[max_temp_idx], time_days[min_temp_idx]],
            y=[temperatures_c[max_temp_idx], temperatures_c[min_temp_idx]],
            mode='markers',
            name='Extremes',
            marker=dict(
                size=8,
                color=[self.styler.color_palette['danger'], self.styler.color_palette['primary']],
                symbol=['circle', 'circle']
            ),
            text=[f"Max: {temperatures_c[max_temp_idx]:.1f}°C",
                  f"Min: {temperatures_c[min_temp_idx]:.1f}°C"],
            textposition="top center",
            showlegend=False
        ))

        self.styler.apply_layout(
            fig,
            title="Solar Panel Temperature Profile",
            xaxis_title="Mission Time (days)",
            yaxis_title="Temperature (°C)"
        )

        return fig

    def create_multi_plot_dashboard(self, results: LifetimeResults) -> go.Figure:
        """
        Create comprehensive dashboard with multiple subplots

        Args:
            results: LifetimeResults object

        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Power Output', 'Efficiency', 'Temperature', 'Energy Generation'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Convert time units
        time_years = results.time_hours / (365.25 * 24)
        time_days = results.time_hours / 24.0
        power_kw = results.power_output / 1000.0
        efficiency_pct = results.efficiency * 100
        temperatures_c = results.environmental_conditions['temperatures'] - 273.15

        # Power output plot
        fig.add_trace(
            go.Scatter(x=time_years, y=power_kw, mode='lines',
                      name='Power', line=dict(color=self.styler.color_palette['primary'])),
            row=1, col=1
        )

        # Efficiency plot
        fig.add_trace(
            go.Scatter(x=time_years, y=efficiency_pct, mode='lines',
                      name='Efficiency', line=dict(color=self.styler.color_palette['tertiary']),
                      showlegend=False),
            row=1, col=2
        )

        # Temperature plot (sample every 10th point for performance)
        temp_sample = slice(None, None, 10)
        fig.add_trace(
            go.Scatter(x=time_days[temp_sample], y=temperatures_c[temp_sample],
                      mode='lines', name='Temperature',
                      line=dict(color=self.styler.color_palette['secondary']),
                      showlegend=False, opacity=0.7),
            row=2, col=1
        )

        # Energy generation (cumulative)
        energy_cumulative = np.cumsum(power_kw) * (results.time_hours[1] - results.time_hours[0])  # kWh
        fig.add_trace(
            go.Scatter(x=time_years, y=energy_cumulative, mode='lines',
                      name='Cumulative Energy', line=dict(color=self.styler.color_palette['quaternary']),
                      showlegend=False),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            template=self.styler.template,
            title_text="Solar Panel Performance Dashboard",
            title_x=0.5,
            height=600,
            font=self.styler.font,
            showlegend=True
        )

        # Update subplot axes
        fig.update_xaxes(title_text="Time (years)", row=1, col=1)
        fig.update_xaxes(title_text="Time (years)", row=1, col=2)
        fig.update_xaxes(title_text="Time (days)", row=2, col=1)
        fig.update_xaxes(title_text="Time (years)", row=2, col=2)

        fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
        fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
        fig.update_yaxes(title_text="Energy (kWh)", row=2, col=2)

        return fig

    def create_performance_comparison_plot(self, results_list: List[LifetimeResults],
                                         scenario_names: List[str]) -> go.Figure:
        """
        Create comparison plot for multiple scenarios

        Args:
            results_list: List of LifetimeResults objects
            scenario_names: List of scenario names

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        colors = [
            self.styler.color_palette['primary'],
            self.styler.color_palette['secondary'],
            self.styler.color_palette['tertiary'],
            self.styler.color_palette['quaternary']
        ]

        for i, (results, name) in enumerate(zip(results_list, scenario_names)):
            time_years = results.time_hours / (365.25 * 24)
            power_kw = results.power_output / 1000.0

            fig.add_trace(go.Scatter(
                x=time_years,
                y=power_kw,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{name}</b><br>' +
                             'Year: %{{x:.2f}}<br>' +
                             'Power: %{{y:.2f}} kW<br>' +
                             '<extra></extra>'
            ))

        self.styler.apply_layout(
            fig,
            title="Scenario Comparison: Power Output",
            xaxis_title="Mission Time (years)",
            yaxis_title="Power Output (kW)"
        )

        return fig

    def export_plot(self, fig: go.Figure, filename: str, format: str = 'html',
                   width: int = 1200, height: int = 600) -> str:
        """
        Export plot to file

        Args:
            fig: Plotly figure
            filename: Output filename
            format: Export format ('html', 'png', 'pdf', 'svg')
            width: Image width
            height: Image height

        Returns:
            Path to exported file
        """
        if format.lower() == 'html':
            fig.write_html(filename, include_plotlyjs='cdn')
        elif format.lower() == 'png':
            fig.write_image(filename, width=width, height=height)
        elif format.lower() == 'pdf':
            fig.write_image(filename, width=width, height=height)
        elif format.lower() == 'svg':
            fig.write_image(filename, width=width, height=height)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        return filename