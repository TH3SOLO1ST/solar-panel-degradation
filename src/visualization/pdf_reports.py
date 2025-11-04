"""
PDF Report Generation
====================

Generates professional PDF reports for solar panel degradation analysis.

This module creates comprehensive PDF reports with charts, tables,
and analysis summaries for presentation and documentation purposes.

Classes:
    PDFReportGenerator: Main PDF report generation class
    ReportSections: Individual report section generators
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import base64
import io

# Try to import reportlab, fall back to basic HTML to PDF if not available
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.widgetbase import Widget
    from reportlab.graphics import renderPDF
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not available, using simplified PDF generation")

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available, charts will be omitted from PDF")

try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

from ..degradation.lifetime_model import LifetimeResults

class ReportSections:
    """Generates individual sections of the PDF report"""

    def __init__(self, styles=None):
        """Initialize report sections"""
        if REPORTLAB_AVAILABLE:
            self.styles = styles or getSampleStyleSheet()
            self.title_style = ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            self.heading_style = ParagraphStyle(
                'CustomHeading',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                textColor=colors.darkblue
            )

    def generate_title_page(self, scenario_name: str, results: LifetimeResults) -> List:
        """Generate title page content"""
        content = []

        if REPORTLAB_AVAILABLE:
            # Main title
            content.append(Paragraph("Solar Panel Degradation Analysis", self.title_style))
            content.append(Spacer(1, 0.5*inch))

            # Scenario information
            content.append(Paragraph(f"<b>Scenario:</b> {scenario_name}", self.styles['Normal']))
            content.append(Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y')}", self.styles['Normal']))
            content.append(Paragraph(f"<b>Mission Duration:</b> {results.performance_metrics['mission_duration_years']:.1f} years", self.styles['Normal']))
            content.append(Spacer(1, 1*inch))

            # Key metrics
            content.append(Paragraph("Key Results", self.heading_style))
            content.append(Paragraph(f"Initial Power Output: {results.performance_metrics['initial_power_W']:.1f} W", self.styles['Normal']))
            content.append(Paragraph(f"Final Power Output: {results.performance_metrics['final_power_W']:.1f} W", self.styles['Normal']))
            content.append(Paragraph(f"Power Degradation: {results.performance_metrics['power_degradation_percent']:.1f}%", self.styles['Normal']))
            content.append(Paragraph(f"Total Energy Generated: {results.energy_yield['total_energy_kWh']:.1f} kWh", self.styles['Normal']))

        return content

    def generate_executive_summary(self, results: LifetimeResults) -> List:
        """Generate executive summary section"""
        content = []

        if REPORTLAB_AVAILABLE:
            content.append(Paragraph("Executive Summary", self.heading_style))

            # Summary narrative
            degradation_pct = results.performance_metrics['power_degradation_percent']
            energy_generated = results.energy_yield['total_energy_kWh']
            mission_years = results.performance_metrics['mission_duration_years']

            summary_text = f"""
            This analysis presents the lifetime performance prediction for solar panels in the specified space mission.
            Over the {mission_years:.1f}-year mission duration, the solar panels are predicted to generate approximately
            {energy_generated:.1f} kWh of electrical energy. The power output is expected to degrade by {degradation_pct:.1f}%,
            from an initial {results.performance_metrics['initial_power_W']:.1f} W to a final {results.performance_metrics['final_power_W']:.1f} W.
            """

            content.append(Paragraph(summary_text, self.styles['Normal']))
            content.append(Spacer(1, 0.3*inch))

            # Degradation breakdown
            content.append(Paragraph("Degradation Sources", self.styles['Heading3']))

            breakdown_data = [['Mechanism', 'Contribution (%)']]
            for mechanism, contribution in results.degradation_breakdown.items():
                breakdown_data.append([mechanism.replace('_', ' ').title(), f"{contribution*100:.2f}"])

            table = Table(breakdown_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            content.append(table)
            content.append(Spacer(1, 0.5*inch))

        return content

    def generate_performance_analysis(self, results: LifetimeResults) -> List:
        """Generate performance analysis section"""
        content = []

        if REPORTLAB_AVAILABLE:
            content.append(Paragraph("Performance Analysis", self.heading_style))

            # Performance metrics table
            metrics_data = [
                ['Metric', 'Value'],
                ['Mission Duration', f"{results.performance_metrics['mission_duration_years']:.1f} years"],
                ['Initial Power', f"{results.performance_metrics['initial_power_W']:.1f} W"],
                ['Final Power', f"{results.performance_metrics['final_power_W']:.1f} W"],
                ['Power Degradation', f"{results.performance_metrics['power_degradation_percent']:.1f}%"],
                ['Initial Efficiency', f"{results.performance_metrics['initial_efficiency']*100:.1f}%"],
                ['Final Efficiency', f"{results.performance_metrics['final_efficiency']*100:.1f}%"],
                ['Average Power', f"{results.performance_metrics['average_power_W']:.1f} W"],
                ['Performance Variability', f"{results.performance_metrics.get('performance_variability', 0)*100:.1f}%"],
                ['Total Energy', f"{results.energy_yield['total_energy_kWh']:.1f} kWh"],
                ['Capacity Factor', f"{results.energy_yield.get('capacity_factor', 0)*100:.1f}%"]
            ]

            table = Table(metrics_data, colWidths=[3*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9)
            ]))

            content.append(table)
            content.append(Spacer(1, 0.5*inch))

        return content

    def generate_environmental_analysis(self, results: LifetimeResults) -> List:
        """Generate environmental conditions analysis"""
        content = []

        if REPORTLAB_AVAILABLE:
            content.append(Paragraph("Environmental Conditions", self.heading_style))

            # Temperature analysis
            temps = results.environmental_conditions['temperatures']
            temp_min = float(min(temps)) - 273.15
            temp_max = float(max(temps)) - 273.15
            temp_mean = float(sum(temps)/len(temps)) - 273.15

            temp_text = f"""
            The solar panels experience significant temperature variations throughout the mission:
            <br/>• Minimum Temperature: {temp_min:.1f}°C
            <br/>• Maximum Temperature: {temp_max:.1f}°C
            <br/>• Average Temperature: {temp_mean:.1f}°C
            <br/>• Temperature Range: {temp_max - temp_min:.1f}°C
            """

            content.append(Paragraph("Thermal Environment", self.styles['Heading3']))
            content.append(Paragraph(temp_text, self.styles['Normal']))
            content.append(Spacer(1, 0.3*inch))

            # Radiation analysis
            radiation_dose = results.environmental_conditions['radiation_dose']
            radiation_text = f"""
            The radiation environment exposure:
            <br/>• Total Ionizing Dose: {radiation_dose.total_ionizing_dose:.1f} rads
            <br/>• Non-Ionizing Dose: {radiation_dose.non_ionizing_dose:.1f} MeV cm²/g
            <br/>• Electron Fluence: {radiation_dose.electron_fluence:.2e} electrons/cm²
            <br/>• Proton Fluence: {radiation_dose.proton_fluence:.2e} protons/cm²
            """

            content.append(Paragraph("Radiation Environment", self.styles['Heading3']))
            content.append(Paragraph(radiation_text, self.styles['Normal']))
            content.append(Spacer(1, 0.3*inch))

            # Solar exposure
            solar_exposure = results.environmental_conditions['solar_exposure_percentage']
            solar_text = f"""
            Solar exposure characteristics:
            <br/>• Total Solar Exposure: {solar_exposure:.1f}% of mission time
            <br/>• Eclipse Duration: {(100 - solar_exposure):.1f}% of mission time
            """

            content.append(Paragraph("Solar Exposure", self.styles['Heading3']))
            content.append(Paragraph(solar_text, self.styles['Normal']))

        return content

    def generate_recommendations(self, results: LifetimeResults) -> List:
        """Generate recommendations section"""
        content = []

        if REPORTLAB_AVAILABLE:
            content.append(Paragraph("Recommendations", self.heading_style))

            recommendations = []

            # Analyze degradation level
            degradation_pct = results.performance_metrics['power_degradation_percent']
            if degradation_pct > 30:
                recommendations.append("• Consider additional radiation shielding to reduce degradation")
                recommendations.append("• Evaluate use of radiation-hardened solar cell technology")
            elif degradation_pct > 20:
                recommendations.append("• Current degradation is within expected range for this orbit")
                recommendations.append("• Monitor performance during mission for validation")
            else:
                recommendations.append("• Low degradation rate indicates excellent design margins")

            # Temperature considerations
            temps = results.environmental_conditions['temperatures']
            temp_range = float(max(temps)) - float(min(temps))
            if temp_range > 150:
                recommendations.append("• Large temperature swings - ensure robust thermal management")
                recommendations.append("• Consider thermal interface materials to reduce stress")

            # Performance variability
            variability = results.performance_metrics.get('performance_variability', 0)
            if variability > 0.1:
                recommendations.append("• High performance variability - investigate orbit or attitude effects")

            # Energy yield
            energy_per_year = results.energy_yield['total_energy_kWh'] / results.performance_metrics['mission_duration_years']
            if energy_per_year < 1000:
                recommendations.append("• Consider increasing solar panel area for higher energy yield")

            if not recommendations:
                recommendations.append("• Solar panel design appears well-optimized for mission requirements")

            for rec in recommendations:
                content.append(Paragraph(rec, self.styles['Normal']))
                content.append(Spacer(1, 0.1*inch))

        return content

class PDFReportGenerator:
    """Main PDF report generation class"""

    def __init__(self):
        """Initialize PDF report generator"""
        self.sections = ReportSections()
        self.temp_dir = Path(tempfile.gettempdir()) / "solar_panel_reports"
        self.temp_dir.mkdir(exist_ok=True)

    def generate_comprehensive_report(self, results: LifetimeResults,
                                    scenario_name: str,
                                    filename: str = None) -> str:
        """
        Generate comprehensive PDF report

        Args:
            results: LifetimeResults object
            scenario_name: Name of the scenario
            filename: Optional custom filename

        Returns:
            Path to generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solar_panel_analysis_{timestamp}.pdf"

        filepath = self.temp_dir / filename

        if REPORTLAB_AVAILABLE:
            self._generate_with_reportlab(results, scenario_name, filepath)
        elif WEASYPRINT_AVAILABLE:
            self._generate_with_weasyprint(results, scenario_name, filepath)
        else:
            raise ImportError("Neither reportlab nor weasyprint available for PDF generation")

        return str(filepath)

    def _generate_with_reportlab(self, results: LifetimeResults,
                                scenario_name: str, filepath: Path):
        """Generate PDF using reportlab"""
        doc = SimpleDocTemplate(str(filepath), pagesize=A4,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)

        content = []

        # Title page
        content.extend(self.sections.generate_title_page(scenario_name, results))
        content.append(PageBreak())

        # Executive summary
        content.extend(self.sections.generate_executive_summary(results))
        content.append(Spacer(1, 0.5*inch))

        # Performance analysis
        content.extend(self.sections.generate_performance_analysis(results))
        content.append(PageBreak())

        # Environmental analysis
        content.extend(self.sections.generate_environmental_analysis(results))
        content.append(Spacer(1, 0.5*inch))

        # Recommendations
        content.extend(self.sections.generate_recommendations(results))

        # Build PDF
        doc.build(content)

    def _generate_with_weasyprint(self, results: LifetimeResults,
                                 scenario_name: str, filepath: Path):
        """Generate PDF using weasyprint (HTML to PDF)"""
        html_content = self._generate_html_report(results, scenario_name)

        # Convert HTML to PDF
        weasyprint.HTML(string=html_content).write_pdf(str(filepath))

    def _generate_html_report(self, results: LifetimeResults, scenario_name: str) -> str:
        """Generate HTML content for PDF conversion"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Solar Panel Degradation Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #1f3a93; border-bottom: 3px solid #1f3a93; padding-bottom: 10px; }}
                h2 {{ color: #1f3a93; margin-top: 30px; }}
                h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .metric {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #1f3a93; }}
                .recommendation {{ margin: 10px 0; padding-left: 20px; }}
                @page {{ margin: 2cm; }}
                @media print {{ body {{ margin: 2cm; }} }}
            </style>
        </head>
        <body>
            <h1>Solar Panel Degradation Analysis Report</h1>

            <div class="metric">
                <strong>Scenario:</strong> {scenario_name}<br>
                <strong>Analysis Date:</strong> {datetime.now().strftime('%B %d, %Y')}<br>
                <strong>Mission Duration:</strong> {results.performance_metrics['mission_duration_years']:.1f} years
            </div>

            <h2>Executive Summary</h2>
            <p>
            This analysis presents the lifetime performance prediction for solar panels in the specified space mission.
            Over the {results.performance_metrics['mission_duration_years']:.1f}-year mission duration, the solar panels are predicted
            to generate approximately {results.energy_yield['total_energy_kWh']:.1f} kWh of electrical energy.
            The power output is expected to degrade by {results.performance_metrics['power_degradation_percent']:.1f}%.
            </p>

            <h3>Key Results</h3>
            <div class="metric">
                <strong>Initial Power Output:</strong> {results.performance_metrics['initial_power_W']:.1f} W<br>
                <strong>Final Power Output:</strong> {results.performance_metrics['final_power_W']:.1f} W<br>
                <strong>Power Degradation:</strong> {results.performance_metrics['power_degradation_percent']:.1f}%<br>
                <strong>Total Energy Generated:</strong> {results.energy_yield['total_energy_kWh']:.1f} kWh
            </div>

            <h2>Performance Analysis</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Mission Duration</td><td>{results.performance_metrics['mission_duration_years']:.1f} years</td></tr>
                <tr><td>Initial Power</td><td>{results.performance_metrics['initial_power_W']:.1f} W</td></tr>
                <tr><td>Final Power</td><td>{results.performance_metrics['final_power_W']:.1f} W</td></tr>
                <tr><td>Power Degradation</td><td>{results.performance_metrics['power_degradation_percent']:.1f}%</td></tr>
                <tr><td>Average Power</td><td>{results.performance_metrics['average_power_W']:.1f} W</td></tr>
                <tr><td>Total Energy</td><td>{results.energy_yield['total_energy_kWh']:.1f} kWh</td></tr>
            </table>

            <h2>Environmental Conditions</h2>
            <h3>Thermal Environment</h3>
            <p>
            The solar panels experience significant temperature variations throughout the mission.
            Temperature range spans from the minimum to maximum values experienced in orbit.
            </p>

            <h3>Radiation Environment</h3>
            <p>
            Total ionizing dose: {results.environmental_conditions['radiation_dose'].total_ionizing_dose:.1f} rads<br>
            Non-ionizing dose: {results.environmental_conditions['radiation_dose'].non_ionizing_dose:.1f} MeV cm²/g
            </p>

            <h2>Recommendations</h2>
            <div class="recommendation">
                • Solar panel design appears well-optimized for mission requirements<br>
                • Continue monitoring performance during mission for validation<br>
                • Current degradation levels are within expected ranges
            </div>

            <p style="margin-top: 50px; font-size: 12px; color: #666;">
            <em>Report generated by Solar Panel Degradation Modeling Tool on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</em>
            </p>
        </body>
        </html>
        """
        return html_template

    def add_charts_to_report(self, results: LifetimeResults):
        """Add charts to the report if plotly is available"""
        if not PLOTLY_AVAILABLE:
            return []

        charts = []
        temp_dir = self.temp_dir / "charts"
        temp_dir.mkdir(exist_ok=True)

        try:
            # Generate power chart
            time_years = [h / (365.25 * 24) for h in results.time_hours]
            power_kw = [p / 1000 for p in results.power_output]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_years, y=power_kw, mode='lines', name='Power Output'))
            fig.update_layout(
                title='Solar Panel Power Output',
                xaxis_title='Mission Time (years)',
                yaxis_title='Power Output (kW)',
                template='plotly_white'
            )

            # Save as image
            chart_path = temp_dir / "power_chart.png"
            fig.write_image(str(chart_path), width=800, height=600, format='png')
            charts.append(str(chart_path))

        except Exception as e:
            print(f"Warning: Could not generate charts: {e}")

        return charts

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")