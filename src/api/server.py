"""
API Server
==========

Flask-based REST API server for solar panel degradation application.

Provides endpoints for scenario management, simulation execution,
and results retrieval for the web frontend.

Routes:
    GET /api/scenarios - Get available scenarios
    POST /api/scenarios - Create custom scenario
    POST /api/simulation/start - Start simulation
    GET /api/simulation/status/{id} - Get simulation status
    GET /api/simulation/results/{id} - Get simulation results
    POST /api/export/pdf - Export PDF report
    POST /api/export/data - Export data files
"""

import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from typing import Dict, Any, Optional
import threading
import traceback

# Import our modules
from ..config.scenario_config import ScenarioConfig, SatelliteSpecs
from ..degradation.lifetime_model import LifetimeModel
from ..degradation.power_calculator import SolarCellSpecs
from ..orbital.orbit_propagator import OrbitElements
from ..thermal.thermal_analysis import ThermalProperties
from ..visualization.data_export import DataExporter

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global state for simulations
simulations = {}
scenario_config = ScenarioConfig()

class SimulationManager:
    """Manages running simulations"""

    def __init__(self):
        self.active_simulations = {}
        self.simulation_threads = {}

    def start_simulation(self, simulation_id: str, scenario: SatelliteSpecs,
                        progress_callback=None) -> str:
        """Start a new simulation"""
        try:
            # Initialize simulation state
            self.active_simulations[simulation_id] = {
                'status': 'initializing',
                'progress': 0.0,
                'start_time': datetime.now(),
                'scenario': scenario,
                'results': None,
                'error': None
            }

            # Start simulation in background thread
            thread = threading.Thread(
                target=self._run_simulation,
                args=(simulation_id, scenario, progress_callback)
            )
            thread.daemon = True
            thread.start()
            self.simulation_threads[simulation_id] = thread

            return simulation_id

        except Exception as e:
            self.active_simulations[simulation_id]['error'] = str(e)
            self.active_simulations[simulation_id]['status'] = 'error'
            raise e

    def _run_simulation(self, simulation_id: str, scenario: SatelliteSpecs,
                       progress_callback=None):
        """Run the actual simulation"""
        try:
            # Update status
            self.active_simulations[simulation_id]['status'] = 'running'
            if progress_callback:
                progress_callback(simulation_id, 'running', 0.1)

            # Convert scenario to model inputs
            solar_specs = SolarCellSpecs(
                technology=scenario.solar_panel_tech,
                area_m2=scenario.panel_area_m2,
                initial_efficiency=scenario.initial_efficiency,
                series_resistance=0.01,
                shunt_resistance=1000.0,
                ideality_factor=1.2,
                temperature_coefficient=-0.0045,
                reference_temperature=298.0
            )

            orbit_elements = OrbitElements(
                semi_major_axis_km=6371.0 + scenario.altitude_km,  # Earth radius + altitude
                eccentricity=scenario.eccentricity,
                inclination_deg=scenario.inclination_deg,
                raan_deg=0.0,
                arg_perigee_deg=0.0,
                mean_anomaly_deg=0.0,
                epoch=datetime.now()
            )

            thermal_props = ThermalProperties(
                mass=scenario.panel_area_m2 * 2.0,  # 2 kg/m²
                specific_heat=900.0,  # J/(kg·K) for aluminum
                emissivity=0.85,
                absorptivity=0.9,
                area=scenario.panel_area_m2
            )

            # Create lifetime model
            lifetime_model = LifetimeModel(solar_specs, orbit_elements, thermal_props)

            if progress_callback:
                progress_callback(simulation_id, 'running', 0.3)

            # Run simulation
            results = lifetime_model.run_lifetime_simulation(scenario.mission_duration_years)

            if progress_callback:
                progress_callback(simulation_id, 'running', 0.9)

            # Store results
            self.active_simulations[simulation_id]['results'] = results
            self.active_simulations[simulation_id]['status'] = 'completed'
            self.active_simulations[simulation_id]['progress'] = 1.0

            if progress_callback:
                progress_callback(simulation_id, 'completed', 1.0)

        except Exception as e:
            error_msg = f"Simulation error: {str(e)}"
            print(f"Simulation {simulation_id} failed: {error_msg}")
            print(traceback.format_exc())

            self.active_simulations[simulation_id]['error'] = error_msg
            self.active_simulations[simulation_id]['status'] = 'error'

    def get_simulation_status(self, simulation_id: str) -> Optional[Dict]:
        """Get status of a simulation"""
        return self.active_simulations.get(simulation_id)

    def get_simulation_results(self, simulation_id: str) -> Optional[Any]:
        """Get results of a completed simulation"""
        sim = self.active_simulations.get(simulation_id)
        if sim and sim['status'] == 'completed':
            return sim['results']
        return None

# Initialize simulation manager
sim_manager = SimulationManager()

def progress_callback(simulation_id: str, status: str, progress: float):
    """Progress callback for simulations"""
    if simulation_id in sim_manager.active_simulations:
        sim_manager.active_simulations[simulation_id]['status'] = status
        sim_manager.active_simulations[simulation_id]['progress'] = progress

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/scenarios', methods=['GET'])
def get_scenarios():
    """Get all available scenarios"""
    try:
        preset_scenarios = scenario_config.get_preset_scenarios()
        scenarios_data = []

        for scenario in preset_scenarios:
            summary = scenario_config.get_scenario_summary(scenario)
            scenarios_data.append({
                'id': scenario.name,
                'name': scenario.name,
                'description': scenario.description,
                'type': 'preset',
                'summary': summary
            })

        return jsonify({
            'success': True,
            'scenarios': scenarios_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/scenarios/<scenario_name>', methods=['GET'])
def get_scenario_details(scenario_name: str):
    """Get details of a specific scenario"""
    try:
        scenario = scenario_config.get_scenario_by_name(scenario_name)
        if not scenario:
            return jsonify({
                'success': False,
                'error': f'Scenario not found: {scenario_name}'
            }), 404

        return jsonify({
            'success': True,
            'scenario': {
                'name': scenario.name,
                'description': scenario.description,
                'orbit_type': scenario.orbit_type,
                'altitude_km': scenario.altitude_km,
                'inclination_deg': scenario.inclination_deg,
                'eccentricity': scenario.eccentricity,
                'period_minutes': scenario.period_minutes,
                'solar_panel_tech': scenario.solar_panel_tech,
                'panel_area_m2': scenario.panel_area_m2,
                'initial_efficiency': scenario.initial_efficiency,
                'mission_duration_years': scenario.mission_duration_years,
                'expected_degradation_pct': scenario.expected_degradation_pct
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/scenarios', methods=['POST'])
def create_custom_scenario():
    """Create a custom scenario"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        # Validate required fields
        required_fields = ['name', 'description', 'altitude_km', 'mission_duration_years']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        # Create scenario
        scenario = scenario_config.create_custom_scenario(data)

        # Validate scenario
        warnings = scenario_config.validate_scenario(scenario)

        return jsonify({
            'success': True,
            'scenario': {
                'name': scenario.name,
                'description': scenario.description,
                'orbit_type': scenario.orbit_type,
                'altitude_km': scenario.altitude_km,
                'inclination_deg': scenario.inclination_deg,
                'eccentricity': scenario.eccentricity,
                'period_minutes': scenario.period_minutes,
                'solar_panel_tech': scenario.solar_panel_tech,
                'panel_area_m2': scenario.panel_area_m2,
                'initial_efficiency': scenario.initial_efficiency,
                'mission_duration_years': scenario.mission_duration_years,
                'expected_degradation_pct': scenario.expected_degradation_pct
            },
            'warnings': warnings
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Start a new simulation"""
    try:
        data = request.get_json()
        if not data or 'scenario_name' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing scenario_name'
            }), 400

        scenario_name = data['scenario_name']
        scenario = scenario_config.get_scenario_by_name(scenario_name)
        if not scenario:
            return jsonify({
                'success': False,
                'error': f'Scenario not found: {scenario_name}'
            }), 404

        # Generate unique simulation ID
        simulation_id = str(uuid.uuid4())

        # Start simulation
        sim_manager.start_simulation(simulation_id, scenario, progress_callback)

        return jsonify({
            'success': True,
            'simulation_id': simulation_id,
            'estimated_time_minutes': scenario.mission_duration_years * 0.5  # Rough estimate
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/simulation/status/<simulation_id>', methods=['GET'])
def get_simulation_status(simulation_id: str):
    """Get status of a running simulation"""
    try:
        status = sim_manager.get_simulation_status(simulation_id)
        if not status:
            return jsonify({
                'success': False,
                'error': 'Simulation not found'
            }), 404

        return jsonify({
            'success': True,
            'simulation_id': simulation_id,
            'status': status['status'],
            'progress': status['progress'],
            'start_time': status['start_time'].isoformat(),
            'error': status.get('error')
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/simulation/results/<simulation_id>', methods=['GET'])
def get_simulation_results(simulation_id: str):
    """Get results of a completed simulation"""
    try:
        results = sim_manager.get_simulation_results(simulation_id)
        if not results:
            status = sim_manager.get_simulation_status(simulation_id)
            if not status:
                return jsonify({
                    'success': False,
                    'error': 'Simulation not found'
                }), 404
            elif status['status'] != 'completed':
                return jsonify({
                    'success': False,
                    'error': f'Simulation not completed. Status: {status["status"]}'
                }), 400
            else:
                return jsonify({
                    'success': False,
                    'error': 'Simulation results not available'
                }), 500

        # Convert results to JSON-serializable format
        results_data = {
            'time_hours': results.time_hours.tolist(),
            'power_output': results.power_output.tolist(),
            'efficiency': results.efficiency.tolist(),
            'degradation_breakdown': results.degradation_breakdown,
            'energy_yield': results.energy_yield,
            'performance_metrics': results.performance_metrics,
            'environmental_summary': {
                'temperature_range_K': {
                    'min': float(np.min(results.environmental_conditions['temperatures'])),
                    'max': float(np.max(results.environmental_conditions['temperatures'])),
                    'mean': float(np.mean(results.environmental_conditions['temperatures']))
                },
                'total_radiation_dose': float(results.environmental_conditions['radiation_dose'].total_ionizing_dose),
                'solar_exposure_percentage': float(results.environmental_conditions['solar_exposure_pct'])
            }
        }

        return jsonify({
            'success': True,
            'simulation_id': simulation_id,
            'results': results_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export/data/<simulation_id>', methods=['POST'])
def export_data(simulation_id: str):
    """Export simulation data"""
    try:
        data = request.get_json()
        export_format = data.get('format', 'csv') if data else 'csv'

        results = sim_manager.get_simulation_results(simulation_id)
        if not results:
            return jsonify({
                'success': False,
                'error': 'Simulation results not found'
            }), 404

        # Create data exporter
        exporter = DataExporter()

        if export_format == 'csv':
            file_path = exporter.export_csv(results, simulation_id)
        elif export_format == 'json':
            file_path = exporter.export_json(results, simulation_id)
        elif export_format == 'excel':
            file_path = exporter.export_excel(results, simulation_id)
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported export format: {export_format}'
            }), 400

        return send_file(
            file_path,
            as_attachment=True,
            download_name=f'simulation_results_{simulation_id[:8]}.{export_format}'
        )

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def create_app():
    """Create and configure Flask app"""
    return app

def run_server(host='127.0.0.1', port=5000, debug=False):
    """Run the API server"""
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_server(debug=True)

# Add numpy import for data processing
import numpy as np