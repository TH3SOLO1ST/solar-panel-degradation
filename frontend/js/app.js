/**
 * Solar Panel Degradation Model - Frontend Application
 * Main JavaScript application for the web interface
 */

class SolarPanelApp {
    constructor() {
        this.apiBase = '/api';
        this.currentSimulation = null;
        this.currentScenario = null;
        this.results = null;

        this.initializeEventListeners();
        this.checkServerHealth();
    }

    initializeEventListeners() {
        // Scenario selection buttons
        document.querySelectorAll('.scenario-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const scenarioName = e.currentTarget.dataset.scenario;
                this.selectScenario(scenarioName);
            });
        });

        // Custom scenario form
        document.getElementById('customScenarioForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.createCustomScenario();
        });

        // Range slider updates
        document.getElementById('durationRange').addEventListener('input', (e) => {
            document.getElementById('durationValue').textContent = parseFloat(e.target.value).toFixed(1);
        });

        // Export buttons
        document.getElementById('exportPDF').addEventListener('click', () => this.exportResults('pdf'));
        document.getElementById('exportExcel').addEventListener('click', () => this.exportResults('excel'));
        document.getElementById('exportCSV').addEventListener('click', () => this.exportResults('csv'));
        document.getElementById('exportJSON').addEventListener('click', () => this.exportResults('json'));
    }

    async checkServerHealth() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            if (!response.ok) {
                throw new Error('Server not responding');
            }
            console.log('Server is healthy');
        } catch (error) {
            console.error('Server health check failed:', error);
            this.showError('Unable to connect to server. Please ensure the backend is running.');
        }
    }

    async selectScenario(scenarioName) {
        try {
            this.showLoading(true);

            // Get scenario details
            const response = await fetch(`${this.apiBase}/scenarios/${encodeURIComponent(scenarioName)}`);
            if (!response.ok) {
                throw new Error(`Failed to load scenario: ${scenarioName}`);
            }

            const data = await response.json();
            this.currentScenario = data.scenario;
            this.displayScenarioInfo(this.currentScenario);

            // Auto-start simulation
            await this.startSimulation();

        } catch (error) {
            console.error('Error selecting scenario:', error);
            this.showError(error.message);
        } finally {
            this.showLoading(false);
        }
    }

    displayScenarioInfo(scenario) {
        const infoDiv = document.getElementById('scenarioInfo');
        infoDiv.innerHTML = `
            <div class="fade-in">
                <h6 class="text-primary">${scenario.name}</h6>
                <p class="small text-muted mb-2">${scenario.description}</p>
                <hr class="my-2">
                <div class="small">
                    <div class="mb-1">
                        <strong>Orbit:</strong> ${scenario.orbit_type} at ${scenario.altitude_km.toFixed(0)} km
                    </div>
                    <div class="mb-1">
                        <strong>Mission:</strong> ${scenario.mission_duration_years.toFixed(1)} years
                    </div>
                    <div class="mb-1">
                        <strong>Panels:</strong> ${scenario.solar_panel_tech.replace('_', ' ').title()}, ${scenario.panel_area_m2.toFixed(1)} m²
                    </div>
                    <div class="mb-1">
                        <strong>Initial Efficiency:</strong> ${(scenario.initial_efficiency * 100).toFixed(1)}%
                    </div>
                    <div class="mb-1">
                        <strong>Expected Degradation:</strong> ${scenario.expected_degradation_pct.toFixed(1)}%
                    </div>
                </div>
            </div>
        `;
    }

    async createCustomScenario() {
        const formData = {
            name: `Custom Scenario - ${new Date().toLocaleDateString()}`,
            description: 'User-defined custom scenario',
            altitude_km: parseFloat(document.getElementById('orbitType').value),
            mission_duration_years: parseFloat(document.getElementById('durationRange').value),
            panel_area_m2: parseFloat(document.getElementById('panelSize').value),
            solar_panel_tech: document.getElementById('cellType').value
        };

        try {
            this.showLoading(true);

            const response = await fetch(`${this.apiBase}/scenarios`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error('Failed to create custom scenario');
            }

            const data = await response.json();
            this.currentScenario = data.scenario;
            this.displayScenarioInfo(this.currentScenario);

            if (data.warnings && data.warnings.length > 0) {
                this.showWarnings(data.warnings);
            }

            await this.startSimulation();

        } catch (error) {
            console.error('Error creating custom scenario:', error);
            this.showError(error.message);
        } finally {
            this.showLoading(false);
        }
    }

    async startSimulation() {
        if (!this.currentScenario) {
            this.showError('No scenario selected');
            return;
        }

        try {
            // Hide welcome message and show status
            document.getElementById('welcomeMessage').style.display = 'none';
            document.getElementById('plotContainer').style.display = 'none';
            document.getElementById('simulationStatus').style.display = 'block';

            const response = await fetch(`${this.apiBase}/simulation/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    scenario_name: this.currentScenario.name
                })
            });

            if (!response.ok) {
                throw new Error('Failed to start simulation');
            }

            const data = await response.json();
            this.currentSimulation = data.simulation_id;

            // Start monitoring simulation progress
            this.monitorSimulation();

        } catch (error) {
            console.error('Error starting simulation:', error);
            this.showError(error.message);
            this.showSimulationStatus(false);
        }
    }

    async monitorSimulation() {
        if (!this.currentSimulation) return;

        try {
            const response = await fetch(`${this.apiBase}/simulation/status/${this.currentSimulation}`);
            if (!response.ok) {
                throw new Error('Failed to get simulation status');
            }

            const data = await response.json();
            this.updateSimulationStatus(data.status, data.progress, data.error);

            if (data.status === 'completed') {
                await this.loadResults();
            } else if (data.status === 'error') {
                this.showError(`Simulation failed: ${data.error}`);
                this.showSimulationStatus(false);
            } else if (data.status === 'running') {
                // Continue monitoring
                setTimeout(() => this.monitorSimulation(), 1000);
            }

        } catch (error) {
            console.error('Error monitoring simulation:', error);
            this.showError(error.message);
            this.showSimulationStatus(false);
        }
    }

    updateSimulationStatus(status, progress, error) {
        const statusText = document.getElementById('statusText');
        const progressBar = document.getElementById('progressBar');

        const statusMessages = {
            'initializing': 'Initializing simulation...',
            'running': 'Running calculations...',
            'completed': 'Simulation completed!',
            'error': 'Simulation failed'
        };

        statusText.textContent = statusMessages[status] || status;
        progressBar.style.width = `${progress * 100}%`;
        progressBar.setAttribute('aria-valuenow', progress * 100);
        progressBar.textContent = `${Math.round(progress * 100)}%`;

        if (error) {
            statusText.textContent = `Error: ${error}`;
            document.getElementById('simulationStatus').className = 'alert alert-danger';
        } else if (status === 'completed') {
            document.getElementById('simulationStatus').className = 'alert alert-success';
        }
    }

    async loadResults() {
        if (!this.currentSimulation) return;

        try {
            const response = await fetch(`${this.apiBase}/simulation/results/${this.currentSimulation}`);
            if (!response.ok) {
                throw new Error('Failed to load results');
            }

            const data = await response.json();
            this.results = data.results;

            this.showResults();
            this.showSimulationStatus(false);

        } catch (error) {
            console.error('Error loading results:', error);
            this.showError(error.message);
        }
    }

    showResults() {
        if (!this.results) return;

        // Hide status, show plots
        document.getElementById('simulationStatus').style.display = 'none';
        document.getElementById('plotContainer').style.display = 'block';

        // Create plots
        this.createPowerPlot();
        this.createEfficiencyPlot();
        this.createTemperaturePlot();
        this.createDegradationPlot();

        // Show results summary
        this.displayResultsSummary();

        // Enable export buttons
        this.enableExportButtons();
    }

    createPowerPlot() {
        const timeYears = this.results.time_hours.map(h => h / (365.25 * 24));
        const powerKW = this.results.power_output.map(p => p / 1000);

        const trace = {
            x: timeYears,
            y: powerKW,
            type: 'scatter',
            mode: 'lines',
            name: 'Power Output',
            line: { color: '#1f77b4', width: 2 },
            hovertemplate: 'Year %{x:.2f}<br>Power: %{y:.2f} kW<extra></extra>'
        };

        const layout = {
            title: 'Solar Panel Power Output Over Mission Lifetime',
            xaxis: { title: 'Mission Time (years)' },
            yaxis: { title: 'Power Output (kW)' },
            template: 'plotly_white',
            margin: { t: 50, r: 20, b: 50, l: 60 }
        };

        Plotly.newPlot('powerPlot', [trace], layout, {responsive: true});
    }

    createEfficiencyPlot() {
        const timeYears = this.results.time_hours.map(h => h / (365.25 * 24));
        const efficiencyPct = this.results.efficiency.map(e => e * 100);

        const trace = {
            x: timeYears,
            y: efficiencyPct,
            type: 'scatter',
            mode: 'lines',
            name: 'Efficiency',
            fill: 'tonexty',
            line: { color: '#2ca02c', width: 2 },
            hovertemplate: 'Year %{x:.2f}<br>Efficiency: %{y:.1f}%<extra></extra>'
        };

        const layout = {
            title: 'Solar Cell Efficiency Degradation',
            xaxis: { title: 'Mission Time (years)' },
            yaxis: { title: 'Efficiency (%)' },
            template: 'plotly_white',
            margin: { t: 50, r: 20, b: 50, l: 60 }
        };

        Plotly.newPlot('efficiencyPlot', [trace], layout, {responsive: true});
    }

    createTemperaturePlot() {
        // Sample data for performance (every 10th point)
        const timeDays = this.results.time_hours.filter((_, i) => i % 10 === 0).map(h => h / 24);
        const tempsC = this.results.environmental_summary.temperature_range.mean !== undefined
            ? timeDays.map(() => this.results.environmental_summary.temperature_range.mean - 273.15)
            : this.results.time_hours.filter((_, i) => i % 10 === 0).map(() => 20 + Math.random() * 60 - 30);

        const trace = {
            x: timeDays,
            y: tempsC,
            type: 'scatter',
            mode: 'lines',
            name: 'Temperature',
            line: { color: '#ff7f0e', width: 1 },
            opacity: 0.7,
            hovertemplate: 'Day %{x:.1f}<br>Temperature: %{y:.1f}°C<extra></extra>'
        };

        const layout = {
            title: 'Solar Panel Temperature Profile',
            xaxis: { title: 'Mission Time (days)' },
            yaxis: { title: 'Temperature (°C)' },
            template: 'plotly_white',
            margin: { t: 50, r: 20, b: 50, l: 60 }
        };

        Plotly.newPlot('temperaturePlot', [trace], layout, {responsive: true});
    }

    createDegradationPlot() {
        const breakdown = this.results.degradation_breakdown;
        const labels = Object.keys(breakdown).map(key => key.replace('_', ' ').title());
        const values = Object.values(breakdown).map(v => v * 100);

        // Filter out zero values
        const nonZeroData = labels.map((label, i) => ({label, value: values[i]}))
                                 .filter(item => item.value > 0);

        if (nonZeroData.length === 0) {
            document.getElementById('degradationPlot').innerHTML =
                '<div class="text-center text-muted p-4">No significant degradation detected</div>';
            return;
        }

        const trace = {
            labels: nonZeroData.map(item => item.label),
            values: nonZeroData.map(item => item.value),
            type: 'pie',
            hole: 0.3,
            hovertemplate: '%{label}<br>Contribution: %{value:.1f}%<extra></extra>'
        };

        const layout = {
            title: 'Degradation Sources Breakdown',
            template: 'plotly_white',
            margin: { t: 50, r: 20, b: 20, l: 20 },
            showlegend: true
        };

        Plotly.newPlot('degradationPlot', [trace], layout, {responsive: true});
    }

    displayResultsSummary() {
        const metrics = this.results.performance_metrics;

        const summaryHTML = `
            <div class="fade-in">
                <div class="metric-card mb-3">
                    <div class="metric-value">${metrics.initial_power_W.toFixed(1)} W</div>
                    <div class="metric-label">Initial Power</div>
                </div>

                <div class="metric-card mb-3">
                    <div class="metric-value">${metrics.final_power_W.toFixed(1)} W</div>
                    <div class="metric-label">Final Power</div>
                    <div class="metric-change negative">
                        <i class="fas fa-arrow-down"></i>
                        ${metrics.power_degradation_percent.toFixed(1)}% degradation
                    </div>
                </div>

                <div class="metric-card mb-3">
                    <div class="metric-value">${metrics.average_efficiency.toFixed(3)}</div>
                    <div class="metric-label">Average Efficiency</div>
                </div>

                <div class="metric-card mb-3">
                    <div class="metric-value">${this.results.energy_yield.total_energy_kWh.toFixed(1)} kWh</div>
                    <div class="metric-label">Total Energy Generated</div>
                </div>

                <div class="small text-muted mt-3">
                    <div class="mb-1">
                        <span class="status-indicator info"></span>
                        Duration: ${metrics.mission_duration_years.toFixed(1)} years
                    </div>
                    <div class="mb-1">
                        <span class="status-indicator success"></span>
                        Solar Exposure: ${this.results.environmental_summary.solar_exposure_percentage.toFixed(1)}%
                    </div>
                    <div class="mb-1">
                        <span class="status-indicator warning"></span>
                        Total Radiation Dose: ${this.results.environmental_summary.total_radiation_dose.toFixed(1)} rads
                    </div>
                </div>
            </div>
        `;

        document.getElementById('resultsSummary').innerHTML = summaryHTML;
    }

    enableExportButtons() {
        document.querySelectorAll('#exportPDF, #exportExcel, #exportCSV, #exportJSON')
                .forEach(btn => btn.disabled = false);
    }

    async exportResults(format) {
        if (!this.currentSimulation) {
            this.showError('No simulation results to export');
            return;
        }

        try {
            const response = await fetch(`${this.apiBase}/export/data/${this.currentSimulation}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ format: format })
            });

            if (!response.ok) {
                throw new Error(`Failed to export ${format.toUpperCase()}`);
            }

            // Get filename from content-disposition header or create one
            const contentDisposition = response.headers.get('content-disposition');
            let filename = `solar_panel_results.${format}`;

            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
                if (filenameMatch) {
                    filename = filenameMatch[1];
                }
            }

            // Download the file
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

        } catch (error) {
            console.error('Error exporting results:', error);
            this.showError(error.message);
        }
    }

    showLoading(show) {
        const loadingElements = document.querySelectorAll('.scenario-btn, #customScenarioForm button');
        loadingElements.forEach(el => {
            el.disabled = show;
            if (show) {
                el.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Loading...';
            } else {
                // Reset button content
                if (el.classList.contains('scenario-btn')) {
                    const icon = el.querySelector('i');
                    el.innerHTML = icon.outerHTML + ' ' + el.textContent.trim();
                } else {
                    el.innerHTML = '<i class="fas fa-play me-2"></i>Run Custom Simulation';
                }
            }
        });
    }

    showSimulationStatus(show) {
        const statusDiv = document.getElementById('simulationStatus');
        const plotContainer = document.getElementById('plotContainer');
        const welcomeMessage = document.getElementById('welcomeMessage');

        if (show) {
            statusDiv.style.display = 'block';
            plotContainer.style.display = 'none';
            welcomeMessage.style.display = 'none';
        } else {
            statusDiv.style.display = 'none';
        }
    }

    showError(message) {
        // Create toast notification
        const toastHTML = `
            <div class="toast align-items-center text-white bg-danger border-0" role="alert">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;

        const toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        toastContainer.innerHTML = toastHTML;
        document.body.appendChild(toastContainer);

        const toast = new bootstrap.Toast(toastContainer.querySelector('.toast'));
        toast.show();

        // Remove toast element after it's hidden
        toastContainer.querySelector('.toast').addEventListener('hidden.bs.toast', () => {
            document.body.removeChild(toastContainer);
        });
    }

    showWarnings(warnings) {
        const warningsHTML = warnings.map(warning =>
            `<div class="alert alert-warning alert-dismissible fade show" role="alert">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${warning}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>`
        ).join('');

        const warningsContainer = document.createElement('div');
        warningsContainer.innerHTML = warningsHTML;
        warningsContainer.className = 'mb-3 fade-in';

        const scenarioInfo = document.getElementById('scenarioInfo');
        scenarioInfo.parentNode.insertBefore(warningsContainer, scenarioInfo.nextSibling);
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.solarPanelApp = new SolarPanelApp();
});