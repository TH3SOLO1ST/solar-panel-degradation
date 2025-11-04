# User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Understanding the Interface](#understanding-the-interface)
3. [Running Simulations](#running-simulations)
4. [Interpreting Results](#interpreting-results)
5. [Creating Custom Scenarios](#creating-custom-scenarios)
6. [Exporting Results](#exporting-results)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### System Requirements

**Minimum Requirements:**
- Python 3.8 or higher
- 4 GB RAM
- 1 GB disk space
- Modern web browser (Chrome, Firefox, Safari, Edge)

**Recommended Requirements:**
- Python 3.9 or higher
- 8 GB RAM
- 2 GB disk space
- Latest version of your preferred browser

### Installation Options

#### Option 1: Quick Start (Recommended)

1. **Download the installer** for your operating system
2. **Run the installer** - follow the on-screen instructions
3. **Launch the application** from your desktop or start menu
4. **The web interface will open automatically** in your default browser

#### Option 2: Manual Installation

1. **Install Python** from [python.org](https://python.org)
2. **Download the source code** from GitHub
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Start the application**:
   ```bash
   python -m src.main server
   ```
5. **Open your browser** to `http://localhost:5000`

## Understanding the Interface

The interface is divided into three main panels:

### Left Panel - Scenario Selection

![Scenario Selection Panel](images/interface-left.png)

**Features:**
- **Quick Start Buttons**: One-click access to preset scenarios
- **Custom Scenario Creator**: Build your own satellite configurations
- **Current Scenario Info**: Details about your selected scenario

**Quick Start Options:**
1. **ISS Example** - International Space Station (408 km altitude)
2. **GEO Satellite** - Communications satellite (35,786 km altitude)
3. **Earth Observation** - Sun-synchronous orbit (785 km altitude)

### Center Panel - Visualization

![Visualization Panel](images/interface-center.png)

**Features:**
- **Real-time Progress**: Watch simulation progress
- **Interactive Charts**: Multiple visualization tabs
- **Plot Tabs**: Power, Efficiency, Temperature, and Sources

**Chart Types:**
- **Power Output**: Shows power generation over mission lifetime
- **Efficiency**: Tracks solar cell efficiency degradation
- **Temperature**: Displays thermal cycling profiles
- **Sources**: Breakdown of degradation causes

### Right Panel - Results and Export

![Results Panel](images/interface-right.png)

**Features:**
- **Results Summary**: Key performance metrics
- **Export Options**: Download results in various formats
- **Help System**: Built-in documentation and explanations

## Running Simulations

### Step-by-Step Guide

#### 1. Choose Your Scenario

**For Beginners:**
- Click one of the preset scenario buttons (ISS, GEO, or Earth Observation)
- The scenario details will appear in the left panel

**For Advanced Users:**
- Click "Create Custom" to build your own scenario
- Set parameters using the form controls

#### 2. Review Scenario Details

Check the scenario information panel:
- Mission duration
- Orbit type and altitude
- Solar panel specifications
- Expected degradation range

#### 3. Start the Simulation

- Click the "Run Simulation" button
- Watch the progress bar in the center panel
- Simulation typically takes 10-60 seconds

#### 4. Explore Results

Once complete:
- **Power Tab**: See how power output changes over time
- **Efficiency Tab**: View efficiency degradation trends
- **Temperature Tab**: Examine thermal cycling effects
- **Sources Tab**: Understand degradation causes

### Understanding Simulation Progress

The progress bar shows these phases:
1. **Initializing** - Setting up orbital parameters
2. **Propagating Orbit** - Calculating satellite position
3. **Calculating Eclipse** - Determining sun/shadow periods
4. **Thermal Analysis** - Computing temperature profiles
5. **Radiation Modeling** - Calculating radiation exposure
6. **Degradation Analysis** - Computing performance loss
7. **Generating Results** - Creating visualizations

## Interpreting Results

### Key Performance Metrics

#### Power Metrics
- **Initial Power**: Starting power output in watts
- **Final Power**: Ending power output in watts
- **Power Degradation**: Percentage of power lost
- **Average Power**: Mean power over mission

#### Efficiency Metrics
- **Initial Efficiency**: Starting conversion efficiency
- **Final Efficiency**: Ending conversion efficiency
- **Efficiency Degradation**: Percentage efficiency loss

#### Environmental Metrics
- **Temperature Range**: Minimum and maximum temperatures
- **Radiation Dose**: Total radiation exposure
- **Solar Exposure**: Percentage of time in sunlight

### Understanding Charts

#### Power Output Chart
- **X-axis**: Mission time in years
- **Y-axis**: Power output in kilowatts
- **Blue line**: Power output over time
- **Dashed line**: Initial power level
- **Annotation**: Final power and degradation

#### Efficiency Chart
- **X-axis**: Mission time in years
- **Y-axis**: Efficiency as percentage
- **Green line**: Efficiency over time
- **Shaded area**: Cumulative degradation
- **Diamond markers**: Degradation milestones (5%, 10%, 15%, 20%)

#### Temperature Chart
- **X-axis**: Mission time in days
- **Y-axis**: Temperature in Celsius
- **Orange line**: Temperature over time
- **Horizontal line**: Mean temperature
- **Shaded area**: ±1 standard deviation

#### Degradation Sources Chart
- **Pie chart**: Breakdown of degradation causes
- **Segments**: Radiation damage, thermal cycling, contamination
- **Percentages**: Relative contribution of each mechanism

### What the Numbers Mean

#### Good Performance Indicators
- **Power degradation < 15%**: Excellent design margin
- **Temperature range < 100°C**: Moderate thermal stress
- **Performance variability < 5%**: Stable power output
- **Solar exposure > 80%**: Good power generation time

#### Warning Signs
- **Power degradation > 30%**: May need design changes
- **Temperature range > 150°C**: High thermal stress
- **Performance variability > 10%**: Unstable operation
- **Efficiency drops suddenly**: Potential failure mode

## Creating Custom Scenarios

### Custom Scenario Parameters

#### Mission Duration
- **Range**: 1-20 years
- **Default**: 5 years
- **Impact**: Longer missions show more cumulative degradation

#### Orbit Altitude
- **Low Earth Orbit (LEO)**: 200-2000 km
  - Frequent eclipse periods
  - Moderate radiation exposure
  - Short orbital periods

- **Medium Earth Orbit (MEO)**: 2000-20000 km
  - Moderate eclipse frequency
  - Variable radiation exposure
  - Medium orbital periods

- **Geostationary Orbit (GEO)**: 35786 km
  - Long eclipse periods (during equinox)
  - High radiation exposure
  - 24-hour orbital period

#### Solar Panel Size
- **Small**: 10-30 m² (CubeSats, small satellites)
- **Medium**: 30-80 m² (Typical communications satellites)
- **Large**: 80-200 m² (High-power satellites)

#### Solar Cell Technology
- **Silicon Cells**:
  - Efficiency: 15-22%
  - Proven technology
  - Lower cost
  - Good radiation resistance

- **Multi-Junction Cells**:
  - Efficiency: 28-35%
  - Higher performance
  - Higher cost
  - Better radiation resistance

### Scenario Validation

The system will warn you about:
- **Very low altitudes** (< 200 km): Atmospheric drag
- **Very high altitudes** (> 50000 km): Practical limits
- **Long missions** (> 20 years): Uncertainty increases
- **Large panel areas** (> 200 m²): Structural concerns
- **High efficiencies** (> 35%): Technology limits

### Best Practices

#### For LEO Missions
- Use silicon cells for cost-effectiveness
- Account for frequent thermal cycling
- Consider radiation shielding for SAA passes

#### For GEO Missions
- Use multi-junction cells for maximum efficiency
- Plan for long eclipse periods during equinox
- Design for high radiation exposure

#### for Earth Observation
- Balance efficiency with cost
- Optimize for consistent sun illumination
- Consider specialized viewing requirements

## Exporting Results

### Export Formats

#### PDF Report
- **Content**: Complete analysis report
- **Sections**: Executive summary, performance metrics, environmental analysis
- **Use**: Presentations, documentation, stakeholder reports

#### Excel Data
- **Content**: Raw simulation data
- **Sheets**: Time series, performance metrics, degradation breakdown
- **Use**: Detailed analysis, custom calculations, data processing

#### CSV Data
- **Content**: Time series data only
- **Format**: Comma-separated values
- **Use**: Import into other tools, custom analysis

#### JSON Data
- **Content**: Complete results structure
- **Format**: JavaScript Object Notation
- **Use**: Web applications, API integration

### Export Process

1. **Complete your simulation**
2. **Choose export format** from the right panel
3. **Click the export button**
4. **File will download automatically**
5. **Check your downloads folder**

### File Naming

Exported files use this naming convention:
```
simulation_[ID]_[TIMESTAMP].[FORMAT]
```

Example: `simulation_a1b2c3d4_20231215_143022.pdf`

### PDF Report Contents

#### Executive Summary
- Mission overview
- Key findings
- Performance highlights

#### Technical Analysis
- Detailed performance metrics
- Environmental conditions
- Degradation mechanisms

#### Visualizations
- Power output trends
- Efficiency degradation
- Temperature profiles
- Degradation sources

#### Recommendations
- Design implications
- Operational considerations
- Future improvements

## Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: "Python not found" error
**Solution**:
1. Install Python 3.8+ from python.org
2. Add Python to system PATH
3. Restart command prompt/terminal

**Issue**: "pip command not found"
**Solution**:
1. Ensure Python was installed with pip
2. Try `python -m pip` instead of `pip`
3. Reinstall Python if necessary

#### Simulation Issues

**Issue**: Simulation fails to start
**Solution**:
1. Check server is running (http://localhost:5000)
2. Refresh browser page
3. Check browser console for errors
4. Restart the server

**Issue**: Simulation takes too long
**Solution**:
1. Be patient - complex missions take longer
2. Check progress bar for current phase
3. Reduce mission duration for testing
4. Close other applications to free memory

**Issue**: Results don't look reasonable
**Solution**:
1. Check scenario parameters
2. Compare with preset scenarios
3. Review warnings for potential issues
4. Try different orbit parameters

#### Display Issues

**Issue**: Charts not displaying
**Solution**:
1. Check browser compatibility
2. Enable JavaScript
3. Clear browser cache
4. Try different browser

**Issue**: Export buttons not working
**Solution**:
1. Wait for simulation to complete
2. Check browser download settings
3. Try different export format
4. Check file permissions

### Performance Optimization

#### For Faster Simulations
- Use shorter mission durations for testing
- Choose simpler orbits (LEO is fastest)
- Close unnecessary browser tabs
- Ensure adequate system memory

#### For Better Results
- Use realistic mission parameters
- Compare multiple scenarios
- Review all chart types
- Export data for detailed analysis

### Getting Help

#### Self-Service Resources
- **Built-in Help**: Click help buttons in the interface
- **User Guide**: This comprehensive document
- **FAQ**: Common questions and answers

#### Community Support
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Documentation**: Technical details and API reference

#### Contact Information
- **Email**: contact@solarpanelmodel.com
- **Website**: https://solarpaneldegradation.readthedocs.io/
- **Support Forum**: https://github.com/ezra-compyle/solar-panel-degradation/discussions

### Tips and Tricks

#### For Beginners
1. **Start with preset scenarios** to understand the tool
2. **Read all help text** to learn terminology
3. **Export results** to examine data offline
4. **Compare scenarios** to see differences

#### For Advanced Users
1. **Use custom scenarios** for specific missions
2. **Export raw data** for custom analysis
3. **Validate against real data** when available
4. **Contribute improvements** to the project

#### For Educators
1. **Use preset scenarios** for demonstrations
2. **Explain each degradation mechanism**
3. **Compare different orbits** as teaching examples
4. **Export PDFs** for student handouts

---

For additional help or questions, please refer to our [comprehensive documentation](https://solarpaneldegradation.readthedocs.io/) or [contact our support team](mailto:contact@solarpanelmodel.com).