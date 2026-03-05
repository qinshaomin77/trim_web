# RVST Web UI

**A Toolkit for Regenerating Vehicle Speed Trajectories to Connect Traffic Microsimulation with Vehicle Emission Models**

RVST Web UI provides a modern web-based interface for the RVST pipeline, enabling seamless processing and analysis of traffic microsimulation data from SUMO with vehicle emission models.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Pipeline Steps](#pipeline-steps)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [FAQ](#faq)

---

## 🎯 Overview

RVST (Regenerating Vehicle Speed Trajectories) bridges the gap between traffic microsimulation and vehicle emission modeling. This web interface automates the complete workflow:

**Input**: SUMO simulation outputs (network + FCD data)  
**Process**: 11-step pipeline including trajectory regeneration, emission calculations, and spatial analysis  
**Output**: Emission maps, comparison results, and detailed analytics

### Why RVST Web UI?

- 🌐 **Browser-Based** - No desktop installation required
- 🚀 **One-Click Launch** - Automatic dependency management
- 📊 **Real-Time Monitoring** - Live progress updates via SSE
- ⏸️ **Pause & Resume** - Flexible workflow control
- 🔍 **Integrated File Browser** - Easy file selection
- 🎨 **Modern Interface** - Built with React

---

## ✨ Features

### Core Capabilities

- **Complete RVST Pipeline** - All 11 processing steps automated
- **Multi-Pollutant Support** - NOx, PM2.5, CO, CO2, THC
- **Time Filtering** - Analyze specific simulation periods
- **Spatial Analysis** - Grid-based emission mapping with ROI support
- **Ground Truth Integration** - Compare simulations with real-world measurements
- **Savitzky-Golay Smoothing** - Optional data smoothing

### Technical Highlights

- **Async Processing** - Non-blocking pipeline execution
- **SSE Streaming** - Real-time progress events
- **Smart Validation** - Comprehensive input checking
- **Auto-Dependency** - Automatic Python package installation
- **Cross-Platform** - Windows, macOS, Linux support

---

## 🚀 Quick Start

### 30-Second Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/RVST_GUI.git
cd RVST_GUI

# Launch (installs dependencies automatically)
python run.py
```

Your browser will open automatically to `http://localhost:8000`

### First Run

1. **Select Input Files**
   - SUMO net.xml (network topology)
   - FCD XML (vehicle trajectories)
   - Output directory (for results)

2. **Configure Options** (optional)
   - Choose pollutants to analyze
   - Set time filters
   - Adjust spatial settings

3. **Run Pipeline**
   - Click **▶ Run All**
   - Monitor progress in real-time
   - View results in output directory

That's it! 🎉

---

## 📦 Installation

### System Requirements

**Minimum**:
- Python 3.8+
- 4 GB RAM
- Modern browser (Chrome/Firefox/Edge/Safari)

**Recommended**:
- Python 3.10+
- 8 GB+ RAM
- Multi-core CPU

### Automatic Installation (Recommended)

```bash
python run.py
```

The launcher automatically:
- ✅ Checks Python version
- ✅ Installs required packages (fastapi, uvicorn, pandas, etc.)
- ✅ Detects available port
- ✅ Starts web server
- ✅ Opens browser

### Manual Installation

```bash
# Install dependencies
pip install fastapi uvicorn sse-starlette pandas numpy lxml matplotlib scipy

# Start server
python app/web_server.py

# Open browser
open http://localhost:8000
```

---

## 📖 Usage Guide

### Input Files

#### Required Files

**1. SUMO net.xml**
- Network definition file from SUMO
- Contains road topology and geometry
- Example: `network.net.xml`

**2. FCD XML**
- Floating Car Data from SUMO simulation
- Contains vehicle positions and speeds over time
- Example: `fcd-output.xml`

**3. Output Directory**
- Folder to save results
- Must have write permissions
- Example: `C:\Projects\RVST\results`

#### Optional Files

**4. Ground Truth CSV**
- Real-world emission measurements
- Required format: columns including `time`, `pollutant`, `value`
- Example: `uav_measurements.csv`

### Configuration Options

#### Pollutant Selection

Select pollutants to analyze:
- **NOx** - Nitrogen Oxides
- **PM2.5** - Fine Particulate Matter
- **CO** - Carbon Monoxide
- **CO2** - Carbon Dioxide
- **THC** - Total Hydrocarbons

*Note: Select at least one pollutant*

#### Time Filters

**SUMO Time Filter**
- Filter FCD data by simulation time
- Start/End time in seconds
- Example: `500` to `3600` (analyze 8:20 - 60:00)

**Ground Truth Time Filter**
- Filter real-world data by time
- Same format as SUMO filter

#### Spatial Settings

**Grid Size**
- Cell size in meters: `10` - `100`
- Smaller = more detailed, slower
- Recommended: `20` meters

**Analysis Axis**
- `both` - Analyze X and Y axes
- `x` - X-axis only
- `y` - Y-axis only

**Region of Interest (ROI)**
- Enable: Limit analysis to specific area
- Format: `min_x,min_y,max_x,max_y`
- Mode: `shared` or `separate` for simulation/ground truth

#### Other Options

**Enable Plots**
- Generate visualization charts
- Increases processing time
- Recommended for result verification

**Use Savitzky-Golay Smoothing**
- Apply data smoothing
- Reduces noise in emission maps

---

## 🔄 Pipeline Steps

The RVST pipeline consists of 11 sequential steps:

### 1. Network Topology 🗺️

**Purpose**: Extract road network structure from SUMO net.xml

**Outputs**:
- `net_topology.csv` - Edge information (ID, length, coordinates)
- Network graph representation

**Duration**: ~5-30 seconds

---

### 2. FCD XML to CSV 📊

**Purpose**: Convert SUMO FCD XML to structured CSV format

**Processing**:
- Parse XML vehicle trajectories
- Apply time filters (if enabled)
- Extract position, speed, acceleration

**Outputs**:
- `fcd.csv` - Vehicle trajectories (timestep, vehicle_id, x, y, speed, etc.)

**Duration**: ~30-180 seconds (file size dependent)

---

### 3. Identify Neighbors 🔍

**Purpose**: Determine neighboring vehicles for each vehicle at each timestep

**Algorithm**:
- Spatial proximity analysis
- Same-lane detection
- Neighboring-lane identification

**Outputs**:
- `neighbors.csv` - Vehicle neighbor relationships

**Duration**: ~10-60 seconds

---

### 4. Trip Split ✂️

**Purpose**: Segment continuous vehicle trajectories into individual trips

**Outputs**:
- `trips/` directory with individual trip files
- Trip metadata

**Duration**: ~5-30 seconds

---

### 5. RVST Post-Processing ⚙️

**Purpose**: Regenerate vehicle speed trajectories with enhanced resolution

**Processing**:
- Trajectory smoothing
- Speed profile regeneration
- Data quality enhancement

**Outputs**:
- Enhanced trajectory data
- Quality metrics

**Duration**: ~30-120 seconds

---

### 6. Fill XY Coordinates 📍

**Purpose**: Add spatial coordinates to all data points

**Processing**:
- Map edge positions to coordinates
- Interpolate missing positions
- Align with network geometry

**Outputs**:
- Data with complete spatial information

**Duration**: ~10-45 seconds

---

### 7. Emission Factor Matching 🔗

**Purpose**: Match vehicles with appropriate emission factors

**Processing**:
- Vehicle type classification
- Emission factor database lookup
- Apply pollutant-specific factors

**Outputs**:
- Emission calculations per vehicle
- Factor matching log

**Duration**: ~20-90 seconds

---

### 8. Spatial Emission Mapping 🗾

**Purpose**: Generate grid-based spatial emission distribution

**Processing**:
- Create spatial grid based on settings
- Aggregate emissions per grid cell
- Generate heat maps

**Outputs**:
- `emission_spatial_map/` - Grid emission data
- Visualization plots (if enabled)

**Duration**: ~30-180 seconds

---

### 9. Emission Comparison 📈

**Purpose**: Compare simulation results with ground truth measurements

**Processing**:
- Align simulation and measurement data
- Calculate statistical metrics (R², RMSE, MAE)
- Generate comparison plots

**Outputs**:
- `comparison_results.csv` - Statistical analysis
- Comparison visualizations

**Duration**: ~15-60 seconds

*Note: Only runs when ground truth data is provided*

---

### 10. Savitzky-Golay Smoothing 🎨

**Purpose**: Apply smoothing filter to emission maps

**Processing**:
- Polynomial smoothing
- Noise reduction
- Preserve emission patterns

**Outputs**:
- Smoothed emission maps
- Quality improvement metrics

**Duration**: ~10-30 seconds

*Note: Only runs when "Use SG Smoothing" is enabled*

---

### 11. Ground Truth Standardization 📐

**Purpose**: Standardize ground truth data format

**Processing**:
- Format validation
- Unit conversion
- Data alignment

**Outputs**:
- `gt_standardized.csv` - Standardized format

**Duration**: ~5-20 seconds

*Note: Only runs when ground truth data is provided*

---

## 🏗️ Architecture

### Project Structure

```
RVST_GUI/
├── run.py                    # Launch script
├── rvst.html                 # React frontend
├── app/
│   ├── web_server.py        # FastAPI backend
│   ├── rvst_gui.py          # Pipeline engine
│   ├── bindings.py          # Step bindings
│   └── adapters.py          # Script adapters
├── scripts/                  # Processing scripts
│   ├── net_topology.py
│   ├── xml2csv_fcd.py
│   ├── identify_neighbors.py
│   ├── trip_split.py
│   ├── rvst_postprocess.py
│   ├── fill_xy.py
│   ├── ef_match.py
│   ├── emission_spatial_map.py
│   ├── emission_compare.py
│   ├── sg_smooth.py
│   └── gt_standardize.py
└── docs/                     # Documentation
```

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Frontend Layer (rvst.html)                              │
│  - React UI components                                   │
│  - Real-time progress display                            │
│  - Input validation                                      │
│  - SSE event handling                                    │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│  Backend Layer (web_server.py)                           │
│  - FastAPI HTTP endpoints                                │
│  - Request validation                                    │
│  - SSE event streaming                                   │
│  - File system operations                                │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│  Pipeline Layer (rvst_gui.py + scripts/)                 │
│  - Step orchestration                                    │
│  - Async execution                                       │
│  - Progress event emission                               │
│  - Data processing                                       │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Input (Browser)
    ↓
Frontend validates and sends to Backend
    ↓
Backend validates and builds Pipeline
    ↓
Pipeline executes steps sequentially
    ↓  (sends progress events via SSE)
Frontend displays real-time progress
    ↓
Results saved to Output Directory
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Custom port
export RVST_PORT=8080
python run.py

# Windows
set RVST_PORT=8080
python run.py
```

### Performance Tuning

**For Large Datasets**:

1. **Increase Grid Size**
   - Use 50m instead of 10m cells
   - Reduces processing time by ~70%

2. **Use Time Filters**
   - Analyze specific periods only
   - Reduces data volume

3. **Disable Plots**
   - Skip visualization generation
   - Saves ~30% processing time

4. **Run Selected Steps**
   - Execute only needed steps
   - Faster iteration during development

**Memory Optimization**:

```bash
# Ensure 64-bit Python
python -c "import sys; print(sys.maxsize > 2**32)"  # Should print True

# Monitor memory usage
# Windows: Task Manager
# Linux: htop
# macOS: Activity Monitor
```

---

## 🐛 Troubleshooting

### Common Issues

#### Server Won't Start

**Symptom**: `Server process exited immediately with code 1`

**Solutions**:

1. **Check Python version**
   ```bash
   python --version  # Must be 3.8+
   ```

2. **Install dependencies**
   ```bash
   pip install fastapi uvicorn sse-starlette pandas numpy lxml
   ```

3. **Check port availability**
   ```bash
   # Windows
   netstat -an | findstr :8000
   
   # Linux/macOS
   lsof -i :8000
   ```

#### Connection Refused

**Symptom**: Browser shows `ERR_CONNECTION_REFUSED`

**Solutions**:

1. **Verify server is running**
   - Check terminal for "Uvicorn running on..."

2. **Use correct URL**
   - `http://localhost:8000` (not https)

3. **Check firewall**
   - Allow Python through firewall

#### Validation Errors

**Symptom**: `=== Pipeline error (validation failed) ===`

**Common Causes**:

1. **File not found**
   - Use Browse buttons
   - Verify file paths
   - Check file permissions

2. **Invalid time range**
   - Ensure Start Time < End Time

3. **No pollutants selected**
   - Select at least one pollutant

#### Pipeline Execution Errors

**Symptom**: Step fails with error

**Debugging**:

1. **Check terminal output**
   ```
   [ERROR] Pipeline execution error:
   ============================================================
   [Detailed error traceback]
   ============================================================
   ```

2. **Common issues**:
   - Missing Python packages: `pip install <package>`
   - Corrupted XML files: Validate with SUMO tools
   - Insufficient memory: Close other applications

3. **Check browser console**
   - Press F12
   - Look for errors in Console tab

### Debug Mode

Enable detailed logging:

```python
# In web_server.py, the debug version automatically provides:
[DEBUG] Validation complete. Errors: 0
[STEP] Starting: net_topology - Building network topology
[STEP] ✓ Completed: net_topology - Done in 12.34s
[INFO] Pipeline status: RUNNING | Steps: 2/11 | Events: 8
```

### Getting Help

If issues persist:

1. **Collect information**:
   - Python version: `python --version`
   - Installed packages: `pip list`
   - Full terminal output
   - Browser console errors (F12)

2. **Check documentation**:
   - `docs/STARTUP_TROUBLESHOOTING.md`
   - `docs/VALIDATION_ERROR_GUIDE.md`

3. **Report issue** with:
   - Operating system
   - Python version
   - Error messages
   - Steps to reproduce

---

## 🔌 API Reference

### Endpoints

#### `GET /`
Returns the main web interface

#### `POST /api/prepare`
Submit pipeline parameters

**Request**:
```json
{
  "net_xml": "/path/to/net.xml",
  "fcd_input": "/path/to/fcd.xml",
  "out_dir": "/path/to/output",
  "prefix": "fcd",
  "pollutants": ["NOx", "PM2.5"],
  "enable_sumo_time_filter": true,
  "sumo_time_start": 500,
  "sumo_time_end": 3600
}
```

**Response**:
```json
{
  "token": "839c534c-4071-48eb-989e-4b3c118ba31f"
}
```

#### `GET /api/run?token={token}`
Start pipeline execution (SSE stream)

**Events**:
```javascript
// Step started
{type: "step_start", step_id: "net_topology", message: "..."}

// Step completed
{type: "step_done", step_id: "net_topology", message: "Done in 12.34s"}

// Pipeline completed
{type: "pipeline_done", message: "=== Pipeline done ==="}
```

#### `POST /api/resume`
Resume paused pipeline

#### `GET /api/status`
Get current pipeline status

**Response**:
```json
{
  "is_running": false,
  "app_status": "Done",
  "can_resume": false,
  "step_status": {"net_topology": "done", ...}
}
```

#### `GET /api/browse/{os_type}?path={path}`
Browse file system

**Parameters**:
- `os_type`: "windows", "mac", or "linux"
- `path`: Directory path to browse

#### `GET /api/health`
Health check endpoint

**Response**:
```json
{
  "status": "ok",
  "version": "2.0"
}
```

---

## ❓ FAQ

### General

**Q: What is RVST?**

A: RVST (Regenerating Vehicle Speed Trajectories) is a toolkit that connects traffic microsimulation outputs (from SUMO) with vehicle emission models. It regenerates high-resolution speed trajectories and calculates spatial emission distributions.

**Q: Do I need SUMO installed?**

A: No. RVST processes SUMO output files (net.xml, fcd.xml) but doesn't require SUMO installation. Generate these files with SUMO separately.

**Q: What file formats are supported?**

A: 
- **Input**: SUMO XML (net.xml, fcd.xml), CSV (ground truth)
- **Output**: CSV, JSON, PNG (plots)

**Q: Can I process multiple simulations?**

A: Yes. Run the pipeline multiple times with different input files. Only one pipeline can run at a time per server instance.

### Technical

**Q: Why does processing take so long?**

A: Processing time depends on:
- FCD file size (number of vehicles × timesteps)
- Spatial grid resolution
- Number of pollutants
- Plot generation (if enabled)

Typical range: 5-30 minutes for medium datasets

**Q: How much memory is needed?**

A: Depends on dataset size:
- Small (<100 MB FCD): 2-4 GB RAM
- Medium (100-500 MB): 4-8 GB RAM
- Large (>500 MB): 8+ GB RAM

Use time filters to reduce memory usage.

**Q: Can I customize the pipeline?**

A: Yes. Add custom steps by:
1. Creating script in `scripts/`
2. Adding binding in `app/bindings.py`
3. Updating pipeline definition

**Q: Can I run this on a server?**

A: Yes. Access via SSH tunnel or configure for external connections:
```bash
# Allow external access (security risk - use firewall)
# Modify web_server.py: host="0.0.0.0"
```

### Results

**Q: Where are results saved?**

A: In the output directory you specified. Structure:
```
output/
├── net_topology/
├── fcd.csv
├── trips/
├── emission_spatial_map/
├── comparison_results.csv
└── manifest.jsonl  # Execution log
```

**Q: How do I interpret results?**

A: 
- `manifest.jsonl`: Step-by-step execution log
- `emission_spatial_map/`: Grid cell emission values
- `comparison_results.csv`: Simulation vs ground truth statistics
- Plots: Visual validation of results

**Q: Can I export results to GIS?**

A: Yes. CSV outputs include X/Y coordinates compatible with:
- QGIS (import as delimited text)
- ArcGIS (add XY data)
- Python (GeoPandas)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **SUMO Team** - Traffic simulation platform
- **FastAPI** - Modern web framework
- **React Team** - UI library
- **RVST Contributors** - Original toolkit developers

---

## 📞 Contact

- **GitHub**: [https://github.com/your-repo/RVST_GUI](https://github.com/your-repo/RVST_GUI)
- **Issues**: [https://github.com/your-repo/RVST_GUI/issues](https://github.com/your-repo/RVST_GUI/issues)
- **Documentation**: [https://rvst-gui.readthedocs.io](https://rvst-gui.readthedocs.io)

---

## 📊 Citation

If you use RVST in your research, please cite:

```bibtex
@article{rvst2024,
  title={RVST: A Toolkit for Regenerating Vehicle Speed Trajectories to Connect Traffic Microsimulation with Vehicle Emission Models},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

---

**Built with ❤️ for the transportation research community**

*Connecting traffic microsimulation with emission modeling - one trajectory at a time.*
