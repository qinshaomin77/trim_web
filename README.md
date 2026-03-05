<div align="center">

# TRIM: A Simulation-Independent Toolkit for<br>Vehicle Speed Trajectory Refinement in Emission Modeling

[Shaomin Qin](https://github.com/qinshaomin77) · [Haobing Liu](https://scholar.google.com/citations?user=e-8R2vMAAAAJ&hl=en) · [Lishengsa Yue](https://ieeexplore.ieee.org/author/37088478561)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![SUMO](https://img.shields.io/badge/SUMO-required-green.svg)](https://sumo.dlr.de/)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Prerequisites](#prerequisites)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Features](#features)
- [Citation](#citation)
- [License](#license)
- [Troubleshooting](#troubleshooting)
- [Contact](#contact)

---

## 🎯 Overview

TRIM is a simulation-independent toolkit designed for vehicle speed trajectory refinement in emission modeling. It provides tools for optimizing vehicle trajectories to reduce emissions while maintaining realistic driving patterns.

**Key Features:**
- 🚗 Simulation-independent trajectory optimization
- 📊 Advanced emission modeling and comparison
- 🗺️ Spatial emission mapping
- 🖥️ Interactive GUI and web-based visualization
- 🔧 Multiple optimization algorithms
- 🔗 Seamless SUMO integration

---

## 🎬 Demo

Watch the end-to-end TRIM workflow in action — from loading raw SUMO trajectories to physically refined speed profiles and emission estimation:

[![TRIM Demo](https://img.shields.io/badge/YouTube-Watch%20Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=KDPiDGzVudo)

> **[▶ TRIM: End-to-End Trajectory Refinement Demo](https://www.youtube.com/watch?v=KDPiDGzVudo)**  
> Raw SUMO FCD output → Two-stage MIQP optimization → Emission estimation & spatial mapping

---

## ⚙️ Prerequisites

- **Operating System**: Windows 10/11 (x64)
- **Python**: >= 3.9
- **SUMO**: Simulation of Urban MObility
- **Gurobi**: Optimization solver

---

## 📦 Dependencies

This project requires the following external tools in addition to Python packages.

### 1) SUMO (Eclipse SUMO)

Please install SUMO and make sure `sumo` / `sumo-gui` are available in your system PATH.

- **Official installation guide**: https://sumo.dlr.de/docs/Downloads.php
- **Documentation**: https://sumo.dlr.de/docs/

**Verification:**
```bash
sumo --version
```

---

### 2) Gurobi Optimizer (License required)

Please install Gurobi and activate a license (academic or commercial).

- **Download & installation**: https://www.gurobi.com/downloads/
- **License request / activation**: https://www.gurobi.com/downloads/end-user-license-agreement/
- **Academic license info**: https://www.gurobi.com/academia/academic-program-and-licenses/

**Verification:**
```bash
gurobi_cl --version
```

---

### 3) Python Packages (requirements.txt)

Install the required Python dependencies via pip:

```bash
pip install -r requirements.txt
```

---

## 🚀 Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/qinshaomin77/trim_web.git
cd trim_web
```

### Step 2: Restore large files

⚠️ **Important**: Due to GitHub file size limitations, `example/sumo_mcg/fcd.xml` has been split into multiple parts.

Run the merge script to restore the original file:

```bash
cd example/sumo_mcg
python merge_fcd_xml.py
cd ../..
```

This will automatically restore `fcd.xml` (101.35 MB).

### Step 3: Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify installations

```bash
# Check SUMO
sumo --version

# Check Python
python --version

# Check Gurobi (if installed)
gurobi_cl --version
```

---

## 🎮 Quick Start

### Run the GUI application

```bash
python app/trim_gui.py
```

### Run web interface

```bash
python run.py
```

Then open your browser and navigate to `http://localhost:8000`

### Command-line optimization

```bash
python scripts/trim_optimize.py \
    --input example/sumo_mcg/fcd.xml \
    --output results/ \
    --method greedy
```

---

## 📁 Project Structure

```
trim_web/
├── app/                           # Main application
│   ├── __init__.py
│   ├── adapters.py               # Data adapters
│   ├── bindings.py               # Python bindings
│   ├── trim_gui.py               # GUI interface
│   └── web_server.py             # Web server
│
├── data/                          # Data files
│   └── accel_envelope.csv        # Acceleration envelope data
│
├── example/                       # Example datasets
│   └── sumo_mcg/
│       ├── fcd.xml.part000       # Split large file (part 1, 50 MB)
│       ├── fcd.xml.part001       # Split large file (part 2, 50 MB)
│       ├── fcd.xml.part002       # Split large file (part 3, 1.35 MB)
│       └── merge_fcd_xml.py      # Merge script
│
├── scripts/                       # Utility scripts
│   ├── emission_compare.py       # Emission comparison
│   ├── emission_spatial_map.py   # Spatial emission mapping
│   ├── fill_xy.py                # Coordinate filling
│   ├── gt_standardize.py         # Ground truth standardization
│   ├── identify_neighbors.py     # Neighbor identification
│   ├── net_topology.py           # Network topology analysis
│   ├── sg_smooth.py              # Savitzky-Golay smoothing
│   ├── trim_optimize.py          # Main optimization script
│   ├── trim_postprocess.py       # Post-processing
│   ├── trip_split.py             # Trip splitting
│   ├── xml2csv.py                # XML to CSV conversion
│   ├── xml2csv_fcd.py            # FCD XML to CSV
│   └── xsd.py                    # XSD validation
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # License information
```

---

## 💡 Usage

### Basic Trajectory Optimization

```bash
python scripts/trim_optimize.py \
    --input example/sumo_mcg/fcd.xml \
    --output results/optimized.xml \
    --method greedy
```

### Generate Emission Comparison

```bash
python scripts/emission_compare.py \
    --before example/sumo_mcg/fcd.xml \
    --after results/optimized.xml \
    --output results/comparison.csv
```

### Create Spatial Emission Maps

```bash
python scripts/emission_spatial_map.py \
    --input results/optimized.xml \
    --output maps/ \
    --resolution 100
```

### Convert XML to CSV

```bash
python scripts/xml2csv_fcd.py \
    --input example/sumo_mcg/fcd.xml \
    --output data/fcd_data.csv
```

---

## ✨ Features

- ✅ **Simulation-independent**: Works with various traffic simulation outputs
- ✅ **Multiple algorithms**: Greedy, dynamic programming, and optimization-based approaches
- ✅ **Emission modeling**: Comprehensive emission calculation and analysis
- ✅ **Visualization**: Interactive GUI and web-based tools
- ✅ **SUMO integration**: Seamless integration with SUMO traffic simulator
- ✅ **Batch processing**: Handle multiple trajectories efficiently
- ✅ **Extensible**: Easy to add custom optimization algorithms

---

## 📚 Documentation

For detailed documentation, please refer to:

- **User Guide**: [docs/user_guide.md](docs/user_guide.md) *(coming soon)*
- **API Reference**: [docs/api_reference.md](docs/api_reference.md) *(coming soon)*
- **Tutorial**: [docs/tutorial.md](docs/tutorial.md) *(coming soon)*
- **Examples**: [example/](example/)

---

## 📖 Citation

If you use TRIM in your research, please cite:

```bibtex
@article{qin2026trim,
  title  = {TRIM: A Simulation-Independent Toolkit for Vehicle Speed Trajectory Refinement in Emission Modeling},
  author = {Shaomin Qin and Haobing Liu and Lishengsa Yue},
  year   = {2026},
}
```

---

## 📄 License

Please see the [LICENSE](LICENSE) file for further details.

---

## 🔧 Troubleshooting

### Issue 1: SUMO not found in PATH

**Error Message:**
```
'sumo' is not recognized as an internal or external command
```

**Solution:**

Add SUMO to your system PATH:

**Windows:**
```cmd
set PATH=%PATH%;C:\Program Files (x86)\Eclipse\Sumo\bin
```

Or add it permanently through System Environment Variables.

**Linux/Mac:**
```bash
export PATH=$PATH:/usr/share/sumo/bin
```

---

### Issue 2: Gurobi license error

**Error Message:**
```
GurobiError: No Gurobi license found
```

**Solution:**

1. Request a license from Gurobi (academic or commercial)
2. Download the license file (`gurobi.lic`)
3. Place it in your home directory or set `GRB_LICENSE_FILE` environment variable

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

---

### Issue 3: Large file (fcd.xml) missing

**Error Message:**
```
FileNotFoundError: example/sumo_mcg/fcd.xml not found
```

**Solution:**

Run the merge script to restore the file:

```bash
cd example/sumo_mcg
python merge_fcd_xml.py
```

Expected output:
```
Found 3 chunk files

Merging: fcd.xml.part000
Merging: fcd.xml.part001
Merging: fcd.xml.part002

✓ File restored: fcd.xml (101.35 MB)
```

---

### Issue 4: Python package import errors

**Solution:**

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt --upgrade
```

---

### Issue 5: Permission denied when running scripts

**Windows:**
```bash
python -m scripts.trim_optimize
```

**Linux/Mac:**
```bash
chmod +x scripts/*.py
python scripts/trim_optimize.py
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

- **Shaomin Qin**: [qinshaomin@tongji.edu.cn](mailto:qinshaomin@tongji.edu.cn)
- **Haobing Liu**: [liuhaobing@tongji.edu.cn](mailto:liuhaobing@tongji.edu.cn)
- **Lishengsa Yue**: [2014yuelishengsa@tongji.edu.cn](mailto:2014yuelishengsa@tongji.edu.cn)
- **Project Repository**: [https://github.com/qinshaomin77/trim_web](https://github.com/qinshaomin77/trim_web)
- **Issues**: [https://github.com/qinshaomin77/trim_web/issues](https://github.com/qinshaomin77/trim_web/issues)

---

## 🙏 Acknowledgments

This research is partially supported by:
- National Key R&D Program of China (No. `2023YFB3906900`)
- National Natural Science Foundation of China (No. `52572378`)
- **SUMO Development Team** for the excellent traffic simulation platform
- **Gurobi Optimization** for providing academic licenses

---

## 📊 Project Status

- ✅ Core optimization algorithms implemented
- ✅ GUI and web interface functional
- ✅ SUMO integration complete
- 🚧 Documentation in progress
- 🚧 Additional examples being added
- 📅 Next release: v1.1.0 (planned)

---

## 📝 Changelog

### Version 1.0.0 (2026)
- Initial release
- Core trajectory optimization algorithms
- GUI and web interface
- SUMO integration
- Example datasets

---

<div align="center">

**Made with ❤️ by the VISTA Team**

⭐ **Star this repository if you find it helpful!**

© 2026 Shaomin Qin · Haobing Liu · Lishengsa Yue · All rights reserved.

</div>