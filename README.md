# Property-Guided CO₂ Carbanion Generation

A reproducible workflow for generating high-activity carbanion molecules for CO₂ activation using property-guided molecular generation.

## Overview

This codebase implements a complete pipeline:
1. **Chemprop Training**: Train a property predictor for Mayr nucleophilicity parameters (N, sN)
2. **Property-Guided Finetuning**: Finetune a generative model with predicted log k(CO₂) + carbanion constraints
3. **Postprocessing & Classification**: Deduplicate, score, and classify generated molecules

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/prop_guided_co2_carbanion.git
cd prop_guided_co2_carbanion
```

### 2. Set up the conda environment

```bash
# Create environment from YAML file
conda env create -f environment.yaml

# Activate environment
conda activate pgco2
```

### 3. Install HGraph2Graph

The generative model requires the HGraph2Graph package:

```bash
# Clone HGraph2Graph into this repository
git clone https://github.com/wengong-jin/hgraph2graph.git

```

## Quick Start

### Using the Jupyter Notebook (Recommended)

```bash
# 1. Activate environment
conda activate pgco2

# 2. Navigate to repository
cd prop_guided_co2_carbanion

# 3. Start Jupyter
jupyter notebook workflow.ipynb
```

The notebook provides an interactive, step-by-step workflow with simple setting:
- Data exploration and visualization
- Automated execution of all three steps
- Real-time progress monitoring
- Result analysis and visualization
- Export of top 50 candidates ranked by combined score and log k(CO₂)

### Using Shell Scripts (Alternative)

Command-line execution for parameters tuning:

```bash
# Step 1: Train Chemprop model
bash 01_train_cp.sh

# Step 2: Run finetuning
bash 02_run_finetune.sh

# Step 3: Analyze results
bash 03_run_analysis.sh
```


## Citation

<!-- TODO: Add publication citation here -->

## Acknowledgements

This repository is built upon the previous work [hgraph2graph](https://github.com/wengong-jin/hgraph2graph). Thanks to the authors for their great work!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue in the repository.

---
