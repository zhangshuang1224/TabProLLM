# TabProLLM: Probabilistic Prompting with LLM for Tabular Data Generation

This repository implements **TabProLLM**, a novel framework for generating high-fidelity and privacy-preserving tabular data using **Large Language Models (LLMs)** guided by **probabilistic prompts** derived from real data distributions.

## 📘 Project Overview

TabProLLM separates the generation of **numerical** and **categorical** columns and constructs LLM prompts based on:
- Gaussian Mixture Models (GMM) for numerical columns.
- Conditional probability distributions for categorical columns.
It ensures structural consistency and statistical fidelity in generated tables.

## 📁 Directory Structure

```
.
├── Data/                   # Raw tabular datasets (e.g., Adult, Iris)
├── Generated_Data/         # Output directory for LLM-generated data
├── Prompt/                 # Auto-generated natural language prompts
├── GMM/                    # GMM modeling and visualization for numerical features
├── KDE/                    # KDE plots for distribution comparison
├── HeatMap/                # Heatmap visualizations for correlation matrices
├── Comparison/             # Metric comparison results across models
├── explain.py              # Evaluation summarization and analysis
├── test.py ~ test7.py      # Functional test scripts
├── .git/, .idea/, venv/    # Version control, IDE configs, virtual env
```

## ⚙️ Function Modules

| Folder         | Description |
|----------------|-------------|
| `Data/`         | Stores source CSV datasets |
| `Prompt/`       | Contains generated prompts (numerical + categorical) |
| `GMM/`          | Fits GMMs to numerical columns |
| `KDE/`          | Generates KDE plots for fidelity evaluation |
| `HeatMap/`      | Correlation heatmaps (real vs generated) |
| `Comparison/`   | SDMetrics-based evaluation metrics |

## 🧪 Experiments

- **Datasets:** Adult, Iris, Wine, California Housing, etc.
- **Baselines:** CTGAN, TVAE, TabDDPM, GPT-4o
- **Metrics:**
  - **Fidelity:** RangeCoverage, KSComplement, etc.
  - **Correlation:** CorrelationSimilarity
  - **Privacy:** DCR, NNDR, NewRowSynthesis

## 🖥️ Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Key packages:
- `openai`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`


## 🔐 API Configuration

If using OpenAI API:

```bash
export OPENAI_API_KEY='your_key_here'
```


