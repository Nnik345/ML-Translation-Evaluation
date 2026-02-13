# MT Evaluation Pipeline

## 📁 Project Structure

```
.
├── config.py              # Configuration settings
├── models.py              # Model initialization and loading
├── metrics.py             # Metric computation functions
├── data_loader.py         # Data loading and preprocessing
├── evaluator.py           # Main evaluation logic
├── utils.py               # Utility functions
├── Evaluation_Modular.ipynb  # Jupyter notebook interface
└── README.md              # This file
```

## 🔧 Module Descriptions

### `config.py`
- Centralized configuration for all parameters
- Model names, file paths, batch sizes
- Easy to modify settings without changing code

### `models.py`
- Functions to download and load COMET models
- NLTK data initialization
- Model management utilities

### `metrics.py`
- Individual metric computation functions:
  - `compute_bleu()` - BLEU score
  - `compute_meteor()` - METEOR score
  - `compute_comet()` - COMET score
  - `compute_comet_qe()` - COMET-QE score
  - `compute_all_metrics()` - Compute all at once

### `data_loader.py`
- `MTDataset` class for handling CSV datasets
- Data validation and preprocessing
- Handles missing values automatically
- Load single or multiple datasets

### `evaluator.py`
- `MTEvaluator` class orchestrates evaluation
- Evaluate single or multiple MT systems
- Automatic result collection and formatting
- Save results to CSV

### `utils.py`
- Environment setup
- GPU detection and info
- Memory cleanup utilities

## 🚀 Usage

- Run the Evaluation.ipynb file

## ⚙️ Configuration

Modify `config.py` to customize:

```python
# Change batch size
COMET_BATCH_SIZE = 16

# Change output directories
RESULTS_DIR = Path("custom_results")

# Add new languages
LANGUAGE_FILES['Kannada'] = DATA_DIR / 'Kannada.csv'

# Adjust decimal places
OUTPUT_DECIMAL_PLACES['BLEU'] = 3
```

## 📝 CSV Format Requirements

Input CSV files should have:
- `Source` column - source text
- `Reference` column - reference translation
- Additional columns - each representing an MT system output

Example:
```csv
Source,Reference,ChatGPT 5.2,BhashaVerse
"Hello world","नमस्ते दुनिया","नमस्ते विश्व","हैलो वर्ल्ड"
```

## 📦 Dependencies

- pandas
- numpy
- sacrebleu
- nltk
- torch
- unbabel-comet

Install with:
```bash
pip install pandas numpy sacrebleu nltk torch unbabel-comet
```
