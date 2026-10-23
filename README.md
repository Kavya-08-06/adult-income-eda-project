# Adult Income: Exploratory Data Analysis and Deep Learning

## Project overview
This case study explores the UCI Adult (Census Income) dataset and trains a Multi-Layer Perceptron (MLP) classifier to predict whether a person's income category is `<=50K` or `>50K` based on census attributes. It demonstrates data cleaning, missing-value handling, outlier exploration, univariate/bivariate/multivariate visualization, a stratified train/validation/test split, basic hyperparameter tuning, model evaluation, and feature importance.

## Dataset
- **Source:** UCI Machine Learning Repository, Adult dataset
- **Dataset page:** https://archive.ics.uci.edu/dataset/2/adult
- **Download source used by code:** https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
- **Fields:** 14 input attributes plus the income target (15 fields total).
- The script downloads the original files into `data/` on first run. The dataset is not redistributed in this repository.

## Run locally
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python src/adult_income_project.py
```

The first run needs an internet connection to download the dataset. Generated figures, metrics, cleaned data, the trained model, and tables are saved under `outputs/`.

## Repository structure
```text
adult-income-eda-project/
├── data/                 # Downloaded dataset and cleaned CSV (generated)
├── notebooks/            # Notebook can be created/imported from the Python script
├── outputs/              # Figures, model metrics, reports (generated after running)
├── reports/              # Project report
├── src/
│   └── adult_income_project.py
├── requirements.txt
└── README.md
```

## Visualizations included
1. Income class distribution
2. Age distribution by income
3. Weekly hours boxplot by income
4. Numerical correlation heatmap
5. Education frequency plot
6. Age versus weekly hours scatterplot
7. Confusion matrix
8. ROC curve and AUC
9. Training loss curve
10. Permutation feature importance

## Methodology notes
- Exact duplicate rows are removed.
- Missing values are imputed inside the preprocessing pipeline to avoid fitting imputers on validation/test data.
- Numerical fields are median-imputed and standardized; categorical fields are mode-imputed and one-hot encoded.
- Data is split into 70% training, 15% validation, and 15% test data using stratification.
- Hyperparameters are compared on the validation set; the test set is used for final evaluation only.
- Income prediction is a statistical classification task and the data can contain historical social biases. Results should not be used to make high-stakes decisions about individuals.

## Seven stages of visualization
The report connects the work to a practical seven-stage visualization workflow: acquire data, parse/structure it, filter/clean it, mine/aggregate it, represent it visually, refine the visual design, and interpret the results.

## References
See the project report for APA-style references.
