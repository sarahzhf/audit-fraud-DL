# audit-fraud-DL
Audit Fraud Detection – Classification Project

Overview :

This repository contains a full machine-learning workflow to detect high-risk or potentially fraudulent audit cases.
The project includes:
	•	Exploratory Data Analysis (EDA)
	•	Data preprocessing
	•	Training and evaluation of multiple classification models
	•	Comparison of model performances
	•	Analysis of misclassified observations
	•	A simple deep learning model (MLP)

All results are contained in the Jupyter notebook analysis.ipynb.

⸻

Project Structure :

audit-fraud-DL/
│
├── analysis.ipynb          # Main notebook with full workflow
├── audit_data.csv.xls      # Dataset used in the project
├── README.md               # Project documentation
└── requirements.txt        # (Optional) dependencies list


Dataset :

The dataset contains numeric audit indicators and a binary target:
	•	Risk = 1 → fraudulent or high-risk audit
	•	Risk = 0 → normal audit

ID-like or redundant columns were removed during preprocessing.


Methods :

1. Preprocessing
	•	Removal of ID and duplicate columns
	•	Handling of missing values
	•	Selection of numeric features only
	•	Train/test split (80/20, stratified)
	•	StandardScaler for models requiring feature scaling

2. Models Implemented

Models tested in increasing complexity:
	•	DummyClassifier
	•	Logistic Regression
	•	Polynomial Logistic Regression (degree=2)
	•	Decision Tree
	•	Random Forest
	•	K-Nearest Neighbours
	•	Multilayer Perceptron (TensorFlow)

3. Metrics

Each model is evaluated using:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1 Score
	•	ROC-AUC

Confusion matrices and misclassified samples are analysed as well.


Results Summary :

Most models achieve very high performance due to the dataset’s strong separability.

Model	F1-Score	Notes
DummyClassifier	0.00	Baseline
Logistic Regression	~0.97	Strong linear performance
Polynomial Logistic Regression	~0.98	Slight improvement
Decision Tree	1.00	Perfect classification
Random Forest	1.00	Most stable model
KNN	~0.95	Good but lower recall
MLP	~0.98	Deep learning not necessary here

A complete comparison plot is included in the notebook.


Visualisations :

The notebook includes the following:
	•	Target distribution
	•	Correlation heatmap
	•	F1 comparison bar chart
	•	Confusion matrix (Random Forest)
	•	MLP training curve (optional)


How to Run ?

1. Clone the repository

git clone https://github.com/sarahzhf/audit-fraud-DL.git
cd audit-fraud-DL

2. Install dependencies

If you have a requirements file:

pip install -r requirements.txt

Otherwise manually install:

pip install pandas numpy scikit-learn seaborn matplotlib tensorflow-macos tensorflow-metal

3. Launch Jupyter Notebook

jupyter notebook analysis.ipynb

Run all cells to reproduce the full workflow.

Notes
	•	The dataset is highly separable, which explains the near-perfect performance of several models.
	•	In a real audit fraud context, additional noise, complexity, and imbalance would be expected.
	•	The purpose of this repository is to compare models and understand their behaviour on a structured dataset.