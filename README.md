# Predictive Analytics for Mental Health Treatment Outcomes

**Course:** TCX3212 Predictive Analytics  
**Group Name:** Ningxin

## Project Overview
This project develops a **Clinical Decision Support System (CDSS)** aimed at predicting mental health treatment outcomes ("Improved", "No Change", "Deteriorated") using strictly pre-treatment baseline clinical and demographic data. 

Following the **CRISP-DM** methodology, this project tackles the inherent challenges of clinical datasets (e.g., small sample size $n=500$, high noise) by strictly eliminating data leakage, employing rigorous 5-Fold Stratified Cross-Validation, and integrating advanced model interpretability frameworks (SHAP & LIME).

## Repository Structure
Based on the course submission requirements, the repository is structured as follows:

```text
├── data/
│   └── mental_health_diagnosis_treatment.csv   # The cleaned clinical dataset
├── presentation/
│   └── GroupNAME_TCX3212_Presentation.pdf      # Exported slides for the 10-min pitch
├── Mental_Health_Prediction_CRISPDM.ipynb      # Main Jupyter Notebook with full source code
├── GroupNAME_TCX3212_Report.pdf                # Maximum 10-page technical report
└── README.md                                   # Project documentation
```

## Installation & Requirements
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap lime scipy
```

## Methodology Highlights

- Data Leakage Prevention: Explicitly dropped post-treatment variables (e.g., Treatment Progress, Adherence, AI-Detected Emotional State) to simulate a true pre-treatment predictive environment.

- Rigorous Validation: 80/20 Train-Test split performed prior to any scaling/encoding. Hyperparameters tuned using GridSearchCV with StratifiedKFold(n_splits=5).

- Algorithms Evaluated: Dummy Classifier (Baseline), Multinomial Logistic Regression, Support Vector Machine (RBF Kernel), and XGBoost.

- Class Imbalance: Handled mathematically via class_weight='balanced' to penalize minority class errors without introducing synthetic data artifacts.


## Key Findings & Interpretability
- Performance: The baseline clinical features exhibited high noise. XGBoost was selected as the final deployment model. While the overall F1-score highlights the limitations of using only static baseline data for psychiatric predictions, the model successfully captures structural clinical variance.

- Global Interpretability (SHAP): Identifies macroeconomic drivers of patient outcomes (e.g., baseline Symptom Severity and Stress Levels).

- Local Interpretability (LIME): Generates instance-level, patient-specific explanations to maintain the CDSS as a "glass box" for clinicians.