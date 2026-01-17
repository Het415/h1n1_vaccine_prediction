# H1N1 Vaccine Prediction

A machine learning classification project that predicts whether individuals received the H1N1 flu vaccine based on behavioral, demographic, and opinion-related features using Logistic Regression.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Het415/h1n1_vaccine_prediction/blob/main/H1N1_VACCINE__PREDICTION.ipynb)

## Overview

This project analyzes the National 2009 H1N1 Flu Survey data to build a predictive model that identifies factors influencing vaccine adoption. The model achieves 82% accuracy in predicting vaccination status, providing insights into public health behaviors and vaccination patterns during the H1N1 pandemic.

## Problem Statement

Understanding the factors that influence H1N1 vaccine uptake is crucial for public health planning and targeted vaccination campaigns. This project aims to:
- Predict H1N1 vaccine adoption using individual characteristics
- Identify key behavioral and demographic factors associated with vaccination
- Provide actionable insights for improving vaccination rates

## Features

- **Comprehensive EDA**: Detailed exploratory analysis of behavioral, opinion, and demographic factors
- **Data Preprocessing**: Robust handling of missing values across 33 features
- **Feature Engineering**: One-hot encoding of categorical variables for model compatibility
- **Binary Classification**: Logistic Regression model to predict vaccination status
- **Model Evaluation**: Confusion matrix and accuracy metrics for performance assessment
- **Visualization**: Insightful plots showing relationships between features and vaccination rates

## Dataset

The dataset contains 26,707 observations with 33 features:

### Feature Categories

**Behavioral Factors:**
- `h1n1_worry`: Level of worry about H1N1 flu (0-3 scale)
- `h1n1_awareness`: Knowledge about H1N1 (0-2 scale)
- `antiviral_medication`: Taking antiviral medication
- `contact_avoidance`: Avoiding close contact with sick people
- `bought_face_mask`: Purchased face mask
- `wash_hands_frequently`: Frequent handwashing
- `avoid_large_gatherings`: Avoiding large gatherings
- `reduced_outside_home_cont`: Reduced outside home contact
- `avoid_touch_face`: Avoiding touching face

**Opinion-Based Features:**
- `is_h1n1_vacc_effective`: Opinion on H1N1 vaccine effectiveness
- `is_h1n1_risky`: Opinion on H1N1 risk level
- `sick_from_h1n1_vacc`: Concern about getting sick from vaccine
- Similar features for seasonal flu vaccine opinions

**Healthcare-Related:**
- `dr_recc_h1n1_vacc`: Doctor recommended H1N1 vaccine
- `dr_recc_seasonal_vacc`: Doctor recommended seasonal vaccine
- `chronic_medic_condition`: Has chronic medical condition
- `is_health_worker`: Works in healthcare
- `has_health_insur`: Has health insurance

**Demographic Features:**
- `age_bracket`: Age group
- `qualification`: Education level
- `race`: Race/ethnicity
- `sex`: Gender
- `income_level`: Income bracket
- `marital_status`: Marital status
- `housing_status`: Own or rent
- `employment`: Employment status
- `census_msa`: Geographic location type
- `no_of_adults`: Number of adults in household
- `no_of_children`: Number of children in household

**Target Variable:**
- `h1n1_vaccine`: Whether individual received H1N1 vaccine (0 = No, 1 = Yes)

## Tech Stack

**Languages & Libraries:**
- Python 3.x
- pandas - Data manipulation and analysis
- NumPy - Numerical computing
- Matplotlib & Seaborn - Data visualization
- scikit-learn - Machine learning and evaluation metrics

**Environment:**
- Jupyter Notebook / Google Colab

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Het415/h1n1_vaccine_prediction.git
cd h1n1_vaccine_prediction
```

2. Install required packages:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

3. Open the notebook:
```bash
jupyter notebook H1N1_VACCINE__PREDICTION.ipynb
```

## Project Structure

```
h1n1_vaccine_prediction/
│
├── H1N1_VACCINE__PREDICTION.ipynb    # Main Jupyter notebook with complete analysis
├── README.md                          # Project documentation
└── data/                              # Dataset directory
```

## Methodology

### 1. Data Loading and Exploration
- Loaded dataset from remote source with 26,707 records and 33 features
- Examined data structure, types, and target variable distribution
- Identified class imbalance: ~78% did not receive vaccine vs. ~22% received vaccine

### 2. Exploratory Data Analysis
- **Missing Value Analysis**: Identified features with significant missing data
  - `has_health_insur`: 12,274 missing values (46%)
  - `dr_recc_h1n1_vacc` and `dr_recc_seasonal_vacc`: 2,160 missing values each
  - Other features: varying degrees of missingness
- **Feature Distributions**: Analyzed distribution of behavioral and demographic variables
- **Cross-tabulation Analysis**: Examined relationships between features and vaccination status
- **Correlation Analysis**: Identified features most strongly associated with vaccine uptake

### 3. Data Preprocessing
- Handled missing values through appropriate imputation strategies
- Applied one-hot encoding to categorical variables (age_bracket, qualification, race, sex, income_level, marital_status, housing_status, employment, census_msa)
- Converted boolean features to binary numeric format
- Ensured all features were numeric for model compatibility

### 4. Model Development
- **Algorithm**: Logistic Regression (suitable for binary classification)
- **Train-Test Split**: 67% training, 33% testing with stratified sampling
- **Random State**: Set for reproducibility
- Trained model on preprocessed features

### 5. Model Evaluation
- **Training Accuracy**: 82.1%
- **Testing Accuracy**: 82.2%
- **Confusion Matrix**: Visualized true positives, false positives, true negatives, and false negatives
- Minimal overfitting observed (similar train/test performance)

## Results

The Logistic Regression model achieved strong performance with 82% accuracy on the test set, indicating effective prediction of H1N1 vaccine uptake.

**Key Insights:**
- Doctor recommendation is the strongest predictor of vaccination
- Higher H1N1 worry levels correlate with increased vaccination rates
- Health insurance status and healthcare worker status influence vaccine adoption
- Demographic factors (age, education, income) play significant roles
- Behavioral factors (handwashing, avoiding gatherings) show associations with vaccination

**Model Performance:**
- **Accuracy**: 82.2%
- **Balanced Performance**: Similar accuracy on training and test sets
- **Practical Application**: Model can identify individuals likely to skip vaccination for targeted outreach

## Usage

To make predictions with the trained model:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load and prepare your data
# Example: Predict for a new individual
sample_data = {
    'h1n1_worry': [2],
    'h1n1_awareness': [1],
    'dr_recc_h1n1_vacc': [1],
    'chronic_medic_condition': [0],
    'is_health_worker': [0],
    # ... include all features
}

df = pd.DataFrame(sample_data)
prediction = model.predict(df)
probability = model.predict_proba(df)

print(f"Vaccination Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Probability: {probability[0][1]:.2%}")
```

## Key Findings

1. **Doctor Recommendation**: Most influential factor in vaccination decision
2. **Risk Perception**: Higher perceived worry about H1N1 increases vaccination likelihood
3. **Healthcare Access**: Health insurance and healthcare worker status positively correlate
4. **Demographics**: Age, education, and income levels significantly impact vaccination rates
5. **Behavioral Patterns**: Individuals practicing preventive behaviors more likely to vaccinate

## Future Improvements

- **Advanced Models**: Experiment with Random Forest, Gradient Boosting, or Neural Networks
- **Feature Engineering**: Create interaction terms and polynomial features
- **Hyperparameter Tuning**: Optimize model parameters using GridSearchCV or RandomizedSearchCV
- **Class Imbalance**: Apply SMOTE or class weighting to handle imbalanced dataset
- **Feature Selection**: Use recursive feature elimination or L1 regularization
- **Cross-Validation**: Implement k-fold cross-validation for robust evaluation
- **Interpretability**: Add SHAP values or LIME for feature importance analysis
- **Multi-Class Extension**: Extend to predict both H1N1 and seasonal flu vaccination

## Public Health Applications

This model can be valuable for:
- **Targeted Campaigns**: Identify demographics needing outreach
- **Resource Allocation**: Focus efforts on groups with low predicted vaccination rates
- **Policy Planning**: Understand barriers to vaccination for policy interventions
- **Healthcare Communication**: Tailor messaging based on individual characteristics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Dataset source: National 2009 H1N1 Flu Survey (DrivenData)
- Inspiration from public health research on vaccination behavior
- scikit-learn documentation and community

## Contact

Het - [GitHub Profile](https://github.com/Het415)

Project Link: [https://github.com/Het415/h1n1_vaccine_prediction](https://github.com/Het415/h1n1_vaccine_prediction)

---

**Note**: This project is for educational and research purposes. Model predictions should not be used as the sole basis for public health decisions without validation by domain experts.
