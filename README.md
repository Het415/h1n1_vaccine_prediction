H1N1 Vaccine Prediction
This project focuses on predicting the likelihood of individuals receiving the H1N1 flu vaccine based on their social, economic, and demographic background, as well as their personal opinions and behaviors regarding the virus.

 Project Overview
The goal is to build a classification model that predicts whether a person received the H1N1 vaccine. The dataset contains a variety of features, ranging from behavioral habits (like hand washing and mask-wearing) to personal opinions (perceived effectiveness of the vaccine) and demographic data (age, race, sex).

 Dataset
The dataset is sourced from the H1N1 Vaccine Prediction Dataset.

Key Features:

h1n1_worry: Level of worry about the H1N1 flu.

h1n1_awareness: Knowledge level about H1N1.

behavioral features: Hand washing, face mask usage, social distancing, etc.

perceptions: Perceived risk of flu and perceived effectiveness of the vaccine.

demographics: Age bracket, race, sex, income level, education, etc.

Target Variable: h1n1_vaccine (0 = No, 1 = Yes).

 Technologies Used
Python

Pandas & NumPy for data manipulation.

Matplotlib & Seaborn for data visualization.

Scikit-Learn for machine learning and evaluation.

Workflow
1. Data Cleaning & Preprocessing
Handling Missing Values: Rows with missing values were dropped to maintain data integrity for the specific models tested.

Feature Selection: Several high-cardinality or less relevant categorical columns (like qualification, income_level, and employment) were dropped to simplify the model.

Encoding: Categorical variables such as age_bracket, race, and sex were converted into numerical format using One-Hot Encoding (pd.get_dummies).

Train-Test Split: The data was split into training (67%) and testing sets (33%) using stratified sampling to ensure balanced class distribution.

2. Exploratory Data Analysis (EDA)
Visualized the distribution of the target variable.

Analyzed the relationship between vaccination rates and features like "H1N1 Worry," "Age Bracket," and "Race" using stacked bar charts.

Created cross-tabulations to understand behavioral correlations.

3. Machine Learning Models
Three main modeling approaches were implemented:

Logistic Regression: Used as a baseline classifier.

Decision Tree Classifier: Tested both a default tree (which showed significant overfitting) and a tuned tree with max_depth=7.

Bagging Classifier: An ensemble method using 50 tuned Decision Trees to improve model stability and performance.
