# Navigating the Mind Maze: Predicting Mental Health Diagnosis from Sleep & Stress Patterns

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-v1.0%2B-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-v1.4%2B-lightgrey?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-v3.5%2B-red?style=for-the-badge&logo=matplotlib&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 💡 Project Overview

Mental health is a critical aspect of overall well-being, yet understanding its predictors can be complex. This project delves into a fascinating dataset exploring the intricate connections between **sleep habits, stress levels, and the likelihood of a professional mental health diagnosis**. Utilizing supervised machine learning, we aim to uncover patterns that contribute to understanding potential indicators of mental health conditions.

This repository hosts the code and analysis for classifying individuals based on whether they have received a professional mental health diagnosis, using various tree-based and linear classification models.

## 🎯 The Challenge: Unmasking Hidden Patterns

Our primary goal is to build a predictive model that can identify individuals who have been professionally diagnosed with a mental health condition. However, this journey often comes with its own "mind maze":

* **Categorical Data Complexity:** The dataset is rich with categorical responses (e.g., sleep durations, stress levels described in words), requiring careful transformation into a machine-readable format.
* **The Silent Majority:** A significant challenge encountered was **class imbalance**, where the number of individuals *without* a diagnosis far outweighed those *with* one. This often leads models to accurately predict the majority class but completely miss the minority – a common pitfall in real-world health data.

## 📊 Dataset

The core of this project is the `Dataset.xlsx` file. It comprises survey responses covering:

* **Demographics:** (e.g., Gender, Age, Major)
* **Sleep Patterns:** (e.g., average sleep hours, time to fall asleep, sleep disturbances)
* **Stress Levels:** (e.g., stress in various situations like exams, social interactions, assignments)
* **Target Variable:** Whether an individual has been professionally diagnosed with a mental health condition.

## 🛠️ Methodology & Algorithms

Our approach involves a standard machine learning pipeline:

1.  **Data Loading & Cleaning:** Reading the `.xlsx` file and standardizing column names for easier access.
2.  **Feature Engineering (Categorical Encoding):** Transforming all categorical textual responses into numerical representations using `LabelEncoder`, making them suitable for machine learning algorithms. The target variable is also encoded.
3.  **Data Splitting:** Dividing the dataset into training (80%) and testing (20%) sets to ensure robust model evaluation on unseen data.
4.  **Supervised Classification Models:**
    * **Logistic Regression:** A fundamental linear model for binary classification, providing a baseline understanding.
    * **Random Forest Classifier:** A powerful ensemble tree-based model known for its robustness and ability to handle complex relationships. It leverages multiple decision trees to improve predictive accuracy.
    * **Decision Tree Classifier (for Visualization):** A single decision tree model is used specifically to visualize the tree structure and understand splitting criteria (like Gini impurity) in a clear, interpretable manner.

## 🚀 Key Findings & Results

After running the models, we observed a crucial pattern:

* **High Overall Accuracy (e.g., ~92%):** Both Logistic Regression and Random Forest models showed high accuracy on the test set.
* **The Imbalance Conundrum (Precision, Recall, F1-Score = 0.00):** Despite high accuracy, the Precision, Recall, and F1-Score for the positive class (individuals *with* a diagnosis) were consistently **0.00**. This strongly indicates a severe **class imbalance** problem. The models learned to predict the majority class (no diagnosis) for almost all instances, thus achieving high overall accuracy while completely failing to identify the minority, more critical class.
* **Feature Importances:** The Random Forest model provided insights into the most influential factors:
    * `Have_you_ever_received_treatmentsupport_for_a_mental_health_problem` emerges as the top predictor.
    * `Gender`, `Age`, and various sleep/stress-related questions also play significant roles.
* **Decision Tree Visualization:** A single, shallow Decision Tree helps to visually trace initial decision paths and understand how Gini impurity drives splits, even if the full complexity of the problem requires deeper models.

## 📈 Future Enhancements

To overcome the challenges, particularly the class imbalance, and build a more reliable predictive model, future work could include:

* **Addressing Class Imbalance:**
    * **Oversampling:** Techniques like SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic samples of the minority class.
    * **Undersampling:** Reducing the number of majority class samples.
    * **Class Weighting:** Adjusting model parameters to give more importance to the minority class during training.
* **Hyperparameter Tuning:** Optimizing the parameters of the chosen models (e.g., `n_estimators`, `max_depth` for Random Forest) using techniques like GridSearchCV or RandomizedSearchCV.
* **Cross-Validation:** Employing k-fold cross-validation for more robust model evaluation.
* **Exploring Other Models:** Investigating algorithms inherently robust to imbalance or those that can handle mixed data types more natively (e.g., Gradient Boosting Machines like XGBoost, LightGBM).
* **Deeper Feature Analysis:** Beyond importance, understanding feature interactions.

## 🚀 How to Run the Code

### Prerequisites

* Python 3.x
* `pip` (Python package installer)

### Installation

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/Mental-Health-Analysis-ML.git](https://github.com/YourGitHubUsername/Mental-Health-Analysis-ML.git)
    cd Mental-Health-Analysis-ML
    ```
    (Replace `YourGitHubUsername` and `Mental-Health-Analysis-ML` with your actual GitHub username and repository name.)

2.  **(Optional) Create and activate a virtual environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required Python libraries:**
    Ensure `Dataset.xlsx` and `requirements.txt` are in the same directory.
    ```bash
    pip install -r requirements.txt
    ```

### Running the Script

1.  Ensure `Dataset.xlsx` is in the same directory as `mental_health_analysis.py`.
2.  Execute the main analysis script:
    ```bash
    python mental_health_analysis.py
    ```
    This will print model performance metrics to the console and display a graphical visualization of a Decision Tree.

## 📁 Project Structure

Okay, here's a "fascinating" README.md file designed to be engaging, informative, and to effectively showcase your project on GitHub. It highlights the problem, your approach, and crucially, the interesting challenges you encountered (like class imbalance).

Copy the content below and save it as README.md in your project's root directory.

Markdown

# Navigating the Mind Maze: Predicting Mental Health Diagnosis from Sleep & Stress Patterns

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-v1.0%2B-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-v1.4%2B-lightgrey?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-v3.5%2B-red?style=for-the-badge&logo=matplotlib&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 💡 Project Overview

Mental health is a critical aspect of overall well-being, yet understanding its predictors can be complex. This project delves into a fascinating dataset exploring the intricate connections between **sleep habits, stress levels, and the likelihood of a professional mental health diagnosis**. Utilizing supervised machine learning, we aim to uncover patterns that contribute to understanding potential indicators of mental health conditions.

This repository hosts the code and analysis for classifying individuals based on whether they have received a professional mental health diagnosis, using various tree-based and linear classification models.

## 🎯 The Challenge: Unmasking Hidden Patterns

Our primary goal is to build a predictive model that can identify individuals who have been professionally diagnosed with a mental health condition. However, this journey often comes with its own "mind maze":

* **Categorical Data Complexity:** The dataset is rich with categorical responses (e.g., sleep durations, stress levels described in words), requiring careful transformation into a machine-readable format.
* **The Silent Majority:** A significant challenge encountered was **class imbalance**, where the number of individuals *without* a diagnosis far outweighed those *with* one. This often leads models to accurately predict the majority class but completely miss the minority – a common pitfall in real-world health data.

## 📊 Dataset

The core of this project is the `Dataset.xlsx` file. It comprises survey responses covering:

* **Demographics:** (e.g., Gender, Age, Major)
* **Sleep Patterns:** (e.g., average sleep hours, time to fall asleep, sleep disturbances)
* **Stress Levels:** (e.g., stress in various situations like exams, social interactions, assignments)
* **Target Variable:** Whether an individual has been professionally diagnosed with a mental health condition.

## 🛠️ Methodology & Algorithms

Our approach involves a standard machine learning pipeline:

1.  **Data Loading & Cleaning:** Reading the `.xlsx` file and standardizing column names for easier access.
2.  **Feature Engineering (Categorical Encoding):** Transforming all categorical textual responses into numerical representations using `LabelEncoder`, making them suitable for machine learning algorithms. The target variable is also encoded.
3.  **Data Splitting:** Dividing the dataset into training (80%) and testing (20%) sets to ensure robust model evaluation on unseen data.
4.  **Supervised Classification Models:**
    * **Logistic Regression:** A fundamental linear model for binary classification, providing a baseline understanding.
    * **Random Forest Classifier:** A powerful ensemble tree-based model known for its robustness and ability to handle complex relationships. It leverages multiple decision trees to improve predictive accuracy.
    * **Decision Tree Classifier (for Visualization):** A single decision tree model is used specifically to visualize the tree structure and understand splitting criteria (like Gini impurity) in a clear, interpretable manner.

## 🚀 Key Findings & Results

After running the models, we observed a crucial pattern:

* **High Overall Accuracy (e.g., ~92%):** Both Logistic Regression and Random Forest models showed high accuracy on the test set.
* **The Imbalance Conundrum (Precision, Recall, F1-Score = 0.00):** Despite high accuracy, the Precision, Recall, and F1-Score for the positive class (individuals *with* a diagnosis) were consistently **0.00**. This strongly indicates a severe **class imbalance** problem. The models learned to predict the majority class (no diagnosis) for almost all instances, thus achieving high overall accuracy while completely failing to identify the minority, more critical class.
* **Feature Importances:** The Random Forest model provided insights into the most influential factors:
    * `Have_you_ever_received_treatmentsupport_for_a_mental_health_problem` emerges as the top predictor.
    * `Gender`, `Age`, and various sleep/stress-related questions also play significant roles.
* **Decision Tree Visualization:** A single, shallow Decision Tree helps to visually trace initial decision paths and understand how Gini impurity drives splits, even if the full complexity of the problem requires deeper models.

## 📈 Future Enhancements

To overcome the challenges, particularly the class imbalance, and build a more reliable predictive model, future work could include:

* **Addressing Class Imbalance:**
    * **Oversampling:** Techniques like SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic samples of the minority class.
    * **Undersampling:** Reducing the number of majority class samples.
    * **Class Weighting:** Adjusting model parameters to give more importance to the minority class during training.
* **Hyperparameter Tuning:** Optimizing the parameters of the chosen models (e.g., `n_estimators`, `max_depth` for Random Forest) using techniques like GridSearchCV or RandomizedSearchCV.
* **Cross-Validation:** Employing k-fold cross-validation for more robust model evaluation.
* **Exploring Other Models:** Investigating algorithms inherently robust to imbalance or those that can handle mixed data types more natively (e.g., Gradient Boosting Machines like XGBoost, LightGBM).
* **Deeper Feature Analysis:** Beyond importance, understanding feature interactions.

## 🚀 How to Run the Code

### Prerequisites

* Python 3.x
* `pip` (Python package installer)

### Installation

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/Mental-Health-Analysis-ML.git](https://github.com/YourGitHubUsername/Mental-Health-Analysis-ML.git)
    cd Mental-Health-Analysis-ML
    ```
    (Replace `YourGitHubUsername` and `Mental-Health-Analysis-ML` with your actual GitHub username and repository name.)

2.  **(Optional) Create and activate a virtual environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required Python libraries:**
    Ensure `Dataset.xlsx` and `requirements.txt` are in the same directory.
    ```bash
    pip install -r requirements.txt
    ```

### Running the Script

1.  Ensure `Dataset.xlsx` is in the same directory as `mental_health_analysis.py`.
2.  Execute the main analysis script:
    ```bash
    python mental_health_analysis.py
    ```
    This will print model performance metrics to the console and display a graphical visualization of a Decision Tree.

## 📁 Project Structure

.
├── Dataset.xlsx              # The raw dataset
├── mental_health_analysis.py # Main Python script for analysis and modeling
├── README.md                 # This file: project description and instructions
├── requirements.txt          # List of Python dependencies



## 📜 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

Feel free to contribute, open issues, or suggest improvements!
