![](img\pexels-markus-winkler-1430818-19867471.jpg)


# Credict Risk Prediction

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Ambigapathi-V/Mental-Health-Prediction?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/Ambigapathi-V/Credict-Risk-Prediction)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Ambigapathi-V/Credict-Risk-Prediction)
![GitHub](https://img.shields.io/github/license/Ambigapathi-V/Credict-Risk-Prediction)
![contributors](https://img.shields.io/github/contributors/Ambigapathi-V/Credict-Risk-Prediction)
![codesize](https://img.shields.io/github/languages/code-size/Ambigapathi-V/Credict-Risk-Prediction)




## Project Overview

Lauki Finance, an Indian NBFC, has partnered with AtliQ AI to develop an advanced credit risk model. This model will predict loan default probabilities and categorize loan applications into credit score segments — Poor, Average, Good, and Excellent — similar to the CIBIL scoring system. The project includes creating a predictive model based on Lauki Finance’s historical loan data, developing a scorecard for credit categorization, and building a Streamlit application to facilitate real-time assessment for loan officers. Following model deployment, a performance monitoring system and operational integration with Straight Through Processing (STP) will enhance automation, reducing manual intervention for high-confidence applications.
## Features

- Borrower Demographics
- Loan Details
- Credit Bureau Information
- Financial Ratios
- Loan Performance Indicators
- Derived Insights


## Demo

Insert gif or link to demo


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

# Installation and Setup

**1.Clone the Repository:**

```bash
  git clone https://github.com/Ambigapathi-V/Credit-Risk-Model
  cd Credit-Risk-Model
```
**2.Set Up a Virtual Environment:**

   ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the main.py:**
```bash
python main.py
```

**4. Run the Streamlit**
```
streamlit run app.py
```

## Base URL
 http://premium-price-prediction-insurance.streamlit.app/


 
## Codes and Resources Used
In this section I give user the necessary information about the software requirements.
- **Editor Used:** Visual Studio Code
- **Python Version:** 3.10

## Python packages Used
1. **pandas:** For data manipulation and analysis
2. **numpy:** For numerical computations
3. **scikit-learn:** For machine learning algorithms
4. **streamlit:** For creating web applications
5. **matplotlib:** For data visualization 
6. **seaborn:** For data visualization
7. **FastAPi** for building APIs



## Project Structure

## Data

The data used in the Code Basics and Machine Learning Basics courses are typically from various publicly available datasets that focus on predictive modeling and classification tasks

## Source Data


The datasets you've used in the CodeBasics Machine Learning course likely refer to datasets commonly used in educational contexts for training and testing machine learning models. CodeBasics often covers datasets that help learners practice various aspects of machine learning, such as classification, regression, clustering, and more

## Data Ingestion

1. **Data Loading:** The ingestion process loads three datasets: customer data, loan data, and bureau data from specified paths.
2. **Data Merging:** Merges the customer, loan, and bureau datasets using cust_id as the common key, combining all relevant information into a single dataset.
3. **Data Saving:** After merging, the consolidated dataset is saved to an output location for further processing or analysis.
4. **Logging:** Each step of the data loading and merging process is logged, including the shapes of the loaded datasets, to help track and troubleshoot the workflow.
5. **Configuration:** The data ingestion process is driven by a DataIngestionConfig class, which specifies the input file paths and output location for the final merged dataset.

## Data Preprocessing

1. **Data Preprocessing Pipeline:** Prepares raw data for model training by scaling features, encoding variables, and handling missing values.

2. **Configurable:** Configurations are managed by ConfigurationManager, allowing easy adjustments without code changes.

3. **Modular Design:** Data preprocessing and cleaning are separated into distinct components for easier maintenance and extension.

4. **Logging:** Tracks pipeline execution with detailed logs for each stage and errors.

5. **Error Handling:** Custom exceptions are raised to capture and handle issues during preprocessing and cleaning.

6. **Sequential Execution:** Stages are executed in sequence, ensuring clear separation between preprocessing and cleaning.

7. **Error Recovery:** Logs and halts execution on errors to prevent training on incorrect data.

8. **Execution Flow:** Each pipeline runs independently or as part of larger workflows using the if __name__ == "__main__": block.

```bash
   ├── data
│   ├── raw_data                     # Original, unprocessed datasets
│   ├── cleaned_data                 # Processed data ready for training
├── src
│   ├── Credit_Risk_Model            # Core implementation
│   │   ├── config                    # Configuration files
│   │   ├── components                # Data ingestion, cleaning, preprocessing, and model training
│   │   ├── pipeline                # Manages pipeline orchestration
│   │   ├── entity                  # Configuration and input/output management
│   │   ├── utils                   # Helper functions and logging
│   │   └── exception               # Custom exception handling
├── notebooks                       # Jupyter notebooks for exploration and analysis
├── requirements.txt                # Project dependencies
├── LICENSE                         # License for the project
├── README.md                      # Project overview and instructions
└── .gitignore                     # Git ignore file

```

## Result And Evaluation

This project aims to predict credit risk by building a classification model. Below are the key metrics and evaluation methodologies used to assess the model's performance:

### Evaluation Metrics:
1. **F1 Score:** 0.78 (A balanced measure of precision and recall)
2. **Precision:** 0.83 (Accuracy of positive predictions)
3. **Recall:** 0.74 (Ability to identify true positives)
4. **Accuracy:** 0.83 (Overall performance of the model)
### Evaluation Methodology:
1. **Train-Test Split:** Divided data into training and testing subsets to ensure robust model performance.
**ROC Curve & AUC:** Plotted the trade-off between true positive and false positive rates.
Precision-Recall Curve: Used to evaluate performance on imbalanced datasets.

![]()

For more detailed evaluation metrics, refer to Dagshub Code : https://dagshub.com/Ambigapathi-V/credict-Risk-Model/experiments#/



## Future Work

Model Optimization: Implement hyperparameter tuning, cross-validation, and ensemble learning techniques to improve model accuracy and robustness.

1. **Scalability and Deployment:** Build a scalable solution using distributed frameworks and deploy the model as a real-time API using tools like Docker, Flask, or cloud services.

2. **Explainability and Interpretability:** Integrate model explainability techniques like SHAP or LIME to enhance transparency and trust in model predictions.

3. **Automated Machine Learning (AutoML):** Explore AutoML frameworks for automated model selection and hyperparameter tuning, reducing manual effort.

4. **Performance Monitoring:** Implement model drift detection and automated retraining pipelines to keep models up-to-date with evolving data patterns.

## Deployment

To deploy this project run

```bash
  npm run deploy
```


    
## Acknowledgements

- **Pandas**: Used for data manipulation and analysis. [Pandas](https://pandas.pydata.org/)
- **Scikit-learn**: Employed for implementing machine learning models. [Scikit-learn](https://scikit-learn.org/)
- **DagsHub**: Utilized for versioning and MLflow tracking. [DagsHub](https://dagshub.com/)

- **Dataset**: Sourced from publicly available datasets.

## License

For this github repository, the License used is [MIT License](https://opensource.org/license/mit/).