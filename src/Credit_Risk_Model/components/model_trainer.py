import os
import sys
import mlflow
from src.Credit_Risk_Model.exception import CustomException
from src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.entity.config_entity import ModelTrainingConfig
from src.Credit_Risk_Model.utils.common import save_object, load_numpy_array_data, evaluate_models
from src.Credit_Risk_Model.utils.ml_utlis.metric.classification_metrics import get_classification_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix

# Initialize DagsHub MLflow tracking
dagshub.init(repo_owner='Ambigapathi-V', repo_name='credict-Risk-Model', mlflow=True)

class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.train_path = config.train_path
        self.test_path = config.test_path
        self.model_path = config.model_path
        self.preprocessor_path = config.preprocessor

    def track_mlflow(self, model_name, y_true, y_pred, y_prob=None):
        """
        Log classification metrics and visualizations to MLflow.
        """
        try:
            # Create a directory for saving plots
            os.makedirs('artifacts', exist_ok=True)

            # Get classification metrics
            classification_metrics = get_classification_score(y_true, y_pred)

            # Extract the metrics from ClassificationMetricArtifact
            accuracy = accuracy_score(y_true, y_pred)
            f1 = classification_metrics.f1_score
            precision = classification_metrics.precision_score
            recall = classification_metrics.recall_score

            # Start MLflow run
            with mlflow.start_run():
                # Log metrics to MLflow
                mlflow.log_metric('accuracy', accuracy)
                mlflow.log_metric('f1_score', f1)
                mlflow.log_metric('precision_score', precision)
                mlflow.log_metric('recall_score', recall)

                # Log ROC curve and AUC if y_prob is provided (for probabilistic classifiers)
                if y_prob is not None:
                    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
                    roc_auc = auc(fpr, tpr)

                    # Log the ROC curve
                    plt.figure()
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC)')
                    plt.legend(loc='lower right')

                    # Save the ROC curve plot
                    roc_curve_path = 'artifacts/roc_curve.png'
                    plt.savefig(roc_curve_path)
                    plt.close()

                    # Log the ROC curve plot as an artifact in MLflow
                    mlflow.log_artifact(roc_curve_path)

                    # Log AUC score
                    mlflow.log_metric('auc_score', roc_auc)

                # Generate and log confusion matrix plot
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('True')

                # Save the confusion matrix plot
                confusion_matrix_path = 'artifacts/confusion_matrix.png'
                plt.savefig(confusion_matrix_path)
                plt.close()

                # Log the confusion matrix plot as an artifact
                mlflow.log_artifact(confusion_matrix_path)

                # Log the trained model
                mlflow.sklearn.log_model(model_name, 'model')

        except Exception as e:
            print(f"An error occurred while tracking the model: {str(e)}")
            # Log the exception for debugging purposes
            mlflow.log_param('error', str(e))

    def train_model(self, X_train, y_train, X_test, y_test):
        """
        Train models and log results to MLflow.
        """
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }
        params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
            },
            "Random Forest": {
                'n_estimators': [8, 16, 32, 128, 256]
            },
            "Gradient Boosting": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {},
            "AdaBoost": {
                'learning_rate': [0.1, 0.01, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

        # Train models and evaluate performance
        model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                       models=models, param=params)

        # Identify the best model based on evaluation score
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]
        
        # Evaluate model on training data
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        
        # Track the training metrics with MLflow
        self.track_mlflow(best_model_name, y_train, y_train_pred)
        
        # Evaluate model on test data
        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        
        # Track the test metrics with MLflow
        self.track_mlflow(best_model_name, y_test, y_test_pred)

        # Save the trained model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        save_object(self.model_path, best_model)

    def initiate_model_trainer(self):
        """
        Initialize training by loading data, splitting into features and target, and training the model.
        """
        try:
            # Load train and test data
            train_arr = load_numpy_array_data(self.train_path)
            test_arr = load_numpy_array_data(self.test_path)

            # Split data into features and target
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Train the model
            self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            logger.error(f"Error in model trainer: {e}")
            raise CustomException(e, sys) from e
