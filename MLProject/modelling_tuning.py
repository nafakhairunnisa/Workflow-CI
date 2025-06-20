import mlflow
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
import argparse
import joblib

load_dotenv()

# DagsHub setup
try:
    from dagshub import auth, init
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if dagshub_token:
        auth.add_app_token(dagshub_token)
        init(repo_owner='nafakhairunnisa', repo_name='Membangun_model', mlflow=True)
    else:
        print("DAGSHUB_TOKEN not set, skipping DagsHub init.")
except ImportError:
    print("DagsHub not installed, skipping DagsHub integration.")
except Exception as e:
    print(f"Skipping DagsHub integration due to error: {e}")

# Load data function
def load_data(data_dir):
    X_train = pd.read_csv(data_dir / 'X_train.csv')
    X_test  = pd.read_csv(data_dir / 'X_test.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze("columns")
    y_test  = pd.read_csv(data_dir / 'y_test.csv').squeeze("columns")
    return X_train, X_test, y_train, y_test

# Log metrics function
def log_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = {
        'accuracy': report['accuracy'],
        'f1_weighted': report['weighted avg']['f1-score'],
        'roc_auc': roc_auc_score(y_true, y_pred),
        'precision_class0': report['0']['precision'],
        'recall_class1': report['1']['recall'],
        'log_loss': log_loss(y_true, y_pred, labels=[0, 1])
    }
    mlflow.log_metrics(metrics)
    mlflow.log_dict(report, "classification_report.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='personality_preprocessing')
    args = parser.parse_args()

    data_dir = Path(args.data_path)

    X_train, X_test, y_train, y_test = load_data(data_dir)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    with mlflow.start_run(run_name="RF_Tuned"):
        mlflow.log_params(grid_search.best_params_)
        y_pred = best_model.predict(X_test)
        log_metrics(y_test, y_pred)
        mlflow.sklearn.log_model(best_model, "model")

        # Save model locally and log as artifact
        model_path = "rf_model_tuned.joblib"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        print(f"Tuned RF - F1 Weighted: {f1_score(y_test, y_pred, average='weighted'):.4f}")

        # log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')

        plt.savefig("confusion_matrix.png")
        plt.close()

        mlflow.log_artifact("confusion_matrix.png")

        print("Confusion matrix logged.")