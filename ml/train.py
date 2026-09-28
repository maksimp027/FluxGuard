import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def generate_synthetic_data(n_samples=10000):
    """Generate synthetic pre-authorization transactions matching the 19-element ONNX input tensor contract."""
    np.random.seed(42)

    amount = np.random.uniform(1.0, 5000.0, n_samples)
    amount_cents = np.round(amount * 100)
    cvv_match = np.random.choice([0.0, 1.0], n_samples, p=[0.05, 0.95])
    is_3ds_passed = np.random.choice([0.0, 1.0], n_samples, p=[0.2, 0.8])
    account_age = np.random.uniform(0.0, 1000.0, n_samples)

    merchant = np.random.multinomial(1, [0.4, 0.2, 0.2, 0.1, 0.1], size=n_samples)
    currency = np.random.multinomial(1, [0.6, 0.2, 0.1, 0.1], size=n_samples)

    card_bin = np.random.randint(400000, 499999, size=n_samples)
    hour = np.random.randint(0, 24, size=n_samples)

    tx_count_1h = np.random.randint(1, 10, size=n_samples)
    tx_amount_1h = amount + np.random.uniform(0.0, 1000.0, n_samples)

    # Index 18 is zero-padded to align with the fixed 19-feature tensor shape expected by the Go ONNX session.
    padding = np.zeros(n_samples)

    X = np.column_stack(
        [
            amount,
            amount_cents,
            cvv_match,
            is_3ds_passed,
            account_age,
            merchant,
            currency,
            card_bin,
            hour,
            tx_count_1h,
            tx_amount_1h,
            padding,
        ]
    )

    risk_score = (
        (1.0 - cvv_match) * 2
        + (1.0 - is_3ds_passed) * 1.5
        + merchant[:, 3] * 1.2
        + merchant[:, 4] * 1.5
        + np.random.normal(0, 1, n_samples)
    )

    threshold = np.percentile(risk_score, 90)
    y = (risk_score >= threshold).astype(int)

    return X, y


def main():
    print("Generating synthetic data...")
    X, y = generate_synthetic_data()
    print(f"Dataset shape: {X.shape}, Fraud ratio: {np.mean(y):.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("fluxguard_fraud_detection")

    with mlflow.start_run() as run:
        num_negative = np.sum(y_train == 0)
        num_positive = np.sum(y_train == 1)
        scale_pos_weight = num_negative / num_positive

        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "scale_pos_weight": scale_pos_weight,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "random_state": 42,
        }

        print("Training XGBoost model...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        mlflow.log_params(params)

        print("Evaluating model...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        pr_auc = average_precision_score(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        metrics = {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
        }
        mlflow.log_metrics(metrics)

        print("Logging model to MLflow...")
        mlflow.xgboost.log_model(model, "model")

        print("=" * 40)
        print("Training Complete!")
        print(f"Run ID: {run.info.run_id}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("=" * 40)


if __name__ == "__main__":
    main()