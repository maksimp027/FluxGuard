import os
import mlflow
import mlflow.xgboost
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType


def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("fluxguard_fraud_detection")

    print("Searching for the best model in MLflow...")
    runs = mlflow.search_runs(
        experiment_names=["fluxguard_fraud_detection"],
        order_by=["metrics.pr_auc DESC"],
        max_results=1,
    )

    if runs.empty:
        print("No runs found in MLflow for this experiment.")
        return

    best_run = runs.iloc[0]
    best_run_id = best_run["run_id"]
    best_pr_auc = best_run["metrics.pr_auc"]

    print(f"Best Run ID: {best_run_id}")
    print(f"Best PR-AUC: {best_pr_auc:.4f}")

    print(f"Loading model for Run ID: {best_run_id}...")
    model_uri = f"runs:/{best_run_id}/model"
    xgb_model = mlflow.xgboost.load_model(model_uri)

    if hasattr(xgb_model, "get_booster"):
        booster = xgb_model.get_booster()
    else:
        booster = xgb_model

    # Reset feature_names so onnxmltools maps positional f0..f18 nodes without name-binding failures.
    booster.feature_names = None

    print("Converting model to ONNX format...")
    initial_types = [("float_input", FloatTensorType([None, 19]))]

    onnx_model = onnxmltools.convert_xgboost(
        booster,
        initial_types=initial_types,
        target_opset=15,
    )

    output_dir = os.path.join(os.path.dirname(__file__), "model_artifacts")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "chargeback_model.onnx")
    print(f"Saving ONNX model to {output_path}...")

    onnxmltools.utils.save_model(onnx_model, output_path)

    print("=" * 40)
    print(f"Successfully exported best model (Run ID: {best_run_id}) to ONNX!")
    print("=" * 40)


if __name__ == "__main__":
    main()