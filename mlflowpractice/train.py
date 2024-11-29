import argparse
import mlflow

def train(alpha, l1_ratio):
    # Log parameters and a dummy metric for MLflow
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("accuracy", 0.95)
    print(f"Training with alpha={alpha}, l1_ratio={l1_ratio}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5, help="Regularization strength")
    parser.add_argument("--l1_ratio", type=float, default=0.1, help="ElasticNet mixing parameter")
    args = parser.parse_args()
    train(args.alpha, args.l1_ratio)

