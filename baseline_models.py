import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def run_baselines():
    # Load processed data
    data = joblib.load("data/processed_data.pkl")
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42)
    }

    baseline_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        baseline_results[name] = {'MAE': mae, 'R2': r2}
        print(f"✅ {name} Trained. R2: {r2:.4f}")

    # Save results for comparison table in PDF
    pd.DataFrame(baseline_results).T.to_csv("data/baseline_results.csv")
    print("\n--- Baseline Results Saved to data/baseline_results.csv ---")

if __name__ == "__main__":
    run_baselines()