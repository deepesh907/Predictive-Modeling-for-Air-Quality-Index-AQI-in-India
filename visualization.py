import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_visuals():
    data = joblib.load("data/processed_data.pkl")
    model = joblib.load("data/hybrid_model.pkl")
    X_test, y_test = data['X_test'], data['y_test']
    
    y_pred = model.predict(X_test)

    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title("Actual vs Predicted AQI (Hybrid Model)")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.savefig("visuals/actual_vs_predicted.png")
    
    # 2. Residual Distribution (Checking for Accuracy Errors)
    plt.figure(figsize=(10, 6))
    sns.histplot(y_test - y_pred, kde=True, color="purple")
    plt.title("Error Distribution (Residuals)")
    plt.savefig("visuals/error_dist.png")
    
    print("📊 Visuals saved in /visuals folder. Ready for presentation!")

if __name__ == "__main__":
    generate_visuals()