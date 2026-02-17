import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

def train_hybrid():
    data = joblib.load("data/processed_data.pkl")
    X_train, y_train = data['X_train'], data['y_train']

    # Define high-performance members of the Hybrid model
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=8)

    # Combine into a Hybrid Voting Regressor
    hybrid_model = VotingRegressor(estimators=[('rf', rf), ('xgb', xgb)])
    
    print("🚀 Training Hybrid XAI-HAP Model (Ensemble)...")
    hybrid_model.fit(X_train, y_train)

    # Save the model for the visualization script
    joblib.dump(hybrid_model, "data/hybrid_model.pkl")
    print("✅ Hybrid Model saved to data/hybrid_model.pkl")

if __name__ == "__main__":
    train_hybrid()