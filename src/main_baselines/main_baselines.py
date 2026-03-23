import pandas as pd
from run_baselines import run_all_baselines

# These arrays must already come from your preprocessing pipeline
# X_train, y_train, X_val, y_val, X_test, y_test = ...

results = run_all_baselines(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    seed=42
)

results_df = pd.DataFrame(results)
print("\nBaseline ranking:")
print(results_df)
