from models_baselines import run_all_baselines

# X_train, y_train, X_val, y_val, X_test, y_test
# should already come from your preprocessing pipeline

results = run_all_baselines(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
)

print("\nBaseline ranking:")
for r in results:
    print(r)
