import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import shap

# Load your data
features = pd.read_csv("features_concrete.csv")
labels = pd.read_csv("strength_concrete.csv")

# Select features
X = features[['Cement', 'Slag', 'Ash', 'Water', 'Superplastic', 'Coarseagg', 'Fineagg', 'Temperature']]
y = labels['Compressive strength']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123)

# Create LightGBM model for regression
model = lgb.LGBMRegressor(
    n_estimators=1000, 
    learning_rate=0.1, 
    max_depth=3, 
    subsample=0.9, 
    random_state=123
)

# Train the model
model.fit(X_train, y_train)

# Get predictions on training and testing sets
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Calculate metrics
train_rmse = mean_squared_error(y_train, train_preds, squared=False)
test_rmse = mean_squared_error(y_test, test_preds, squared=False)
train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)
train_mae = mean_absolute_error(y_train, train_preds)
test_mae = mean_absolute_error(y_test, test_preds)

# Mean Absolute Percentage Error (MAPE) calculation
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

train_mape = calculate_mape(y_train, train_preds)
test_mape = calculate_mape(y_test, test_preds)

print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
print(f"Train R^2: {train_r2:.4f}, Test R^2: {test_r2:.4f}")
print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
print(f"Train MAPE: {train_mape:.4f}, Test MAPE: {test_mape:.4f}")

# Create tables for real and predicted values
train_results = pd.DataFrame({'Real Values': y_train, 'Predicted Values': train_preds})
test_results = pd.DataFrame({'Real Values': y_test, 'Predicted Values': test_preds})

# Save tables to CSV files
train_results.to_csv("train_results.csv", index=False)
test_results.to_csv("test_results.csv", index=False)

# Display tables
print("Training Set Results:")
print(train_results)

print("\nTesting Set Results:")
print(test_results)

# SHAP
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
expected_value = explainer.expected_value
# 计算训练和测试数据的SHAP值
shap_values_train = explainer.shap_values(X_train)
shap_values_test = explainer.shap_values(X_test)

# 以下是您的SHAP可视化代码
# 请确保它适用于您的回归任务，可以根据需要进行修改
# 这些代码将在模型上训练的解释性可视化结果

# 1. SHAP Summary Plot
shap.summary_plot(shap_values, X_train)

# 2. SHAP Bar Plot
shap.summary_plot(shap_values, X_train, plot_type="bar")

# 3. SHAP Dependence Plot for pairwise feature interactions
for feature in X_train.columns:
    shap.dependence_plot(feature, shap_values, X_train)

# 4. SHAP Force Plot for a single prediction
for i in range(5):
    shap_plot = shap.force_plot(explainer.expected_value, shap_values[i,:], X_train.iloc[i,:])
    shap.save_html(f"force_plot_{i}.html", shap_plot)

# 5. SHAP Force Plot for multiple predictions
sample_indices = np.random.choice(X_train.shape[0], 100, replace=False)  # Adjust 100 to your desired sample size
shap.force_plot(explainer.expected_value, shap_values[sample_indices], X_train.iloc[sample_indices])

# 6. SHAP Interaction Values
explainer_with_interaction = shap.TreeExplainer(model, feature_perturbation="interventional")
shap_interaction_values = explainer_with_interaction.shap_interaction_values(X_train)
shap.summary_plot(shap_interaction_values, X_train)

# 7. SHAP Decision Plot
for i in range(5):  # Change 5 to the number of instances you want to visualize
    shap.decision_plot(expected_value, shap_values_test[i], 
                       X_test.iloc[i], 
                       link='logit', 
                       highlight=misclassified if 'misclassified' in locals() else None, 
                       ignore_warnings=True)

# 8. SHAP Heatmap Plot
# 我们使用shap.Explanation构造函数的正确参数
shap_values_explanation = shap.Explanation(values=shap_values[:100],
                                           base_values=np.array([explainer.expected_value] * 100),
                                           data=X_train.iloc[:100].values,
                                           feature_names=X_train.columns.tolist())
# 使用shap.Explanation对象绘制热图
shap.plots.heatmap(shap_values_explanation)

# 9. SHAP Waterfall Plot
# 对每个实例创建一个shap.Explanation对象，然后绘制水坎图
for i in range(1):  # Change 5 to the number of instances you want to visualize
    shap_value_explanation = shap.Explanation(values=shap_values[i],
                                              base_values=explainer.expected_value,
                                              data=X_train.iloc[i],
                                              feature_names=X_train.columns.tolist())
    shap.plots.waterfall(shap_value_explanation, max_display=14)
