import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

# ===============================
# 1. 基本设置
# ===============================

feature_columns = [
    "SLOPE", "ELEVATION", "PD",
    "AT0", "AT10",
    "FC", "CONTAG", "FRAC",
    "SHDI", "CRDI"
]

target_column = "F"

grid_file = r"C:\Users\4\Desktop\残差检查\wangge.xls"
town_file = r"C:\Users\4\Desktop\残差检查\zhenyu.xls"


# ===============================
# 2. 模型训练函数（带网格搜索）
# ===============================

def train_best_xgb(X, y, scale_name):
    print("\n" + "=" * 70)
    print(f"🚀 正在训练模型: {scale_name}")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9]
    }

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("✅ 最佳参数:", grid_search.best_params_)

    return best_model


# ===============================
# 3. Cross-scale prediction函数
# ===============================

def cross_scale_prediction(model, X_target, y_target, experiment_name):
    print("\n" + "=" * 60)
    print(experiment_name)
    print("=" * 60)

    y_pred = model.predict(X_target)

    r2 = r2_score(y_target, y_pred)
    rmse = np.sqrt(mean_squared_error(y_target, y_pred))
    mae = mean_absolute_error(y_target, y_pred)

    print(f"Cross-scale R²   = {r2:.3f}")
    print(f"Cross-scale RMSE = {rmse:.3f}")
    print(f"Cross-scale MAE  = {mae:.3f}")

    return r2, rmse, mae


# ===============================
# 4. 主程序：读取数据
# ===============================

df_grid = pd.read_excel(grid_file)
df_town = pd.read_excel(town_file)

# 特征一致性检查
print("\n特征变量一致性检查:",
      list(df_grid[feature_columns].columns)
      == list(df_town[feature_columns].columns))

X_grid = df_grid[feature_columns]
y_grid = df_grid[target_column]

X_town = df_town[feature_columns]
y_town = df_town[target_column]


# ===============================
# 5. 训练两个尺度模型
# ===============================

grid_model = train_best_xgb(X_grid, y_grid, "Grid scale (网格)")
town_model = train_best_xgb(X_town, y_town, "Township scale (镇域)")


# ===============================
# 6. 实验A：Grid → Township
# ===============================

cross_scale_prediction(
    grid_model,
    X_town,
    y_town,
    "实验 A：Grid 模型预测 Township"
)


# ===============================
# 7. 实验B：Township → Grid
# ===============================

cross_scale_prediction(
    town_model,
    X_grid,
    y_grid,
    "实验 B：Township 模型预测 Grid"
)
