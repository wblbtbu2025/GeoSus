import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# PySAL 空间统计库
from libpysal.weights import KNN
from esda.moran import Moran

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
coord_columns = ["x", "y"]

# ===============================
# 2. 指标函数
# ===============================

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


# ===============================
# 3. Spatial Block CV（空间分块交叉验证）
# ===============================

def spatial_block_cv(df, n_blocks=5):
    """
    简单空间分块CV：按X坐标分组
    """
    print("\n--> 正在进行 Spatial Block Cross-Validation...")

    df = df.copy()
    df["block"] = pd.qcut(df["x"], n_blocks, labels=False)

    block_scores = []

    for b in range(n_blocks):
        train_df = df[df["block"] != b]
        test_df  = df[df["block"] == b]

        X_train = train_df[feature_columns]
        y_train = train_df[target_column]

        X_test  = test_df[feature_columns]
        y_test  = test_df[target_column]

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            objective="reg:squarederror",
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2, rmse, mae = metrics(y_test, y_pred)
        block_scores.append(r2)

    mean_r2 = np.mean(block_scores)

    print(f"✅ Spatial Block CV Mean R² = {mean_r2:.3f}")

    return mean_r2


# ===============================
# 4. 主流程：XGBoost + Moran’s I
# ===============================

def run_xgb_moran_test(file_path, scale_name):

    print("\n" + "=" * 70)
    print(f"🚀 开始分析尺度: {scale_name}")
    print("=" * 70)

    # ---------- Step 1: 读取数据 ----------
    df = pd.read_excel(file_path)
    print(f"数据加载成功: {file_path}")
    print("样本量:", df.shape[0])

    # ---------- Step 2: 提取变量 ----------
    X = df[feature_columns]
    y = df[target_column]
    coords = df[coord_columns]

    # ---------- Step 3: 拆分训练/测试 ----------
    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        X, y, coords, test_size=0.3, random_state=42
    )

    # ---------- Step 4: 网格搜索超参数 ----------
    print("\n--> 正在进行 XGBoost 网格搜索...")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9]
    }

    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\n✅ 超参数搜索完成")
    print("最佳参数:", grid_search.best_params_)

    # ---------- Step 5: 模型性能评估 ----------
    print("\n--> 正在评估模型性能...")

    y_train_pred = best_model.predict(X_train)
    y_test_pred  = best_model.predict(X_test)

    r2_train, rmse_train, mae_train = metrics(y_train, y_train_pred)
    r2_test, rmse_test, mae_test    = metrics(y_test, y_test_pred)

    print("\n===== XGBoost 模型性能结果 =====")
    print(f"Training set: R²={r2_train:.3f}, RMSE={rmse_train:.3f}, MAE={mae_train:.3f}")
    print(f"Test set:     R²={r2_test:.3f}, RMSE={rmse_test:.3f}, MAE={mae_test:.3f}")

    # ---------- Overfitting 提示 ----------
    if (r2_train - r2_test) > 0.15:
        print("⚠️ 提示：训练集与测试集差异较大，可能存在过拟合风险。")

    # ---------- Step 6: 残差 ----------
    residuals = y_test - y_test_pred

    # ---------- Step 7: Moran’s I ----------
    print("\n--> 正在进行 Moran’s I 残差空间自相关检验...")

    w = KNN.from_array(coords_test[["x", "y"]].values, k=8)
    w.transform = "R"

    # ✅ permutations=999 更严谨
    moran = Moran(residuals.values, w, permutations=999)

    print("\n===== Moran’s I Residual Test =====")
    print("Moran’s I =", round(moran.I, 4))
    print("p-value   =", round(moran.p_sim, 4))

    if moran.p_sim < 0.05:
        print("⚠️ 残差存在显著空间聚集 → 说明存在尺度效应或遗漏空间变量")
    else:
        print("✅ 残差无显著空间自相关 → 阈值识别结果稳健")

    # ---------- Step 8: Spatial Block CV ----------
    block_r2 = spatial_block_cv(df)

    return {
        "Scale": scale_name,
        "R2_test": r2_test,
        "RMSE_test": rmse_test,
        "MAE_test": mae_test,
        "Moran_I": moran.I,
        "p_value": moran.p_sim,
        "SpatialBlock_R2": block_r2
    }


# ===============================
# 5. 分别运行网格尺度与镇域尺度
# ===============================

results = []

results.append(
    run_xgb_moran_test(
        r"C:\Users\4\Desktop\残差检查\wangge.xls",
        "Grid scale (网格)"
    )
)

results.append(
    run_xgb_moran_test(
        r"C:\Users\4\Desktop\残差检查\zhenyu.xls",
        "Township scale (镇域)"
    )
)

# ===============================
# 6. 汇总输出（论文表格可直接使用）
# ===============================

summary_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("📌 最终汇总结果（可直接写入论文）")
print("=" * 70)
print(summary_df)

# 保存为Excel
summary_df.to_excel(r"C:\Users\4\Desktop\尺度效应_Moran汇总结果.xlsx", index=False)

print("\n✅ 汇总表格已保存到桌面: 尺度效应_Moran汇总结果.xlsx")
