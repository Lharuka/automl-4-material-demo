import pandas as pd
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import shap
from itertools import product

def read_df(data_path: str, n_tgt: int = 1) -> pd.DataFrame:
    """
    读取不含行号列名的csv，添加列名，ntgt为target数，去重，转浮点数，返回df
    """
    df = pd.read_csv(data_path, index_col=None, header=None)
    columns = []
    for i in range(df.shape[1] - n_tgt):
        columns.append(f"feature_{i + 1}")

    for i in range(df.shape[1] - n_tgt, df.shape[1]):
        columns.append(f"target_{i + 1 - df.shape[1] + n_tgt}")
    df.columns = columns

    df = df.loc[:, ~(df.nunique(dropna=False) <= 1)]
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df

def automl(df: pd.DataFrame, model_path: str, n_tgt: int = 1, tgt: str = "target"):
    """
    automl预测target，返回真值和预测值
    """
    df1 = df.iloc[:, :df.shape[1] - n_tgt]
    df1[tgt] = df[tgt]
    predictor = TabularPredictor(label=tgt, problem_type="regression", path=model_path)
    predictor.fit(train_data=df1)
    true = df1[tgt].values
    pred = predictor.predict(df1).values
    print("\n========================================")
    print(f"模型已保存至{model_path}")

    return true, pred, predictor

def plot_fig(true, pred, fig_path) -> None:
    """
    计算metrics，画校准曲线
    """
    p1 = min(true.min(), pred.min())
    p2 = max(true.max(), pred.max())

    rmse = root_mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    print(f"RMSE: {rmse:.5f}, MAE: {mae:.5f}, R^2: {r2:.4f}")

    plt.figure()
    plt.scatter(true, pred, label=f"RMSE: {rmse:.5f}\nMAE: {mae:.5f}")
    plt.plot([p1, p2], [p1, p2], color="red", ls="--", label=f"r2: {r2:.3f}")
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.legend()
    plt.savefig(fig_path)
    plt.show()

def shap_al(predictor, df, shap_path, n_tgt=1):
    df1 = df.iloc[:, :df.shape[1] - n_tgt]
    def model_predict(x):
        x_df = pd.DataFrame(x)
        x_df.columns = df1.columns
        return predictor.predict(x_df).values
    
    background = df1.values
    n_samples = min(100, len(df1))
    background_summary = shap.kmeans(background, n_samples)
    explainer = shap.KernelExplainer(model_predict, background_summary)
    shap_values = explainer.shap_values(df1[:n_samples])

    K = min(10, df1.shape[1])
    importance = np.abs(shap_values).mean(0)
    top_idx = np.argsort(importance)[-K:][::-1]
    df_top = df1[:n_samples].iloc[:, top_idx]
    shap_top = shap_values[:, top_idx]

    fig, ax = plt.subplots()
    shap.summary_plot(shap_top, df_top, show=False)
    fig.savefig(shap_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def create_param_grid(df, numeric_samples=10):
    """
    从数据帧创建参数网格
    
    参数:
        df: 输入数据帧
        numeric_samples: 浮点数列采样数量
    
    返回:
        param_grid: 字典，键是列名，值是参数列表
    """
    param_grid = {}
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            # 浮点数列：等间隔采样
            min_val, max_val = df[col].min(), df[col].max()
            param_grid[col] = np.linspace(min_val, max_val, numeric_samples).tolist()
        else:
            # 类型/分类列：枚举所有唯一值
            param_grid[col] = df[col].unique().tolist()
    
    return param_grid

def grid_search_df(df, numeric_samples=10):
    """
    执行网格搜索，返回所有参数组合的数据帧
    
    参数:
        df: 输入数据帧
        numeric_samples: 浮点数列采样数量
    
    返回:
        包含所有参数组合的新数据帧
    """
    # 1. 创建参数网格
    param_grid = create_param_grid(df, numeric_samples)
    
    # 2. 生成所有组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    # 使用 itertools.product 生成笛卡尔积
    combinations = list(product(*values))
    
    # 3. 转换为数据帧
    result_df = pd.DataFrame(combinations, columns=keys)
    
    return result_df