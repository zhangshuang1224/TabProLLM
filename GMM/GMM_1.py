import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')

def detect_decimal_precision(series):
    if pd.api.types.is_integer_dtype(series):
        return 0
    sample = series.dropna().astype(str)
    if sample.empty:
        return 2
    try:
        decimals = sample.apply(lambda x: len(x.split(".")[1]) if "." in x else 0)
        return int(decimals.mode()[0])
    except:
        return 2

def generate_multicolumn_gmm_prompt_and_plot(df, n_components=10):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_rows = len(df)
    num_cols = len(numeric_cols)

    header = [
        f"请你生成 {num_rows} 条记录的数据集，共 {num_cols} 列。每列数据需严格满足我给出的混合正态分布结构与数值范围。此外，这 {num_cols} 列数据必须一次性联合生成，以确保它们之间的皮尔逊相关系数矩阵尽可能接近我指定的目标矩阵。\n",
        "数据生成过程应遵循以下关键流程：\n",
        "1.构建目标皮尔逊相关系数矩阵，作为变量间相关性的结构基础；\n",
        "2.构造与该相关矩阵一致的协方差矩阵，并从中采样联合正态分布数据；\n",
        "3.将联合正态采样结果的每一列，通过匹配排序位置，赋值为对应目标混合正态分布中的数值，从而实现边缘分布转换；\n",
        "4.通过按照联合正态变量中的排序顺序，将目标边缘分布的数值逐一匹配赋值，从而在实现目标分布形状的同时尽量保留变量之间原有的相关性结构，逐步逼近目标相关矩阵；\n",
        "5.对生成数据施加边界限制，确保所有值严格落在指定范围内；\n",
        "6.如有必要，重复采样与调整过程，直至生成数据同时满足边缘分布和相关性结构的要求；\n",
        "7.生成的数据不能为NaN；\n",
        "8.如果生成的皮尔逊系数和原皮尔逊相关系数差异超过0.1，你需要重新生成。\n"
    ]

    prompt = header

    for col in numeric_cols:
        data = df[col].dropna().values.reshape(-1, 1)
        decimal_places = detect_decimal_precision(df[col])


        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)


        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(data_scaled)

        weights = gmm.weights_
        means_scaled = gmm.means_.flatten()
        stds_scaled = np.sqrt(gmm.covariances_).flatten()


        means = scaler.inverse_transform(means_scaled.reshape(-1, 1)).flatten()
        stds = stds_scaled * scaler.scale_[0]


        sorted_idx = np.argsort(means)
        sorted_weights = weights[sorted_idx]
        sorted_means = means[sorted_idx]
        sorted_stds = stds[sorted_idx]


        prompt.append(f"{col}：")
        for i in range(n_components):
            percent = round(sorted_weights[i] * 100, 3)
            mu = round(sorted_means[i], decimal_places)
            sigma = round(sorted_stds[i], decimal_places)
            prompt.append(
                f"- {percent:.3f}% 的数据来自均值为 {mu:.3f}，标准差为 {sigma:.3f} 的正态分布；")

        lower_bound = round(data.min(), decimal_places)
        upper_bound = round(data.max(), decimal_places)

        if decimal_places == 0:
            prompt.append(f"\n所有生成的数据，限定在 {int(lower_bound)} 到 {int(upper_bound)} 之间，为整数。\n")
        else:
            prompt.append(
                f"\n所有生成的数据，限定在 {lower_bound} 到 {upper_bound} 之间，保留 {decimal_places} 位小数。\n")


        labels = gmm.predict(data_scaled)
        label_map = {old: new for new, old in enumerate(sorted_idx)}
        sorted_labels = np.array([label_map[l] for l in labels])

        plt.figure(figsize=(12, 6))
        for i in range(n_components):
            comp_data_scaled = data_scaled[sorted_labels == i].flatten()
            comp_data = scaler.inverse_transform(comp_data_scaled.reshape(-1, 1)).flatten()
            sns.histplot(comp_data, bins=20, kde=True, stat='density', alpha=0.5, label=f'Component {i + 1}')
        sns.kdeplot(data.flatten(), color='black', linewidth=2, label='Overall KDE')
        plt.title(f'GMM Segmentation of {col} (Sorted by Mean)')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    prompt.append("各列之间的皮尔逊相关系数如下：\n")
    corr_matrix = df[numeric_cols].corr()
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            var1 = numeric_cols[i]
            var2 = numeric_cols[j]
            corr_value = corr_matrix.loc[var1, var2]
            prompt.append(f"- {var1} 和 {var2} 的皮尔逊相关系数为 {corr_value:.2f}；")

    prompt_text = "\n".join(prompt)
    print(prompt_text)
    basename = os.path.splitext(os.path.basename(data_file))[0]
    if basename == "Real estate valuation data set":
        basename = "Real Estate"
    save_dir = os.path.join("../Prompt", basename)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{basename}_gmm_prompt.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)



data_file = "../Data/Real estate valuation data set.csv"
ext = os.path.splitext(data_file)[1].lower()

if ext == ".csv":
    df = pd.read_csv(data_file)
elif ext in [".xlsx", ".xls"]:
    df = pd.read_excel(data_file)
else:
    raise ValueError(f"不支持的文件类型: {ext}")
generate_multicolumn_gmm_prompt_and_plot(df, n_components=10)

