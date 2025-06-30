import pandas as pd
import os


def save_prompt_to_txt(prompt_text, filename='Prompt/House/generated_prompt_geo.txt'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(prompt_text)
    print(f"\nPrompt 已保存到 {filename}")


def generate_geo_ocean_prompt(df):
    num_rows = len(df)
    lon_bins = [-124.35, -122, -120, -118, -116, -114.31]
    lat_bins = [32.54, 34, 36, 38, 40, 41.95]
    df['lon_bin'], lon_bin_edges = pd.cut(df['longitude'], bins=lon_bins, include_lowest=True, retbins=True)
    df['lat_bin'], lat_bin_edges = pd.cut(df['latitude'], bins=lat_bins, include_lowest=True, retbins=True)


    def make_geo_bin_label(lon_interval, lat_interval):
        if pd.isna(lon_interval) or pd.isna(lat_interval):
            return "NaN"
        else:
            lon_start, lon_end = lon_interval.left, lon_interval.right
            lat_start, lat_end = lat_interval.left, lat_interval.right
            return f"{lon_start}-{lon_end}_{lat_start}-{lat_end}"

    df['geo_bin'] = df.apply(lambda row: make_geo_bin_label(row['lon_bin'], row['lat_bin']), axis=1)


    prompt = [
        f"请你生成 {num_rows} 条记录补充在原数据集后面，生成字段包括 ocean_proximity，并保持以下类别分布结构。\n",
        "数据生成过程应遵循以下关键流程：\n",
        "1️.首先根据目标类别分布，生成各列数据，确保每一列类别分布严格符合给定目标比例，允许单个类别占比在目标值 ±0.1% 范围内浮动；\n",
        "2️.在保证各列类别分布基本不变的前提下，调整变量之间的联合分布结构；\n",
        "3️.在联合分布调整过程中，允许通过构建条件概率分布并采用顺序采样方式，逐步生成整行数据，以提高联合分布的合理性和拟合能力，拟合误差 <2%，优先保证主类别准确；\n",
        "4️.对生成数据施加边界限制，确保所有类别符合定义，且数据中不得包含缺失值（NaN）；\n",
        "5️.允许通过多轮迭代对数据进行调整优化，调整过程中不得重新采样单变量分布；\n",
        "6️.最终生成的数据应保持所有类别限定在指定集合内，不能出现未定义类别，且不得包含缺失值（NaN）。\n",
        "7️.最终将 geo_bin 列删除，将生成的 ocean_proximity 列按行对齐补充到原数据集中，作为新增字段 ocean_proximity，保持原数据行数不变，不得新增行或改变原数据行顺序。\n"
    ]


    prompt.append("#### 变量 `ocean_proximity` 的类别分布：")
    ocean_counts = df['ocean_proximity'].value_counts(normalize=True) * 100
    for cat, percent in ocean_counts.items():
        prompt.append(f"- {round(percent, 3)}% 的数据类别为 `{cat}`；")
    prompt.append("所有生成的数据，必须限定在上述类别集合内，且类别分布应严格匹配。\n")


    prompt.append("#### 变量 `geo_bin` 的类别分布：")

    prompt.append("geo_bin 字段格式为 \"longitude_min--longitude_max_latitude_min-latitude_max\"，表示对应记录所属的经纬度区间（地理分箱），其中：")
    prompt.append("- 前半段 longitude_min--longitude_max 为longitude区间")
    prompt.append("- 后半段 latitude_min-latitude_max 为latitude区间")
    prompt.append("例如 geo_bin = \"-116.0--114.31_32.539-34.0\" 表示该记录位于longitude -116.0 到 -114.31，latitude 32.539 到 34.0 区间内。")
    prompt.append("geo_bin 用于划分空间分区，指导 ocean_proximity 的条件采样，不能作为最终数据字段保留。\n")


    total_rows = len(df)
    geo_bins = df['geo_bin'].unique()
    for geo_bin in sorted(geo_bins):
        count = (df['geo_bin'] == geo_bin).sum()
        percent = count / total_rows * 100
        prompt.append(f"- {round(percent, 3)}% 的数据类别为 `{geo_bin}`；")
    prompt.append("所有生成的数据，必须限定在上述 geo_bin 集合内，且 geo_bin 应根据经纬度划分合理。\n")


    prompt.append("#### 条件概率分布 P(`ocean_proximity` | `geo_bin`)：\n")
    crosstab = pd.crosstab(df['geo_bin'], df['ocean_proximity'], normalize='index') * 100

    for geo_bin in crosstab.index:
        prompt.append(f"- `geo_bin` = `{geo_bin}` 时：")
        row = crosstab.loc[geo_bin]
        normalized_row = row / row.sum() * 100
        sorted_row = normalized_row.sort_values(ascending=False)
        for ocean_proximity, prob_percent in sorted_row.items():
            prompt.append(f"    - P(`ocean_proximity` = `{ocean_proximity}` | `geo_bin` = `{geo_bin}`) ≈ {prob_percent:.4f}%")
        prompt.append("")


    prompt_text = "\n".join(prompt)
    print(prompt_text)
    save_prompt_to_txt(prompt_text)


df = pd.read_csv("Data/house.csv")
generate_geo_ocean_prompt(df)
