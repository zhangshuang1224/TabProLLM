import pandas as pd
import os


def save_prompt_to_txt(prompt_text, filename='Prompt/Abalone/generated_prompt_X1.txt'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(prompt_text)
    print(f"\nPrompt 已保存到 {filename}")


def generate_x6group_x2_prompt(df):
    num_rows = len(df)
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.5]
    labels = ['Group_1', 'Group_2', 'Group_3', 'Group_4', 'Group_5']
    df['X6_group'] = pd.cut(df['X6'], bins=bins, labels=labels)
    df['X2_new'] = df['X6_group'].map(lambda x: f"X2_{labels.index(x)+1}" if pd.notnull(x) else 'Unknown')
    prompt = [
        f"请你生成 {num_rows} 条记录补充在原数据集后面，生成字段包括 X2_new，并保持以下类别分布结构。\n",
        "数据生成过程应遵循以下关键流程：\n",
        "1️.首先根据目标类别分布，生成各列数据，确保每一列类别分布严格符合给定目标比例，允许单个类别占比在目标值 ±0.1% 范围内浮动；\n",
        "2️.在保证各列类别分布基本不变的前提下，调整变量之间的联合分布结构；\n",
        "3️.在联合分布调整过程中，允许通过构建条件概率分布并采用顺序采样方式，逐步生成整行数据，以提高联合分布的合理性和拟合能力，拟合误差 <2%，优先保证主类别准确；\n",
        "4️.对生成数据施加边界限制，确保所有类别符合定义，且数据中不得包含缺失值（NaN）；\n",
        "5️.允许通过多轮迭代对数据进行调整优化，调整过程中不得重新采样单变量分布；\n",
        "6️.最终生成的数据应保持所有类别限定在指定集合内，不能出现未定义类别，且不得包含缺失值（NaN）。\n",
        "7.最终将 X6_group 列删除，将生成的 X2 列按行对齐补充到原数据集中，作为新增字段 X2，保持原数据行数不变（即对原数据中的每一行，根据其 X6 映射 X6_group 后采样对应的 X2，保证一一对应补充到原行），不得新增行或改变原数据行顺序。\n"
    ]

    prompt.append("#### 变量 `X6_group` 的类别分布：")
    total_rows = len(df)
    for i in range(len(labels)):
        range_start = bins[i]
        range_end = bins[i+1]
        range_label = f"{range_start}-{range_end}"
        count = ((df['X6'] >= range_start) & (df['X6'] < range_end)).sum()
        percent = count / total_rows * 100
        prompt.append(f"- {round(percent, 3)}% 的数据类别为 `{range_label}`；")
    prompt.append("所有生成的数据，必须限定在上述类别集合内，且类别分布应严格匹配。\n")


    prompt.append("#### 变量 `X1` 的类别分布：")
    x1_counts = df['X1'].value_counts(normalize=True) * 100
    for cat, percent in x1_counts.items():
        prompt.append(f"- {round(percent, 3)}% 的数据类别为 `{cat}`；")
    prompt.append("所有生成的数据，必须限定在上述类别集合内，且类别分布应严格匹配。\n")


    prompt.append("#### 条件概率分布 P(`X1` | `X6_group`)：\n")
    crosstab = pd.crosstab(df['X6_group'], df['X1'], normalize='index') * 100

    for x6_group in crosstab.index:

        i = labels.index(x6_group)
        range_start = bins[i]
        range_end = bins[i+1]
        range_label = f"{range_start}-{range_end}"

        prompt.append(f"- `X6_group` = `{range_label}` 时：")
        row = crosstab.loc[x6_group]
        normalized_row = row / row.sum() * 100
        sorted_row = normalized_row.sort_values(ascending=False)
        for x1, prob_percent in sorted_row.items():
            prompt.append(f"    - P(`X1` = `{x1}` | `X6_group` = `{range_label}`) ≈ {prob_percent:.4f}%")
        prompt.append("")


    prompt_text = "\n".join(prompt)
    print(prompt_text)
    save_prompt_to_txt(prompt_text)


df = pd.read_excel("Data/abalone.xlsx")
generate_x6group_x2_prompt(df)
