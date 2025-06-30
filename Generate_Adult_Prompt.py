import pandas as pd
import os
from pathlib import Path

def save_prompt_lines(prompt_lines, filename):
    prompt_text = "\n".join(prompt_lines)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(prompt_text)
    print(f"Prompt saved to {filename}")

def save_prompt_text(prompt_text, filename='Prompt/Adult/native-country.txt'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(prompt_text)
    print(f"Prompt saved to {filename}")

def build_prompt_header(num_rows, target_field):
    return [
        f"请你生成 {num_rows} 条记录补充在原数据集后面，生成字段包括 {target_field}，并保持以下类别分布结构。\n",
        "数据生成过程应遵循以下关键流程：\n",
        "1️.首先根据目标类别分布，生成各列数据，确保每一列类别分布严格符合给定目标比例，允许单个类别占比在目标值 ±0.1% 范围内浮动；\n",
        "2️.在保证各列类别分布基本不变的前提下，调整变量之间的联合分布结构；\n",
        "3️.在联合分布调整过程中，允许通过构建条件概率分布并采用顺序采样方式，逐步生成整行数据，以提高联合分布的合理性和拟合能力，拟合误差 <2%，优先保证主类别准确；\n",
        "4️.对生成数据施加边界限制，确保所有类别符合定义，且数据中不得包含缺失值（NaN）；\n",
        "5️.允许通过多轮迭代对数据进行调整优化，调整过程中不得重新采样单变量分布；\n",
        "6️.最终生成的数据应保持所有类别限定在指定集合内，不能出现未定义类别，且不得包含缺失值（NaN）。\n"
    ]

def generate_relationship_prompt(df):
    num_rows = len(df)
    prompt = build_prompt_header(num_rows, "relationship")
    prompt.append("7.将生成的 relationship 列按行对齐补充到原数据集中，作为新增字段 relationship，保持原数据行数不变，不得新增行或改变原数据行顺序。\n")

    prompt.append("#### 变量 `marital-status` 的类别分布：")
    mar_dist = df['marital-status'].value_counts(normalize=True) * 100
    for cat, percent in mar_dist.items():
        prompt.append(f"- {round(percent, 3)}% 的数据类别为 `{cat}`；")
    prompt.append("所有生成的数据，必须限定在上述类别集合内，且类别分布应严格匹配。\n")

    prompt.append("\n#### 变量 `relationship` 的类别分布：")
    rel_dist = df['relationship'].value_counts(normalize=True) * 100
    for cat, percent in rel_dist.items():
        prompt.append(f"- {round(percent, 3)}% 的数据类别为 `{cat}`；")
    prompt.append("所有生成的数据，必须限定在上述类别集合内，且类别分布应严格匹配。\n")

    prompt.append("#### 条件概率分布 P(`relationship` | `marital-status`)：")
    crosstab = pd.crosstab(df['marital-status'], df['relationship'], normalize='index') * 100
    for mar in crosstab.index:
        prompt.append(f"- `marital-status` = `{mar}` 时：")
        row = crosstab.loc[mar]
        for relationship, prob in row.sort_values(ascending=False).items():
            prompt.append(f"    - P(`relationship` = `{relationship}` | `marital-status` = `{mar}`) ≈ {prob:.4f}%")
        prompt.append("")
    return prompt

def generate_marital_by_agegroup_prompt(df):
    num_rows = len(df)
    prompt = build_prompt_header(num_rows, "marital-status")
    prompt.append("7.最终将 age_group 列删除，将生成的 marital-status 列按行对齐补充到原数据集中，作为新增字段 marital-status，保持原数据行数不变，不得新增行或改变原数据行顺序。\n")

    prompt.append("#### marital-status 类别分布：")
    for k, v in df['marital-status'].value_counts(normalize=True).items():
        prompt.append(f"- {v * 100:.3f}% 的数据类别为 `{k}`；")
    prompt.append("")

    crosstab = pd.crosstab(df['age_group'], df['marital-status'], normalize='index') * 100
    prompt.append("#### 条件概率分布 P(`marital-status` | `age_group`)：")
    for age_group in crosstab.index:
        prompt.append(f"- `age_group` = `{age_group}` 时：")
        row = crosstab.loc[age_group]
        for val, prob in row.sort_values(ascending=False).items():
            prompt.append(f"    - P(`marital-status` = `{val}` | `age_group` = `{age_group}`) ≈ {prob:.4f}%")
        prompt.append("")
    return prompt

def generate_conditional_prompt(df, target_field, cond_field):
    num_rows = len(df)
    prompt = build_prompt_header(num_rows, target_field)

    if cond_field == "age_group":
        prompt.append(
            f"7.最终将 age_group 列删除，将生成的 {target_field} 列按行对齐补充到原数据集中，作为新增字段 {target_field}，保持原数据行数不变，不得新增行或改变原数据行顺序。\n"
        )
    else:
        prompt.append(
            f"7.将生成的 {target_field} 列按行对齐补充到原数据集中，作为新增字段 {target_field}，保持原数据行数不变，不得新增行或改变原数据行顺序。\n"
        )

    prompt.append(f"#### 变量 `{cond_field}` 的类别分布：")
    for cat, pct in df[cond_field].value_counts(normalize=True).items():
        prompt.append(f"- {pct * 100:.3f}% 的数据类别为 `{cat}`；")
    prompt.append("")

    prompt.append(f"#### 变量 `{target_field}` 的类别分布：")
    for cat, pct in df[target_field].value_counts(normalize=True).items():
        prompt.append(f"- {pct * 100:.3f}% 的数据类别为 `{cat}`；")
    prompt.append("")

    prompt.append(f"#### 条件概率分布 P(`{target_field}` | `{cond_field}`)：")
    crosstab = pd.crosstab(df[cond_field], df[target_field], normalize='index') * 100
    for cond_val in crosstab.index:
        prompt.append(f"- `{cond_field}` = `{cond_val}` 时：")
        row = crosstab.loc[cond_val]
        for val, prob in row.sort_values(ascending=False).items():
            prompt.append(f"    - P(`{target_field}` = `{val}` | `{cond_field}` = `{cond_val}`) ≈ {prob:.4f}%")
        prompt.append("")
    return prompt

def generate_all_prompts(df, output_dir='Prompt/Adult'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df['age_group'] = pd.cut(df['age'], bins=[17, 25, 35, 45, 55, 65, 90],
                             labels=['17-24', '25-34', '35-44', '45-54', '55-64', '65-90'])

    file = output_path / "relationship.txt"
    save_prompt_lines(generate_relationship_prompt(df), file)

    file = output_path / "marital-status.txt"
    save_prompt_lines(generate_marital_by_agegroup_prompt(df), file)

    mappings = [
        ("sex", "relationship"),
        ("occupation", "sex"),
        ("education", "age_group"),
        ("workclass", "age_group"),
        ("race", "native-country")
    ]

    for target, cond in mappings:
        file = output_path / f"{target}.txt"
        save_prompt_lines(generate_conditional_prompt(df, target, cond), file)

def generate_native_country_prompt(df):
    num_rows = len(df)
    prompt = [
        f"请你生成 {num_rows} 条记录补充在原数据集后面，生成字段包括 native-country，并保持以下类别分布结构。\n",
        "数据生成过程应遵循以下关键流程：\n",
        "1️.首先根据目标类别分布，生成各列数据，确保每一列类别分布严格符合给定目标比例\n",
        "2.对生成数据施加边界限制，确保所有类别符合定义，且数据中不得包含缺失值（NaN）\n",
        "3.允许通过多轮迭代对数据进行调整优化\n",
        "4.最终生成的数据应保持所有类别限定在指定集合内，不能出现未定义类别，且不得包含缺失值（NaN）。\n",
        "5.将生成的 native-country 列按行对齐补充到原数据集中，作为新增字段 native-country，保持原数据行数不变，不得新增行或改变原数据行顺序。\n"
    ]

    prompt.append("#### 变量 `native-country` 的类别分布：")
    native_country_counts = df['native-country'].value_counts(normalize=True) * 100
    for cat, percent in native_country_counts.items():
        prompt.append(f"- {round(percent, 3)}% 的数据类别为 `{cat}`；")
    prompt.append("所有生成的数据，必须限定在上述类别集合内，且类别分布应严格匹配。\n")

    save_prompt_text("\n".join(prompt))


if __name__ == '__main__':
    df = pd.read_csv("Data/adult.csv")
    generate_all_prompts(df)
    generate_native_country_prompt(df)
