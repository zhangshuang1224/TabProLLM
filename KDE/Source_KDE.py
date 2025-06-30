import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import math
import os

matplotlib.use('TkAgg')

root_dir = r'D:\LLM_data\Data'
output_root = 'KDE'

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        file_lower = file.lower().strip()
        if file_lower.endswith('.csv') or file_lower.endswith('.xlsx'):
            file_path = os.path.join(subdir, file)

            try:
                if file_lower.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
            except Exception as e:
                continue

            numeric_features = df.select_dtypes(include=['number']).columns.tolist()

            if len(numeric_features) == 0:
                continue

            # 输出目录
            relative_subdir = os.path.relpath(subdir, root_dir)
            output_subdir = os.path.join(output_root, relative_subdir)
            os.makedirs(output_subdir, exist_ok=True)

            base_filename = os.path.splitext(file)[0]

            # 一张大图 grid 画所有 feature 的 KDE
            n = len(numeric_features)
            cols = 3
            rows = math.ceil(n / cols)

            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 4))
            axes = axes.flatten()

            for i, feature in enumerate(numeric_features):
                try:
                    sns.kdeplot(df[feature].dropna(), fill=True, linewidth=2, ax=axes[i])
                    axes[i].set_title(f'KDE of {feature}')
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel("Density")
                except Exception as e:
                    print(f'Warning: Could not plot KDE for feature {feature} in file {file_path}. Error: {e}')
                    axes[i].set_visible(False)


            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            kde_path = os.path.join(output_subdir, f'{base_filename}_KDE.png')
            plt.savefig(kde_path, dpi=300)
            plt.close()
