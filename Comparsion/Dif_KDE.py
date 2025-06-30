import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import math
import os


matplotlib.use('TkAgg')


real_root = r'D:\LLM_data\Data'
synthetic_root = r'D:\LLM_data\Generated_Data'
output_root = 'Dif_KDE_Output'

os.makedirs(output_root, exist_ok=True)


subdir_map = {
    'abalone': 'Abalone',
    'adult': 'Adult',
    'diabete': 'Diabete',
    'house': 'House',
    'iris': 'Iris',
    'heart': 'Heart',
    'real estate valuation data set': 'Real Estate',
    'winequality-red': 'Wine Quality'
}


prefix_map = {
    'real estate valuation data set': 'Real',
    'winequality-red': 'Wine'
}

# 遍历真实数据文件
for subdir, dirs, files in os.walk(real_root):
    for file in files:
        file_lower = file.lower().strip()

        if not (file_lower.endswith('.csv') or file_lower.endswith('.xlsx')):
            continue

        file_path_real = os.path.join(subdir, file)

        try:
            if file_lower.endswith('.csv'):
                df_real = pd.read_csv(file_path_real)
            else:
                df_real = pd.read_excel(file_path_real)
        except Exception as e:
            print(f'Warning: Could not read real file {file_path_real}. Error: {e}')
            continue

        base_filename = os.path.splitext(file)[0]
        base_filename_lower = base_filename.lower().strip()
        base_filename_capital = base_filename[0].upper() + base_filename[1:]

        # 获取目标生成目录名
        target_subdir = subdir_map.get(base_filename_lower, '')
        if not target_subdir:
            print(f'Warning: No subdir mapping found for {file_path_real}. Skipping.')
            continue

        synthetic_candidate_dir = os.path.join(synthetic_root, target_subdir)
        if not os.path.exists(synthetic_candidate_dir):
            print(f'Warning: Synthetic dir {synthetic_candidate_dir} not found. Skipping.')
            continue

        prefix = f"Generated_{prefix_map.get(base_filename_lower, base_filename_capital)}"

        synthetic_found = False
        for syn_file in os.listdir(synthetic_candidate_dir):
            if syn_file.startswith(prefix) and syn_file.lower().endswith('.csv'):
                synthetic_candidate_path = os.path.join(synthetic_candidate_dir, syn_file)
                synthetic_found = True

                try:
                    df_synthetic = pd.read_csv(synthetic_candidate_path)
                except Exception as e:
                    print(f'Warning: Could not read synthetic file {synthetic_candidate_path}. Error: {e}')
                    continue

                numeric_features_real = df_real.select_dtypes(include=['number']).columns
                numeric_features_synthetic = df_synthetic.select_dtypes(include=['number']).columns
                numeric_features = list(set(numeric_features_real) & set(numeric_features_synthetic))

                if len(numeric_features) == 0:
                    print(f'No numeric features found in common for {file_path_real} and {synthetic_candidate_path}. Skipping.')
                    continue

                n = len(numeric_features)
                cols = 3
                rows = math.ceil(n / cols)

                fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 4))
                axes = axes.flatten()

                for i, feature in enumerate(numeric_features):
                    try:
                        sns.kdeplot(df_real[feature].dropna(), label='Real', linewidth=2, ax=axes[i])
                        sns.kdeplot(df_synthetic[feature].dropna(), label='Synthetic', linewidth=2, ax=axes[i])
                        axes[i].set_title(f'KDE Compare: {feature}')
                        axes[i].set_xlabel(feature)
                        axes[i].set_ylabel('Density')
                        axes[i].legend()
                    except Exception as e:
                        print(f'Warning: Could not plot KDE for feature {feature}. Error: {e}')
                        axes[i].set_visible(False)

                for j in range(i + 1, len(axes)):
                    fig.delaxes(axes[j])

                plt.tight_layout()

                output_subdir = os.path.join(output_root, target_subdir)
                os.makedirs(output_subdir, exist_ok=True)

                compare_path = os.path.join(output_subdir, f'{base_filename_capital}_vs_{os.path.splitext(syn_file)[0]}_KDE_Compare.png')
                plt.savefig(compare_path, dpi=300)
                plt.close()

                print(f'KDE Compare saved: {compare_path}')

        if not synthetic_found:
            print(f'Warning: No matching synthetic file for {file_path_real} in {synthetic_candidate_dir}. Skipping.')
