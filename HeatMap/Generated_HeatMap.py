import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('TkAgg')


root_dir = r'D:\LLM_data\Generated_Data'


output_root = 'Heatmap_Output'

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

            corr_matrix = df[numeric_features].corr()

            sns.set(style="white")
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
            plt.title(f'Heatmap of Numeric Features\n{file}')

            relative_subdir = os.path.relpath(subdir, root_dir)
            output_subdir = os.path.join(output_root, relative_subdir)
            os.makedirs(output_subdir, exist_ok=True)

            base_filename = os.path.splitext(file)[0]
            png_path = os.path.join(output_subdir, f'{base_filename}_heatmap.png')
            plt.savefig(png_path, dpi=300)

            plt.close()
