import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

info_file_path = './data/GSE92742_information.txt'
expression_file_path = './data/GSE92742_expression_profile.txt'

info_data = pd.read_csv(info_file_path, sep="\t")

filtered_kd = info_data[info_data.iloc[:, 8].str.contains('trt_sh|trt_sh\.cgs', na=False)]
filtered_oe = info_data[info_data.iloc[:, 8] == 'trt_oe']

knockdown_df = filtered_kd.iloc[:, [3, 0, 2]].copy()
knockdown_df.columns = ['CellLine', 'SampleNameKD', 'GeneNameKD']

overexpression_df = filtered_oe.iloc[:, [3, 0, 2]].copy()
overexpression_df.columns = ['CellLine', 'SampleNameOE', 'GeneNameOE']

print("Knockdown DataFrame:")
print(knockdown_df.head())
print("\nOverexpression DataFrame:")
print(overexpression_df.head())

expression_df = pd.read_csv(expression_file_path, sep="\t")
expression_df.columns = ['SampleName'] + [f'Gene_{i}' for i in range(1, expression_df.shape[1])]

results = []
for _, row in pd.merge(knockdown_df, overexpression_df, left_on=['CellLine', 'GeneNameKD'], right_on=['CellLine', 'GeneNameOE']).iterrows():
    sample_name_kd = row['SampleNameKD']
    sample_name_oe = row['SampleNameOE']
    gene_name = row['GeneNameKD']

    expression_kd = expression_df[expression_df['SampleName'] == sample_name_kd].iloc[:, 1:].values
    expression_oe = expression_df[expression_df['SampleName'] == sample_name_oe].iloc[:, 1:].values

    if expression_kd.size == 0 or expression_oe.size == 0:
        continue

    similarity_matrix = cosine_similarity(expression_kd, expression_oe)
    similarity = similarity_matrix[0][0]

    results.append([row['CellLine'], sample_name_kd, sample_name_oe, gene_name, round(similarity, 4)])

results_df = pd.DataFrame(results, columns=['CellLine', 'KnockdownSample', 'OverexpressionSample', 'GeneName', 'CosineSimilarity'])

output_file = './result/GSE92742_samegene_knockdown_overexpression.txt'
results_df.to_csv(output_file, sep="\t", index=False)

print(f"Results saved to {output_file}")
