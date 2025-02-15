import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
def calculate_cosine_similarity_shRNA(df1_path, df2_path, output_path, different_shRNA=True):
    df1 = pd.read_csv(df1_path, sep='\t', header=None)
    df2 = pd.read_csv(df2_path, sep='\t', header=None, index_col=0)

    volume_min, volume_max = 0, 10
    time_min, time_max = 96, 144
    filtered_df1 = df1[(df1[4] >= volume_min) & (df1[4] <= volume_max) & 
                        (df1[6] >= time_min) & (df1[6] <= time_max) & (df1[8].isin(['trt_sh', 'trt_sh.cgs']))]
    if different_shRNA:
        grouped = filtered_df1.groupby([2, 3])
    else:
        grouped = filtered_df1.groupby([1, 2, 3])

    results = []
    for name, group in grouped:
        samples = group[0].values
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                sample1, sample2 = samples[i], samples[j]
                if different_shRNA and group.iloc[i, 1] == group.iloc[j, 1]:
                    continue
                if sample1 in df2.index and sample2 in df2.index:
                    expr1, expr2 = df2.loc[sample1], df2.loc[sample2]
                    cosine_similarity_value = round(cosine_similarity([expr1], [expr2])[0][0], 4)
                    results.append([sample1, sample2, cosine_similarity_value])
    result_df = pd.DataFrame(results, columns=['Sample1', 'Sample2', 'Cosine'])
    result_df.to_csv(output_path, index=False, sep='\t')

calculate_cosine_similarity_shRNA(
    './data/GSE92742_information.txt',
    './data/GSE92742_expression_profile.txt',
    './result/GSE92742_different_shRNA_96h-144h.txt',
    different_shRNA=True
)
calculate_cosine_similarity_shRNA(
    './data/GSE92742_information.txt',
    './data/GSE92742_expression_profile.txt',
    './result/GSE92742_same_shRNA_96h-144h.txt',
    different_shRNA=False
)
