import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(time_fixed, output_filename, volume_max=10):
    df1 = pd.read_csv('./data/GSE92742_information.txt', sep='\t', header=None)
    df2 = pd.read_csv('./data/GSE92742_expression_profile.txt', sep='\t', header=None, index_col=0)

    print("df1 shape:", df1.shape)
    print("df2 shape:", df2.shape)
    print(df1.head())

    df1[6] = df1[6].astype(str)
    df1[8] = df1[8].astype(str)
    df2.index = df2.index.astype(str)
    filtered_df1 = df1[(df1[6] == str(time_fixed)) & (df1[8].isin(['trt_oe']))]

    print("filtered_df1 shape:", filtered_df1.shape)

    if filtered_df1.empty:
        print(f"No data after filtering for time {time_fixed}h.")
        return

    grouped = filtered_df1.groupby([2, 3])
    results = []
    
    for _, group in grouped:
        samples = group.iloc[:, 0].values
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                sample1, sample2 = str(samples[i]), str(samples[j])
                if sample1 in df2.index and sample2 in df2.index:
                    expr1 = df2.loc[sample1].values.reshape(1, -1)
                    expr2 = df2.loc[sample2].values.reshape(1, -1)
                    cosine_similarity_value = round(cosine_similarity(expr1, expr2)[0][0], 4)
                    results.append([sample1, sample2, cosine_similarity_value])

    if not results:
        print(f"No cosine similarity calculated for time {time_fixed}h.")
        return

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    result_df = pd.DataFrame(results, columns=['Sample1', 'Sample2', f'GSE92742-Overexpression_{time_fixed}h'])
    result_df.to_csv(output_filename, index=False, sep='\t')
    print(f"Saved results to {output_filename}")

calculate_cosine_similarity(24, './result/GSE92742_Overexpression_24h_cosine.txt')
calculate_cosine_similarity(48, './result/GSE92742_Overexpression_48h_cosine.txt')
calculate_cosine_similarity(72, './result/GSE92742_Overexpression_72h_cosine.txt')
calculate_cosine_similarity(96, './result/GSE92742_Overexpression_96h_cosine.txt')
