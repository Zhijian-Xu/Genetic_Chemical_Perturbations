import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


df1 = pd.read_csv('./data/CMap2020_knockdown_information.txt', sep='\t', header=None)
df2 = pd.read_csv('./data/CMap2020_knockdown_profile.txt', sep='\t', header=None, index_col=0)


volume_min, volume_max = 0, 10
time_min, time_max = 72, 144

filtered_df1 = df1[(df1[4] > volume_min) & (df1[4] <= volume_max) & (df1[6] > time_min) & (df1[6] <= time_max)]
grouped = filtered_df1.groupby([1, 2, 3])

results = []

for name, group in grouped:
    samples = group[0].values 
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            sample1, sample2 = samples[i], samples[j]
            if sample1 in df2.index and sample2 in df2.index:
                expr1, expr2 = df2.loc[sample1], df2.loc[sample2]
                condition_pos = (expr1 > 0) & (expr2 > 0)
                condition_neg = (expr1 < 0) & (expr2 < 0)
                condition_zero = (expr1 == 0) & (expr2 == 0)
                similar_elements = sum(condition_pos) + sum(condition_neg) + sum(condition_zero)
                jaccard_simiarity = round(similar_elements / len(expr1), 4) 

                pearson_similarity = round(pearsonr(expr1, expr2)[0], 4)
                spearman_similarity = round(spearmanr(expr1, expr2)[0], 4)
                cosine_similarity_value = round(cosine_similarity([expr1], [expr2])[0][0], 4)
                results.append([sample1, sample2, jaccard_simiarity, pearson_similarity, spearman_similarity, cosine_similarity_value])


result_df = pd.DataFrame(results, columns=['Sample1', 'Sample2', 'Jaccard', 'Pearson', 'Spearman', 'Cosine'])

result_df.to_csv('./result/CMap2020-sameshRNA.txt', index=False, sep='\t')
