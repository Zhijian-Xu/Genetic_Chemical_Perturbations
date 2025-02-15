import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df1 = pd.read_csv('./data/GSE92742_information.txt', sep='\t', header=None)
df2 = pd.read_csv('./data/GSE92742_expression_profile.txt', sep='\t', header=None, index_col=0)

parameters = [
    (0.01, 1, 12, 24, 'dose0.01uM-1uM_time24', './result/GSE92742_compound_dose0.01uM-1uM_time24.txt'),
    (0.01, 1, 0, 6, 'dose0.01uM-1uM_time6', './result/GSE92742_compound_dose0.01uM-1uM_time6.txt'),
    (1, 100, 12, 24, 'dose1uM-100uM_time24', './result/GSE92742_compound_dose1uM-100uM_time24.txt'),
    (1, 100, 0, 6, 'dose1uM-100uM_time6', './result/GSE92742_compound_dose1uM-100uM_time6.txt')
]

for dose_min, dose_max, time_min, time_max, cosine_label, output_file in parameters:
    filtered_df1 = df1[(df1[4] > dose_min) & (df1[4] <= dose_max) & 
                       (df1[6] > time_min) & (df1[6] <= time_max) & (df1[8].isin(['trt_cp']))]
    grouped = filtered_df1.groupby([1, 3, 8, 13])
    results = []
    for name, group in grouped:
        samples = group[0].values
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                sample1, sample2 = samples[i], samples[j]
                if sample1 in df2.index and sample2 in df2.index:
                    expr1, expr2 = df2.loc[sample1], df2.loc[sample2]
                    similarity_matrix = cosine_similarity([expr1], [expr2])
                    cosine = round(similarity_matrix[0][0], 4)
                    results.append([sample1, sample2, cosine])

    result_df = pd.DataFrame(results, columns=['Sample1', 'Sample2', cosine_label])
    result_df.to_csv(output_file, index=False, sep='\t')
