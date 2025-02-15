import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(df1_path, df2_path, dose_range, time_range, output_path):
    df1 = pd.read_csv(df1_path, sep='\t', header=None)
    df2 = pd.read_csv(df2_path, sep='\t', header=None, index_col=0)

    dose_min, dose_max = dose_range
    time_min, time_max = time_range
    filtered_df1 = df1[(df1[0] > dose_min) & (df1[0] <= dose_max) & 
                       (df1[3] > time_min) & (df1[3] <= time_max)]

    grouped = filtered_df1.groupby([4, 5])
    valid_samples = [group for name, group in grouped if len(group) > 1]

    results = []
    for group in valid_samples:
        samples = group[12].values
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                sample1, sample2 = samples[i], samples[j]
                if sample1 in df2.index and sample2 in df2.index:
                    expr1 = df2.loc[sample1].values.reshape(1, -1)
                    expr2 = df2.loc[sample2].values.reshape(1, -1)
                    cosine_similarity_value = round(cosine_similarity(expr1, expr2)[0][0], 4)
                    results.append([sample1, sample2, cosine_similarity_value])

    result_df = pd.DataFrame(results, columns=['Sample1', 'Sample2', 'Cosine'])
    result_df.to_csv(output_path, index=False, sep='\t')

df1_path = './data/CMap2020_compound_information.txt'
df2_path = './data/CMap2020_compound_profile.txt'

params = [
    ((0.01, 1), (0, 6), './result/CMap2020_compound_dose0.01um-1um_time6.txt'),
    ((0.01, 1), (12, 24), './result/CMap2020_compound_dose0.01um-1um_time24.txt'),
    ((1, 100), (0, 6), './result/CMap2020_compound_dose1um-100um_time6.txt'),
    ((1, 100), (12, 24), './result/CMap2020_compound_dose1um-100um_time24.txt')
]

for dose_range, time_range, output_path in params:
    calculate_cosine_similarity(df1_path, df2_path, dose_range, time_range, output_path)