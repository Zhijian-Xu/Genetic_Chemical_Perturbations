import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sh_472_target_1064drug = pd.read_csv('./data/472target_1064drug.txt', sep='\t', header=None, names=['gene', 'drug'])
data_1064_drug = pd.read_csv('./data/1064drug.txt', sep='\t', usecols=[4, 5, 12], header=None, names=['cell_line', 'drug', 'drug_sample'])
data_472_shtarget = pd.read_csv('./data/472target_knockdown.txt', sep='\t', usecols=[1, 9, 12], header=None, names=['cell_line', 'target_sample', 'gene'])
cp_need = pd.read_csv('./data/1064drug_profile.txt', sep='\t', header=None, names=['drug_sample'] + ['expr' + str(i) for i in range(1, 979)])
sh_need = pd.read_csv('./data/472target_profile.txt', sep='\t', header=None, names=['target_sample'] + ['expr' + str(i) for i in range(1, 979)])

merged_data = pd.merge(sh_472_target_1064drug, data_1064_drug, on='drug')
merged_data = pd.merge(merged_data, data_472_shtarget, on=['gene', 'cell_line'])

results = []
for index, row in merged_data.iterrows():
    drug_expr = cp_need[cp_need['drug_sample'] == row['drug_sample']].iloc[:, 1:].values.flatten()
    target_expr = sh_need[sh_need['target_sample'] == row['target_sample']].iloc[:, 1:].values.flatten()
    if len(drug_expr) == len(target_expr) and len(drug_expr) > 0:
        condition_pos = (drug_expr > 0) & (target_expr > 0)
        condition_neg = (drug_expr < 0) & (target_expr < 0)
        condition_zero = (drug_expr == 0) & (target_expr == 0)
        similar_elements = sum(condition_pos) + sum(condition_neg) + sum(condition_zero)
        jaccard_simiarity = round(similar_elements / len(drug_expr), 4)
        similarity_matrix = cosine_similarity([drug_expr], [target_expr])
        similarity = round(similarity_matrix[0][0], 4)
        pearson_similarity = round(pearsonr(drug_expr, target_expr)[0], 4)
        spearman_similarity = round(spearmanr(drug_expr, target_expr)[0], 4)
        results.append([row['drug_sample'], row['target_sample'], jaccard_simiarity, pearson_similarity, spearman_similarity, similarity])


results_df = pd.DataFrame(results, columns=['drug-sample', 'target-sample', 'jaccard', 'pearson', 'spearmon', 'cosine'])
results_df.to_csv('./result/CMap2020_inhibitors_knockdown_correlation.txt', index=False, sep='\t')
