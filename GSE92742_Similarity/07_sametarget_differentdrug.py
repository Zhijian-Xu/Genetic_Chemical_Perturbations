import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sametarget = pd.read_csv('./data/sametarget.txt', sep='\t', header=None, names=['drug', 'gene'])
drug_info = pd.read_csv('./data/GSE92742_compound_10uM_information.txt', sep='\t', header=None,
                        names=['drug_sample', 'drug', 'compound', 'cell_line', 'dose', 'unit', 'time', 'treatment', 
                               'value1', 'value2', 'value3', 'value4', 'batch', 'value5',  'value6','extended_info'])
cp_need = pd.read_csv('./data/GSE92742_compound_profiles.txt', sep='\t', header=None, 
                      names=['drug_sample'] + [f'expr{i}' for i in range(1, 979)])


grouped = sametarget.groupby('gene')['drug'].apply(list).reset_index()

results = []

for _, row in grouped.iterrows():
    gene = row['gene']
    drugs = row['drug']
    
    drug_records = drug_info[drug_info['drug'].isin(drugs)]
    for cell_line, group in drug_records.groupby('cell_line'):
        if len(group) > 1: 
            drug_samples = group['drug_sample'].tolist()
            drug_pairs = [(drug_samples[i], drug_samples[j]) 
                          for i in range(len(drug_samples)) 
                          for j in range(i + 1, len(drug_samples))]
            for sample1, sample2 in drug_pairs:
                expr1 = cp_need[cp_need['drug_sample'] == sample1].iloc[:, 1:].values.flatten()
                expr2 = cp_need[cp_need['drug_sample'] == sample2].iloc[:, 1:].values.flatten()
                
                if len(expr1) == len(expr2) and len(expr1) > 0:
                    similarity = round(cosine_similarity([expr1], [expr2])[0][0], 4)
                    results.append([gene, sample1, sample2, cell_line, similarity])

results_df = pd.DataFrame(results, columns=['target_gene', 'drug_sample1', 'drug_sample2', 'cell_line', 'cosine_similarity'])
results_df.to_csv('./result/GSE92742_sametargets-different-drug-10um.txt', index=False, sep='\t')