import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

def process_data(expression_file, information_file, target_file, condition_column, condition_values, target_column, output_file):
    """
    Process the dataset, compute similarity metrics, and save the results.
    Parameters:
        expression_file (str): Path to the expression profile file.
        information_file (str): Path to the information file.
        target_file (str): Path to the compound-target file.
        condition_column (int): The column index to filter in information_file.
        condition_values (list): The values to filter for in the condition_column.
        target_column (str): The name of the target gene column (Knockdown_gene or Overexpression_gene).
        output_file (str): Path to save the results.
    """
    # Load compound-target relationship
    compound_target = pd.read_csv(target_file, sep='\t')

    # Load compound information
    compound_information = pd.read_csv(
        information_file, sep='\t', usecols=[0, 1, 3], names=['drug_sample', 'Compound', 'cell_line']
    )

    # Load and filter gene perturbation information
    data_info = pd.read_csv(information_file, sep='\t', header=None)
    data_info_filtered = data_info[data_info.iloc[:, condition_column].isin(condition_values)]

    # Select relevant columns
    target_information = data_info_filtered.iloc[:, [0, 2, 3]]
    target_information.columns = ['target_sample', target_column, 'cell_line']

    # Load expression profiles
    compound_expression_profile = pd.read_csv(
        expression_file, sep='\t', names=['drug_sample'] + [f'expr{i}' for i in range(1, 979)]
    )

    target_expression_profile = pd.read_csv(
        expression_file, sep='\t', names=['target_sample'] + [f'expr{i}' for i in range(1, 979)]
    )

    # Merge datasets
    merged_data = (
        compound_target
        .merge(compound_information, on='Compound')
        .merge(target_information, left_on=[target_column, 'cell_line'], right_on=[target_column, 'cell_line'])
        .drop(columns=[target_column])  # Remove redundant column
    )

    # Set index for faster lookup
    compound_expression_profile.set_index('drug_sample', inplace=True)
    target_expression_profile.set_index('target_sample', inplace=True)

    # Function to compute similarity metrics
    def compute_similarity(row):
        if row['drug_sample'] not in compound_expression_profile.index or row['target_sample'] not in target_expression_profile.index:
            return None  # Skip missing samples

        # Extract expression values
        drug_expr = compound_expression_profile.loc[row['drug_sample']].values
        target_expr = target_expression_profile.loc[row['target_sample']].values

        # Ensure valid data
        if len(drug_expr) != len(target_expr) or len(drug_expr) == 0:
            return None

        # Compute Jaccard similarity based on positive/negative/zero matches
        condition_pos = (target_expr > 0) & (drug_expr > 0)
        condition_neg = (target_expr < 0) & (drug_expr < 0)
        condition_zero = (target_expr == 0) & (drug_expr == 0)
        jaccard_similarity = round((condition_pos.sum() + condition_neg.sum() + condition_zero.sum()) / len(target_expr), 4)

        # Compute Pearson, Spearman, and Cosine similarity
        pearson_similarity = round(pearsonr(drug_expr, target_expr)[0], 4)
        spearman_similarity = round(spearmanr(drug_expr, target_expr)[0], 4)
        cosine_sim = round(cosine_similarity([drug_expr], [target_expr])[0][0], 4)

        return [row['drug_sample'], row['target_sample'], jaccard_similarity, pearson_similarity, spearman_similarity, cosine_sim]

    # Apply function and remove null results
    results = merged_data.apply(compute_similarity, axis=1).dropna()
    results_df = pd.DataFrame(results.tolist(), columns=['drug-sample', 'target-sample', 'jaccard', 'pearson', 'spearman', 'cosine'])

    # Save results
    results_df.to_csv(output_file, index=False, sep='\t')


# Process Knockdown Data
process_data(
    expression_file='./data/GSE92742_expression_profile.txt',
    information_file='./data/GSE92742_information.txt',
    target_file='./data/GSE92742-742drug-324kd',
    condition_column=8,
    condition_values=['trt_sh', 'trt_sh.cgs'],
    target_column='Knockdown_gene',
    output_file='./result/GSE92742_inhibitors_knockdown_correlation.txt'
)
