import pandas as pd
import numpy as np

# Load data
train_df = pd.read_csv("Données/lamme_training.csv")
pred_df = pd.read_csv("Données/lamme_prediction.csv")

def transform_pair_metrics(df):
    """Transform per-gene metrics into pairwise relational features."""
    df_out = df.copy()
    
    # Transform continuous variables: compute asymmetry and mean
    continuous_pairs = [
        ('TE_Coverage_G1', 'TE_Coverage_G2', 'TE_Coverage'),
        ('TE_Number_G1', 'TE_Number_G2', 'TE_Number'),
        ('PPI_Interactant_G1', 'PPI_Interactant_G2', 'PPI_Interactant'),
        ('Pfam_G1', 'Pfam_G2', 'Pfam')
    ]
    
    for c1, c2, base in continuous_pairs:
        if c1 in df_out.columns and c2 in df_out.columns:
            df_out[f'{base}_Asymmetry'] = np.abs(df_out[c1] - df_out[c2])
            df_out[f'{base}_Mean'] = (df_out[c1] + df_out[c2]) / 2
            df_out = df_out.drop(columns=[c1, c2])

    # TE_Density with log smoothing
    if 'TE_Density_G1' in df_out.columns:
        den_g1 = np.log1p(df_out['TE_Density_G1'])
        den_g2 = np.log1p(df_out['TE_Density_G2'])
        df_out['TE_Density_Asymmetry'] = np.abs(den_g1 - den_g2)
        df_out['TE_Density_Mean'] = (den_g1 + den_g2) / 2
        df_out = df_out.drop(columns=['TE_Density_G1', 'TE_Density_G2'])

    # Transform categorical variables: create matching indicators
    cat_cols = df_out.select_dtypes(exclude=[np.number]).columns
    for c1 in [c for c in cat_cols if c.endswith('_G1')]:
        base_name = c1.replace('_G1', '')
        c2 = base_name + '_G2'
        if c2 in df_out.columns:
            df_out[base_name + '_Match'] = (df_out[c1] == df_out[c2]).astype(int)
            df_out = df_out.drop(columns=[c1, c2])
            
    return df_out

# Apply transformations
train_df = transform_pair_metrics(train_df)
pred_df = transform_pair_metrics(pred_df)

def clean_dataset(df, is_train=True, train_medians=None):
    """Impute missing values using training medians."""
    df_clean = df.copy()

    # Zero-fill similarity indices
    for col in ["PPI_Jaccard", "Pfam_Jaccard"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)

    # Flag missing expression data
    if "Distance" in df_clean.columns:
        df_clean["Expression_Data_Missing"] = df_clean["Distance"].isna().astype(int)

    # Calculate medians from training set
    if is_train:
        train_medians = {}
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if col not in ["Gene1", "Gene2", "Classification", "Expression_Data_Missing"]:
                train_medians[col] = df_clean[col].median()

    # Apply imputation
    for col, val in train_medians.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(val)
    
    for col in df_clean.select_dtypes(exclude=[np.number]).columns:
        if col not in ["Gene1", "Gene2"]:
            df_clean[col] = df_clean[col].fillna("Missing")

    return df_clean, train_medians

# Clean data
train_df_clean, medians = clean_dataset(train_df, is_train=True)
pred_df_clean, _ = clean_dataset(pred_df, is_train=False, train_medians=medians)

# Extract features and target
X_train = train_df_clean.drop(columns=['Gene1', 'Gene2', 'Classification'])
y_train = train_df_clean['Classification']
X_pred = pred_df_clean.drop(columns=['Gene1', 'Gene2'])

# One-hot encode and align feature spaces
X_train = pd.get_dummies(X_train, drop_first=False)
X_pred = pd.get_dummies(X_pred, drop_first=False)
Y_train = y_train.astype(int)  # Ensure target is integer for classification
X_train, X_pred = X_train.align(X_pred, join='left', axis=1, fill_value=0)

# Save cleaned datasets
X_train.to_csv("Données/X_train_clean.csv", index=False)
X_pred.to_csv("Données/X_pred_clean.csv", index=False)
Y_train.to_csv("Données/Y_train_clean.csv", index=False)