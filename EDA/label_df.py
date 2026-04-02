import pandas as pd

# Read the feather file
df = pd.read_feather('data_andre.feather')

# Extract only products and labels
label_df = df[['products', 'cat_label', 'dep_label', 'sdep_label']].copy()
label_df.to_csv('product_labels.csv', index=False)
# Display the result
print(label_df.head())
print(label_df.shape)