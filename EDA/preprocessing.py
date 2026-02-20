import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import pyarrow.feather as feather
from statsmodels.tsa.seasonal import STL
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings('ignore')


def preprocess_features(df: pd.DataFrame, 
                       fit_encoders: bool = True,
                       encoding_type: str = 'label',
                       encoders: Dict = None,
                       scalers: Dict = None,
                       onehot_categories: Dict = None,
                       use_log1p: bool = False) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
   
    df = df.copy()
    
    # Define feature groups
    categorical_cols = ['cat_label', 'sdep_label', 'dep_label', 'dmn_label']
    numerical_cols = ['value'] 
    promotional_cols =  [col for col in df.columns if col.startswith('promo_value_')]
    
    # Filter to columns that exist
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    promotional_cols = [col for col in promotional_cols if col in df.columns]

    if encoding_type not in ['label', 'onehot']:
        raise ValueError("encoding_type must be 'label' or 'onehot'")
    
    # Initialize containers for return values
    encoders_dict = {}
    scalers_dict = {}
    onehot_dict = {}
    
    if encoding_type == 'label':
        # Label Encoding
        if fit_encoders:
            encoders_dict = {}
            
            # Fit and transform categorical features
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = df[col].fillna('missing')
                df[col] = le.fit_transform(df[col])
                encoders_dict[col] = le
        else:
            if encoders is None:
                raise ValueError("encoders must be provided when fit_encoders=False with encoding_type='label'")
            
            # Transform categorical features
            for col in categorical_cols:
                df[col] = df[col].fillna('missing')
                # Handle unseen categories
                known_labels = set(encoders[col].classes_)
                df[col] = df[col].apply(
                    lambda x: encoders[col].transform([x])[0] if x in known_labels else -1
                )
    
    elif encoding_type == 'onehot':
        # One-hot Encoding
        if fit_encoders:
            onehot_dict = {}
            
            # Fit and transform each categorical column
            for col in categorical_cols:
                df[col] = df[col].fillna('missing')
                onehot_dict[col] = df[col].unique().tolist()
                # Create one-hot encoded columns
                onehot_encoded = pd.get_dummies(df[[col]], prefix=col, drop_first=False)
                df = df.drop(columns=[col])
                df = pd.concat([df, onehot_encoded], axis=1)
        else:
            if onehot_categories is None:
                raise ValueError("onehot_categories must be provided when fit_encoders=False with encoding_type='onehot'")
            
            # Transform with known categories
            for col in categorical_cols:
                df[col] = df[col].fillna('missing')
                # Create one-hot with only known categories, unknown = all zeros
                onehot_encoded = pd.get_dummies(df[[col]], prefix=col, drop_first=False)
                
                # Ensure all expected columns exist
                expected_cols = [f"{col}_{cat}" for cat in onehot_categories[col]]
                for expected_col in expected_cols:
                    if expected_col not in onehot_encoded.columns:
                        onehot_encoded[expected_col] = 0
                
                # Keep only expected columns in correct order
                onehot_encoded = onehot_encoded[[col for col in expected_cols if col in onehot_encoded.columns]]
                
                df = df.drop(columns=[col])
                df = pd.concat([df, onehot_encoded], axis=1)
    
    # ========== NUMERICAL SCALING ==========
    if fit_encoders:
        scalers_dict = {}
        
        # Store log1p flag for inverse transformation
        scalers_dict['use_log1p'] = use_log1p
        
        # Fit and transform numerical features
        for col in numerical_cols:
            scaler = MinMaxScaler()
            df[col] = df[col].fillna(0)
            
            # Apply log1p transformation if enabled (for 'value' column)
            if use_log1p and col == 'value':
                df[col] = np.log1p(df[col])
            
            df[col] = scaler.fit_transform(df[[col]]).flatten()
            scalers_dict[col] = scaler
    else:
        if scalers is None:
            raise ValueError("scalers must be provided when fit_encoders=False")
        
        # Get log1p flag from training scalers
        use_log1p_from_train = scalers.get('use_log1p', False)
        
        # Transform numerical features
        for col in numerical_cols:
            df[col] = df[col].fillna(0)
            

            df[col] = scalers[col].transform(df[[col]]).flatten()
            # Apply log1p transformation if it was used in training
            if use_log1p_from_train and col == 'value':
                df[col] = np.log1p(df[col])
            
            #df[col] = scalers[col].transform(df[[col]]).flatten()
    
    return df, encoders_dict, scalers_dict, onehot_dict


def inverse_transform_target(values: np.ndarray, scaler: StandardScaler, 
                             scalers_dict: Dict) -> np.ndarray:
   
    # Ensure 2D shape for scaler
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    
    # Inverse scale
    values_unscaled = scaler.inverse_transform(values).flatten()
    
    # Inverse log1p if it was applied
    if scalers_dict.get('use_log1p', False):
        values_unscaled = np.expm1(values_unscaled)
        values_unscaled = np.round(values_unscaled, 2)  # Round to 2 decimal places for currency values
    
    return values_unscaled

# Need to test this function with a sample dataframe to ensure it works correctly, especially the handling of log1p and inverse transformations.

'''
if __name__ == '__main__':
    from pathlib import Path
    
    # Get the dataset path based on script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dataset_dir = project_root / 'dataset'
    
    # Load data
    df = feather.read_table(str(dataset_dir / 'subset_set.feather'), memory_map=True).to_pandas()
    df.to_csv(str(dataset_dir / 'subset_set.csv'), index=False)
    
    # Preprocess features
    train_df, encoders, scalers, onehot_cats = preprocess_features(
                df,
                fit_encoders=True,
                encoding_type='onehot',
                use_log1p=False
            )
    
    train_df.to_csv(str(dataset_dir / 'train_preprocessed.csv'), index=False)
    
    # Inverse transform 'value' column
    train_df_recovered = train_df.copy()
    if 'value' in scalers:
        train_df_recovered['value'] = inverse_transform_target(train_df['value'].values, scalers['value'], scalers)
    
    train_df_recovered.to_csv(str(dataset_dir / 'train_preprocessed_recovered.csv'), index=False)
    
    print("✓ Preprocessing complete!")
    print(f"  - Loaded from: {dataset_dir / 'subset_set.feather'}")
    print(f"  - Saved to: {dataset_dir / 'train_preprocessed_recovered.csv'}")


'''