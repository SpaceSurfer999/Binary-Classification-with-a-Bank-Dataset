import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import optuna
from itertools import combinations

TARGET = 'y'
NUMS = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATS = ['job', 'marital', 'education', 'default', 
        'housing', 'loan', 'contact', 'month', 'poutcome']

# Load data
train = pd.read_csv("data/train.csv", index_col='id')
test = pd.read_csv("data/test.csv", index_col='id')

# Convert categorical features to category type
train[CATS] = train[CATS].astype('category')
test[CATS] = test[CATS].astype('category')

# Create combined features
combined_features = []
all_columns = NUMS + CATS

# Create feature pairs (combinations of two features)
for col1, col2 in tqdm(list(combinations(all_columns, 2))):
    feature_name = f'{col1}-{col2}'
    
    for df in [train, test]:
        df[feature_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
    
    combined_features.append(feature_name)

FEATURES = [col for col in train.columns if col != TARGET]

# Target encoding function
def target_encode(df_train, df_test, column, target=TARGET):
    
    # Calculate mean target for each category in training data
    encoding_map = df_train.groupby(column)[target].mean()
    
    # Apply encoding to train and test - convert to float immediately
    df_train[f'TE_{column}'] = df_train[column].map(encoding_map).astype('float32')
    df_test[f'TE_{column}'] = df_test[column].map(encoding_map).astype('float32')
    
    # Fill missing values with overall mean
    overall_mean = df_train[target].mean()
    df_train[f'TE_{column}'].fillna(overall_mean, inplace=True)
    df_test[f'TE_{column}'].fillna(overall_mean, inplace=True)

# Optuna objective function for hyperparameter optimization
def objective(trial):
 
    # Suggest hyperparameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'early_stopping_rounds': 50  # Moved to constructor parameters
    }
    
    # Use first fold for quick optimization
    skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    fold_scores = []
    
    for train_idx, val_idx in skf.split(train, train[TARGET]):
        # Prepare data for this fold
        X_train_fold = train.iloc[train_idx].copy()
        X_val_fold = train.iloc[val_idx].copy()
        y_train_fold = train[TARGET].iloc[train_idx].copy()
        y_val_fold = train[TARGET].iloc[val_idx].copy()
        
        # Apply target encoding
        for col in CATS + combined_features:
            target_encode(X_train_fold, X_val_fold, col)
        
        # Drop original categorical columns
        columns_to_drop = CATS + combined_features
        X_train_fold = X_train_fold.drop(columns_to_drop, axis=1)
        X_val_fold = X_val_fold.drop(columns_to_drop, axis=1)
        
        # Get common features
        common_features = [f for f in X_train_fold.columns if f != TARGET]
        
        # Train model - FIXED: early_stopping_rounds is now in constructor
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train_fold[common_features], y_train_fold,
            eval_set=[(X_val_fold[common_features], y_val_fold)],
            verbose=0
        )
        
        # Calculate validation score
        val_pred = model.predict_proba(X_val_fold[common_features])[:, 1]
        fold_score = roc_auc_score(y_val_fold, val_pred)
        fold_scores.append(fold_score)
    
    return np.mean(fold_scores)

# Run Optuna optimization
print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction='maximize')  
study.optimize(objective, n_trials=5) 

print("Best trial:")
trial = study.best_trial
print(f"  AUC: {trial.value:.4f}")
print("  Best hyperparameters:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Get best parameters from Optuna and add fixed parameters
best_params = study.best_params
best_params.update({
    'random_state': 42,
    'eval_metric': 'auc',
    'use_label_encoder': False,
    'early_stopping_rounds': 100  # For final training
})

# Cross-validation with best parameters
oof_predictions = np.zeros(len(train))
test_predictions = np.zeros(len(test))
n_folds = 5

kfold = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)

for fold, (train_idx, val_idx) in enumerate(kfold.split(train, train[TARGET])):
    print(f'Fold {fold + 1}')
    
    # Split data
    X_train = train.iloc[train_idx].copy()
    X_val = train.iloc[val_idx].copy()
    y_train = train[TARGET].iloc[train_idx].copy()
    y_val = train[TARGET].iloc[val_idx].copy()
    X_test = test.copy()
    
    # Apply target encoding to categorical features
    for col in CATS + combined_features:
        target_encode(X_train, X_val, col)
        target_encode(X_train, X_test, col)
    
    # Remove original categorical columns
    columns_to_drop = CATS + combined_features
    X_train = X_train.drop(columns_to_drop, axis=1)
    X_val = X_val.drop(columns_to_drop, axis=1)
    X_test = X_test.drop(columns_to_drop, axis=1)
    
    # Get common features
    common_features = [f for f in X_train.columns if f != TARGET]
    
    # Train model with best parameters
    model = xgb.XGBClassifier(**best_params)
    model.fit(
        X_train[common_features], y_train,
        eval_set=[(X_val[common_features], y_val)],
        verbose=100
    )
    
    # Predictions
    oof_predictions[val_idx] = model.predict_proba(X_val[common_features])[:, 1]
    test_predictions += model.predict_proba(X_test[common_features])[:, 1]
    
    fold_score = roc_auc_score(y_val, oof_predictions[val_idx])
    print(f'Fold {fold + 1} AUC: {fold_score:.4f}')

# Average test predictions across folds
test_predictions /= n_folds

# Final results
final_cv_auc = roc_auc_score(train[TARGET], oof_predictions)
print(f'Overall CV AUC: {final_cv_auc:.4f}')

# Save predictions
submission = pd.DataFrame({'id': test.index, 'y': test_predictions})
submission.to_csv('submission.csv', index=False)
print('Submission saved to submission.csv')