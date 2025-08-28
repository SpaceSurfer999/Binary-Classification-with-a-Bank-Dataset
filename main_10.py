# %% [markdown]
# # Binary Classification with Bank Dataset
#
# ## 1. Configuration and Imports

# %%
from datetime import datetime, timedelta
import time
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import optuna
import gc
from sklearn.model_selection import StratifiedKFold, train_test_split
from pandas.errors import PerformanceWarning
from sklearn.metrics import roc_auc_score
from optuna.samplers import TPESampler
from itertools import combinations
from xgboost import XGBClassifier
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
warnings.simplefilter(action="ignore", category=PerformanceWarning)
TARGET = 'y'
NUMS = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATS = ['job', 'marital', 'education', 'default',
        'housing', 'loan', 'contact', 'month', 'poutcome']

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %% [markdown]
# ## 2. Data Processor Class

# %%


class DataProcessor:
    """Класс для обработки данных"""

    def __init__(self):
        self.te_columns = []
        self.te_orig_columns = []
        # Новые колонки для TE из оригинальных данных

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        start_time = time.time()
        """Загрузка и подготовка данных"""
        # print(f"Current time:  {datetime.now()}")
        print("📂 Loading data...")
        train = pd.read_csv('data/train.csv', index_col='id')
        test = pd.read_csv('data/test.csv', index_col='id')
        orig = pd.read_csv('data/bank-full.csv', delimiter=',')

        # Prepare target variable
        orig['y'] = orig['Target'].map({'yes': 1, 'no': 0}).astype(np.int8)

        # Convert categorical columns
        for df in [train, test, orig]:
            df[CATS] = df[CATS].astype('category')

        print(
            f"✅ Data loaded: train={train.shape}, test={test.shape}, orig={orig.shape}")
        return train, test, orig

    def add_original_data_as_columns(self, train: pd.DataFrame, test: pd.DataFrame,
                                     orig: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Добавление оригинальных данных как новых колонок через Target Encoding"""
        print("🔄 Adding original data as new columns via Target Encoding...")

        # Сохраняем индексы
        train_index = train.index
        test_index = test.index

        TE_ORIG = []

        print(f"Processing {len(categorical_cols)} columns... ", end="")
        for i, c in enumerate(categorical_cols):
            if i % 10 == 0:
                print(f"{i}, ", end="")

            # Вычисляем среднее значение целевой переменной по группам в оригинальных данных
            tmp = orig.groupby(c, observed=False)['y'].mean()
            tmp = tmp.astype('float32')
            tmp.name = f"TE_ORIG_{c}"
            TE_ORIG.append(f"TE_ORIG_{c}")

            # Добавляем новые колонки во все датафреймы
            train = train.merge(tmp, on=c, how='left')
            test = test.merge(tmp, on=c, how='left')
            orig = orig.merge(tmp, on=c, how='left')  # Добавляем также в orig

        print()
        self.te_orig_columns = TE_ORIG

        # Восстанавливаем индексы
        train.index = train_index
        test.index = test_index

        print(f"✅ Added {len(TE_ORIG)} new columns from original data")
        return train, test, orig  # Возвращаем все три датафрейма

    def create_interaction_features(self, train: pd.DataFrame, test: pd.DataFrame,
                                    orig: pd.DataFrame, n_combinations: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Создание комбинированных признаков"""
        end_time = time.time()
        duration_sec = end_time-start_time
        duration = timedelta(seconds=duration_sec)
        print(f"С начала тестирования прошло: ⏱️ {duration}")
        print("🛠️ Creating interaction features...")

        # Сохраняем индексы
        train_index = train.index
        test_index = test.index

        columns = NUMS + CATS

        for r in [n_combinations]:
            for cols in tqdm(list(combinations(columns, r)), desc="Creating interaction features"):
                name = '-'.join(cols)
                cols_list = list(cols)

                # Create combined features
                for df in [train, test, orig]:
                    df[name] = df[cols_list].astype(
                        str).apply('_'.join, axis=1)

                # Unified encoding
                combined = pd.concat(
                    [train[name], test[name], orig[name]], ignore_index=True)
                combined, _ = combined.factorize()

                # Distribute encoded values
                train_len, test_len = len(train), len(test)
                train[name] = combined[:train_len]
                test[name] = combined[train_len:train_len + test_len]
                orig[name] = combined[train_len + test_len:]

                self.te_columns.append(name)
        # Сохраняем индексы
        train_index = train.index
        test_index = test.index

        print(f"✅ Created {len(self.te_columns)} interaction features")
        end_time = time.time()
        duration_sec = end_time-start_time
        duration = timedelta(seconds=duration_sec)
        print(f"С начала тестирования прошло: ⏱️ {duration}")
        return train, test, orig

# %% [markdown]
# ## 3. Feature Encoder Class

# %%


class FeatureEncoder:
    """Класс для кодирования признаков"""

    @staticmethod
    def target_encode(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame,
                      col: List[str], target: pd.Series, kfold: int = 3,
                      smooth: int = 20, agg: str = 'mean') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Target Encoding с регуляризацией"""
        col_name = '_'.join(col)
        encoded_col_name = f'TE_{agg.upper()}_{col_name}'

        # Create copies
        train = train.copy()
        valid = valid.copy()
        test = test.copy()

        # Add target temporarily
        train_temp = train.copy()
        train_temp['target_temp'] = target.values

        # Global statistics
        if agg == 'mean':
            global_stat = target.mean()
        elif agg == 'median':
            global_stat = target.median()
        elif agg == 'min':
            global_stat = target.min()
        elif agg == 'max':
            global_stat = target.max()
        else:
            global_stat = 0

        # Create folds
        train_temp['kfold'] = train_temp.index % kfold

        # Full dataset encoding
        full_grouped = train_temp.groupby(
            col, observed=False,)['target_temp'].agg([agg, 'count']).reset_index()
        full_grouped.columns = col + [f'{agg}', 'count']

        # Smoothing
        if agg == 'nunique':
            full_encoding = full_grouped[agg] / full_grouped['count']
        else:
            full_encoding = ((full_grouped[agg] * full_grouped['count']) +
                             (global_stat * smooth)) / (full_grouped['count'] + smooth)

        # Create mappings
        if len(col) > 1:
            full_mapping = dict(
                zip([tuple(x) for x in full_grouped[col].values], full_encoding))
        else:
            full_mapping = dict(zip(full_grouped[col[0]], full_encoding))

        # Cross-validation encoding
        train[encoded_col_name] = 0.0
        for i in range(kfold):
            # Statistics without current fold
            tmp_df = train_temp[train_temp['kfold'] != i]
            grouped = tmp_df.groupby(col, observed=False,)['target_temp'].agg(
                [agg, 'count']).reset_index()
            grouped.columns = col + [f'{agg}', 'count']

            # Smoothing for fold
            if agg == 'nunique':
                encoding = grouped[agg] / grouped['count']
            else:
                encoding = ((grouped[agg] * grouped['count']) +
                            (global_stat * smooth)) / (grouped['count'] + smooth)

            # Create fold mapping
            if len(col) > 1:
                fold_mapping = dict(
                    zip([tuple(x) for x in grouped[col].values], encoding))
            else:
                fold_mapping = dict(zip(grouped[col[0]], encoding))

            # Apply to current fold
            mask = train_temp['kfold'] == i
            if len(col) > 1:
                train.loc[mask, encoded_col_name] = train.loc[mask, col].apply(
                    lambda x: fold_mapping.get(tuple(x), global_stat), axis=1)
            else:
                train.loc[mask, encoded_col_name] = train.loc[mask, col[0]].map(
                    lambda x: fold_mapping.get(x, global_stat))

        # Apply to validation and test
        for df in [valid, test]:
            if len(col) > 1:
                df[encoded_col_name] = df[col].apply(
                    lambda x: full_mapping.get(tuple(x), global_stat), axis=1)
            else:
                df[encoded_col_name] = df[col[0]].map(
                    lambda x: full_mapping.get(x, global_stat))
            df[encoded_col_name] = df[encoded_col_name].astype(np.float32)

        train[encoded_col_name] = train[encoded_col_name].astype(np.float32)
        return train, valid, test

    @staticmethod
    def count_encode(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame,
                     col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Частотное кодирование"""
        counts = train[col].value_counts()

        train[f'CE_{col}'] = train[col].map(counts)
        valid[f'CE_{col}'] = valid[col].map(counts).fillna(0).astype(np.int32)
        test[f'CE_{col}'] = test[col].map(counts).fillna(0).astype(np.int32)

        return train, valid, test

# %% [markdown]
# ## 4. Model Trainer Class

# %%


class ModelTrainer:
    """Класс для обучения модели"""

    def __init__(self, n_splits: int = 5, random_state: int = 9):
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf_opt = StratifiedKFold(
            n_splits=2, random_state=random_state, shuffle=True)
        self.skf = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, shuffle=True)

    def optimize_hyperparameters_xgb(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                                     categorical_cols: List[str], n_trials: int = 20) -> dict:
        """Оптимизация гиперпараметров для xgb.train() с Optuna"""
        print("🎯 Starting hyperparameter optimization for xgb.train()...")

        # Convert to DMatrix
        dtrain = xgb.QuantileDMatrix(X_train, label=y_train,
                                     enable_categorical=True, max_bin=256)

        def objective(trial):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                "max_depth": 0,
                "subsample": trial.suggest_float("subsample", 0.7, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.95),
                "max_leaves": trial.suggest_int("max_leaves", 16, 128),
                "alpha": trial.suggest_float("alpha", 0.1, 10.0, log=True),
                "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 15),
                "grow_policy": "lossguide",
                "device": "cuda",
                "seed": self.random_state,
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5, 15),
            }

            # Create validation split
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
            )

            dtrain_split = xgb.QuantileDMatrix(X_train_split, label=y_train_split,
                                               enable_categorical=True, max_bin=256)
            dval_split = xgb.DMatrix(X_val_split, label=y_val_split,
                                     enable_categorical=True)

            # Train with early stopping
            model = xgb.train(
                params=params,
                dtrain=dtrain_split,
                num_boost_round=2000,
                evals=[(dtrain_split, "train"), (dval_split, "valid")],
                early_stopping_rounds=100,
                verbose_eval=False
            )

            # Get best score
            return model.best_score

        study = optuna.create_study(direction="maximize",
                                    sampler=TPESampler(seed=self.random_state),
                                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"✅ Best trial value: {study.best_value:.6f}")
        print("Best parameters:", study.best_params)

        return study.best_params

    def _preprocess_data(self, train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame,
                         te_columns: List[str], te_orig_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """Предобработка данных с добавлением оригинальных колонок"""
        print("🔧 Preprocessing data...")
        print(f"Current time:  {datetime.now()}")
        end_time = time.time()
        duration_sec = end_time-start_time
        duration = timedelta(seconds=duration_sec)
        print(f"С начала тестирования прошло: ⏱️ {duration}")
        feature_encoder = FeatureEncoder()

        # Create copies
        train_processed = train.copy()
        test_processed = test.copy()
        orig_processed = orig.copy()

        # Save targets
        train_target = train_processed[TARGET]
        orig_target = orig_processed[TARGET]

        # Features only
        train_features = train_processed.drop(TARGET, axis=1)
        test_features = test_processed
        orig_features = orig_processed.drop(TARGET, axis=1)

        # Target encoding из оригинальных данных (уже добавлены в DataProcessor)
        # Эти колонки уже присутствуют в данных

        # Обычное target encoding
        for col in te_columns:
            train_features, test_features, orig_features = feature_encoder.target_encode(
                train_features, test_features, orig_features, [col],
                target=train_target, smooth=10, agg='mean', kfold=10
            )

        # Count encoding
        for col in te_columns:
            train_features, test_features, orig_features = feature_encoder.count_encode(
                train_features, test_features, orig_features, col
            )

        # Reconstruct DataFrames
        train_processed = pd.concat([train_features, train_target], axis=1)
        test_processed = test_features
        orig_processed = pd.concat([orig_features, orig_target], axis=1)

        # Обновляем список фичей, включая новые колонки из оригинальных данных
        features_updated = list(train_features.columns)
        print(f"Процесс обработки завершен ✅")
        print(f"Размер train_processed: {train_processed.shape}")
        print(f"Features: {len(features_updated)}")
        print(f"TE Original columns: {len(te_orig_columns)}")

        end_time = time.time()
        duration_sec = end_time-start_time
        duration = timedelta(seconds=duration_sec)
        print(f"С начала тестирования прошло: ⏱️ {duration}")
        return train_processed, test_processed, orig_processed, features_updated

    def train_model(self, train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame,
                    te_columns: List[str], te_orig_columns: List[str], features: List[str],
                    use_optuna: bool = False, n_optuna_trials: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Обучение модели с QuantileDMatrix и опциональной оптимизацией"""

    # Preprocess data
        train_processed, test_processed, orig_processed, features_updated = self._preprocess_data(
            train, test, orig, te_columns, te_orig_columns
        )

        # Hyperparameter optimization
        if use_optuna:
            print("🔍 Running hyperparameter optimization...")
            X_opt = train_processed[features_updated]
            y_opt = train_processed[TARGET]

            best_params = self.optimize_hyperparameters_xgb(
                X_opt, y_opt, CATS, n_trials=n_optuna_trials
            )

            # Base parameters with optimized values
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "learning_rate": best_params.get("learning_rate", 0.1),
                "max_depth": 0,
                "subsample": best_params.get("subsample", 0.8),
                "colsample_bytree": best_params.get("colsample_bytree", 0.7),
                "max_leaves": best_params.get("max_leaves", 32),
                "alpha": best_params.get("alpha", 2.0),
                "lambda": best_params.get("lambda", 1.0),
                "min_child_weight": best_params.get("min_child_weight", 5),
                "scale_pos_weight": best_params.get("scale_pos_weight", 8),
                "grow_policy": "lossguide",
                "device": "cuda",
                "seed": self.random_state,
            }
        else:
            # Base parameters without optimization
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "learning_rate": 0.1,
                "max_depth": 0,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "max_leaves": 32,
                "alpha": 2.0,
                "lambda": 1.0,
                "min_child_weight": 5,
                "scale_pos_weight": 8,
                "grow_policy": "lossguide",
                "device": "cuda",
                "seed": self.random_state,
            }

        oof = np.zeros(len(train_processed))
        pred = np.zeros(len(test_processed))

        # Cross-validation
        print("🚀 Starting cross-validation training...")

        for idx, (train_idx, val_idx) in enumerate(self.skf.split(train_processed, train_processed[TARGET])):
            print(f"📊 Fold {idx + 1}/{self.n_splits}")
            print(f"Current time:  {datetime.now()}")

            # Prepare data
            X_train_fold = train_processed.iloc[train_idx][features_updated]
            X_val = train_processed.iloc[val_idx][features_updated]
            y_train_fold = train_processed.iloc[train_idx][TARGET]
            y_val = train_processed.iloc[val_idx][TARGET]
            X_test_fold = test_processed[features_updated]

            # Data augmentation
            X_train_augmented = pd.concat(
                [X_train_fold, orig_processed[features_updated]])
            y_train_augmented = pd.concat(
                [y_train_fold, orig_processed[TARGET]])

            # Convert to DMatrix
            dtrain = xgb.QuantileDMatrix(X_train_augmented, label=y_train_augmented,
                                         enable_categorical=True, max_bin=256)
            dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
            dtest = xgb.DMatrix(X_test_fold, enable_categorical=True)

            # Train model
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=10_000,
                evals=[(dtrain, "train"), (dval, "valid")],
                early_stopping_rounds=200,
                verbose_eval=200
            )

            # Predictions
            oof[val_idx] = model.predict(
                dval, iteration_range=(0, model.best_iteration + 1))
            pred += model.predict(dtest, iteration_range=(0,
                                  model.best_iteration + 1))

            fold_auc = roc_auc_score(y_val, oof[val_idx])
            print(f"📈 Fold {idx + 1} AUC: {fold_auc:.6f}")

            # Cleanup
            del model, dtrain, dval, dtest
            gc.collect()

        # Final results
        pred /= self.n_splits
        cv_auc = roc_auc_score(train_processed[TARGET], oof)
        print(f"🎯 Final CV AUC: {cv_auc:.6f}")
        print(f"Current time:  {datetime.now()}")
        end_time = time.time()
        duration_sec = end_time-start_time
        duration = timedelta(seconds=duration_sec)
        print(f"С начала тестирования прошло: ⏱️ {duration}")

        return oof, pred


# %% [markdown]
# ## 5. Main Execution - Data Preparation

# %%
start_time = time.time()
start_datetime = datetime.now()
print(f"⌛ Начало тестирования: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
processor = DataProcessor()
train, test, orig = processor.load_data()

# Сохраняем исходные индексы тестового набора
test_ids = test.index

# Добавляем оригинальные данные как новые колонки
train, test, orig = processor.add_original_data_as_columns(
    train, test, orig, CATS)

# Create interaction features
train, test, orig = processor.create_interaction_features(
    train, test, orig, n_combinations=2)

# Get features (включая новые колонки из оригинальных данных)
features = [col for col in train.columns if col != TARGET]
print(f"📊 Total Features: {len(features)}")
print(f"📋 Target encoding columns: {len(processor.te_columns)}")
print(f"📋 Original TE columns: {len(processor.te_orig_columns)}")

# %% [markdown]
# ## 6. Hyperparameter Optimization (Optional)

# %%
# Run this cell separately for hyperparameter optimization
if False:  # Set to True to run optimization
    trainer = ModelTrainer(n_splits=2, random_state=81)
    X_opt = train[features]
    y_opt = train[TARGET]

    best_params = trainer.optimize_hyperparameters(X_opt, y_opt, n_trials=20)
    print("🎉 Optimization completed!")
    print("Best parameters:", best_params)

# %% [markdown]
# ## 7. Model Training

# %% # Train final model

# Train final model with new features
# Train final model with new features
trainer = ModelTrainer(n_splits=6, random_state=81)
oof, pred = trainer.train_model(
    train, test, orig,
    processor.te_columns,
    processor.te_orig_columns,
    features,
    use_optuna=True,  # Включить оптимизацию
    n_optuna_trials=20  # Количество trials
)
# %% [markdown]
# ## 8. Create Submission

# %%
# Create submission file
# test = pd.read_csv('data/test.csv', index_col='id')
submission = pd.DataFrame({
    'id': test_ids,
    'y': pred
})
submission.to_csv('submission_main_10_3.csv', index=False)
print("✅ Submission file created: submission.csv")

# Plot feature importance (if needed)
# model = XGBClassifier()
# model.fit(train[features], train[TARGET])
# xgb.plot_importance(model, max_num_features=20)
# plt.show()

# %% [markdown]
# ## 9. Quick Test Mode

# %%
# Quick test with small data subset
if False:  # Set to True for quick testing
    # Use only first 10000 samples for testing
    train_sample = train.head(10000).copy()
    test_sample = test.head(5000).copy()
    orig_sample = orig.head(5000).copy()

    quick_trainer = ModelTrainer(n_splits=3, random_state=81)
    oof_quick, pred_quick = quick_trainer.train_model(
        train_sample, test_sample, orig_sample,
        processor.te_columns[:10],  # Only first 10 TE columns
        features[:20],  # Only first 20 features
        use_optuna=False
    )

    print("✅ Quick test completed!")

# %%
# fig, ax = plt.subplots(figsize=(10, 5))
# xgb.plot_importance(model, max_num_features=20, importance_type='gain', ax=ax)
# plt.title("Top 20 Feature Importances (XGBoost)")
# plt.show()
