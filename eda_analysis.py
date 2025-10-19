import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

# ---------------//////////////////////---------------
#                       DATA LOADING

# Load datasets
train = pd.read_csv("data/train.csv", index_col='id')
test = pd.read_csv("data/test.csv", index_col='id')
origin = pd.read_csv("data/bank-full.csv", delimiter=',')

# Preprocess original dataset
origin["y"] = origin["Target"].map({'yes': 1, 'no': 0})
origin.index.name = 'id'
origin.index += test.index.max() + 1

# Define features and target
features = train.columns[:-1].difference(['id'])
target = train['y']
origin_target = origin['y']

# Print dataset sizes
print('Train size: ', train.shape)
print('Test size:  ', test.shape)

# ---------------//////////////////////---------------
#                   EXPLORATORY DATA ANALYSIS

# Identify categorical and numerical columns
cat_col = list(train.select_dtypes(include='object').columns)
num_col = list(train.select_dtypes(
    exclude='object').columns.difference(['y', 'id']))

# Create violin plots for numerical features
columns = num_col
n_cols = 3
n_rows = int(np.ceil(len(columns)/n_cols))
fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
ax = ax.flatten()

for i, column in enumerate(columns):
    sns.violinplot(data=train, x='y', y=train[column], hue='y',
                   ax=ax[i], palette=['#78BC61', '#FF7F50'])
    ax[i].set_title(f"{column} Distribution by Target")
    ax[i].set_xlabel(None)

    # Collect legend handles and labels
    plot_axes = [ax[i]]
    handles = []
    labels = []
    for plot_ax in plot_axes:
        handles += plot_ax.get_legend_handles_labels()[0]
        labels += plot_ax.get_legend_handles_labels()[1]

# Hide empty subplots
for i in range(i+1, len(ax)):
    ax[i].axis('off')

# Create unified legend
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04),
           fontsize=14, ncol=3)
plt.tight_layout()

# Create specialized duration plot
plt.figure(figsize=(12, 6))
sns.violinplot(
    data=train,
    y=train['y'].astype(str),  
    x='duration',
    hue=train['y'].astype(str),  
    palette=['#78BC61', '#FF7F50'],
    split=False,
    inner='quartile'
)

# Configure grid and aesthetics
x_min, x_max = plt.xlim()
grid_interval = 200
grid_positions = np.arange(0, x_max + grid_interval, grid_interval)
plt.xticks(grid_positions, rotation=45)
plt.grid(axis='x', which='major', color='gray', linestyle='--', linewidth=0.5)
ax = plt.gca()
ax.set_axisbelow(True)
plt.title('Duration Distribution by Target Class')
plt.xlabel('Duration (seconds)')
plt.ylabel('Target Class')

# Compare categorical feature distributions between train and test
columns = cat_col
n_cols = 3
n_rows = int(np.ceil(len(columns)/n_cols))
fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
ax = ax.flatten()

for i, column in enumerate(columns):
    sns.histplot(train[column], label='Train', ax=ax[i], color='#78BC61')
    sns.histplot(test[column], label='Test', ax=ax[i], color='#FF7F50')
    ax[i].set_title(f"{column} Distribution Comparison")
    ax[i].tick_params(axis='x', rotation=45)

    # Collect legend elements
    plot_axes = [ax[i]]
    handles = []
    labels = []
    for plot_ax in plot_axes:
        handles += plot_ax.get_legend_handles_labels()[0]
        labels += plot_ax.get_legend_handles_labels()[1]

# Hide empty subplots
for i in range(i+1, len(ax)):
    ax[i].axis('off')

# Add unified legend
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04),
           fontsize=14, ncol=2)
plt.tight_layout()

# Target class distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=train, x=train['y'].astype(str), hue='y',
             palette=['#78BC61', '#FF7F50'])
plt.title('Target Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')

# ---------------//////////////////////---------------
#                 ADVANCED DISTRIBUTION ANALYSIS

def plot_dot_continuous(df, column, target, ax_2, 
                       show_yticks=False, color='green', df_set='train'):
    """Helper function to plot mean target values per bin"""
    bins = pd.cut(df[column], bins=n_bins)
    bins = pd.IntervalIndex(bins)
    bins = (bins.left + bins.right) / 2
    target = target.groupby(bins).mean()
    ax_2.plot(target.index, target, linestyle='', marker='.', color=color,
             label=f'Mean {df_set} {target.name}')
    ax_2.grid(visible=False)
    
    if not show_yticks:
        ax_2.set_yticks([])
    return ax_2

def plot_continuous(features, ax):
    """Plot distribution and mean target values for a feature"""
    temp = total_df.copy()
    # Plot distribution
    sns.histplot(data=temp, x=features, hue='set', ax=ax, 
                 hue_order=hue_labels, common_norm=False, 
                 palette=custom_palette, **histplot_hyperparams)
    # Plot mean values
    ax_2 = ax.twinx()
    ax_2 = plot_dot_continuous(
        temp.query('set=="train"'), features, target, ax_2,
        color='#48703A', df_set='train'
    )
    ax_2 = plot_dot_continuous(
        temp.query('set=="origin"'), features, origin_target, ax_2,
        color='#FF7F50', df_set='origin'
    )
    return ax_2

# Combine all datasets for comparison
total_df = pd.concat([
    train.assign(set='train'),
    test.assign(set='test'), 
    origin.assign(set='origin'),
], ignore_index=True).reset_index(drop=True)

# Configuration
hue_labels = ['train', 'test', 'origin']
numeric_features = ['age', 'balance', 'campaign', 'day', 
                   'duration', 'pdays', 'previous']
n_bins = 50
histplot_hyperparams = {
    'kde': True,
    'stat': 'percent', 
    'bins': n_bins
}
custom_palette = {
    'train': '#78BC61',   # Green
    'test': '#3498DB',    # Blue  
    'origin': '#FF7F50'   # Orange
}

# Create comprehensive distribution plots
columns = num_col
n_cols = 3
n_rows = math.ceil(len(columns)/n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*5))
ax = ax.flatten()

for i, column in enumerate(columns):
    ax2 = plot_continuous(column, ax[i])
    ax[i].set_title(f'{column} Distribution', pad=60)
    ax[i].set_xlabel(None)

# Hide empty subplots and adjust layout
for i in range(i+1, len(ax)):
    ax[i].axis('off')
plt.tight_layout()