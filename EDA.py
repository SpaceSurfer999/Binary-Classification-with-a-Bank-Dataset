import pandas as pd
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import optuna
import math


# %%   ---------------//////////////////////---------------
#                            GET DATA

# %%
train = pd.read_csv("data/train.csv", index_col='id')
test = pd.read_csv("data/test.csv", index_col='id')
origin = pd.read_csv("data/bank-full.csv", delimiter=',')


# %%
origin["y"] = origin["Target"].map({'yes': 1, 'no': 0})
origin.index.name = 'id'
origin.index += test.index.max()+1

# %%

features = train.columns[:-1].difference(['id'])
target = train['y']
origin_target = origin['y']
# %%

print('Train size: ', train.shape)
print('Test size:  ', test.shape)
# %%   ---------------//////////////////////---------------
#                            EDA

# %%
cat_col = list(train.select_dtypes(include='object').columns)
num_col = list(train.select_dtypes(
    exclude='object').columns.difference(['y', 'id']))
# %%

# print(sns.color_palette("pastel").as_hex())
# %%
columns = num_col
n_cols = 3
n_rows = int(np.ceil(len(columns)/n_cols))
fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
ax = ax.flatten()

for i, column in enumerate(columns):
    sns.violinplot(data=train, x='y', y=train[column], hue='y',
                   ax=ax[i], palette=['#78BC61', '#FF7F50'])

    ax[i].set_title(f"{column} violin plot of classes.")
    ax[i].set_label(None)

    plot_axes = [ax[i]]
    handles = []
    labels = []
    for plot_ax in plot_axes:
        handles += plot_ax.get_legend_handles_labels()[0]
        labels += plot_ax.get_legend_handles_labels()[1]

for i in range(i+1, len(ax)):
    ax[i].axis('off')

fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04),
           fontsize=14, ncol=3)
plt.tight_layout()
# %%
plt.figure(figsize=(12, 6))

# Преобразуем y в строковый или категориальный тип
sns.violinplot(
    data=train,
    y=train['y'].astype(str),   # Преобразуем в строки!
    x='duration',
    hue=train['y'].astype(str),  # Тоже преобразуем для hue
    palette=['#78BC61', '#FF7F50'],
    # alpha=0.5,
    split=False,
    inner='quartile'
)

x_min, x_max = plt.xlim()  # Получаем текущие границы
grid_interval = 200  # Интервал сетки
grid_positions = np.arange(0, x_max + grid_interval, grid_interval)
plt.xticks(grid_positions, rotation=45)

plt.grid(axis='x', which='major',
         color='gray',
         linestyle='--',
         linewidth=0.5,
         # alpha=0.5
         )

# Сетка на заднем плане (под графиком)
if isinstance(ax, np.ndarray):
    ax = ax[0]  # Берем первую ось из массива

# Теперь можем использовать set_axisbelow
ax.set_axisbelow(True)
plt.title('Horizontal Violin Plot: Duration Distribution by Target (y)')
plt.xlabel('Duration')
plt.ylabel('Target (y)')
plt.legend()

plt.show()

# %%
columns = cat_col
n_cols = 3
n_rows = int(np.ceil(len(columns)/n_cols))
fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
ax = ax.flatten()

for i, column in enumerate(columns):
    sns.histplot(train[column], label='Train',
                 ax=ax[i], color='#78BC61')
    sns.histplot(test[column], label='test',
                 ax=ax[i], color='#FF7F50')

    ax[i].set_title(f"{column} hist plot.")
    ax[i].set_label(None)

    plot_axes = [ax[i]]
    handles = []
    labels = []
    for plot_ax in plot_axes:
        handles += plot_ax.get_legend_handles_labels()[0]
        labels += plot_ax.get_legend_handles_labels()[1]

for i in range(i+1, len(ax)):
    ax[i].axis('off')

fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04),
           fontsize=14, ncol=3)
plt.tight_layout()


# %% Difference between target class
plt.figure(figsize=(8, 6))
sns.histplot(data=train,
             x=train['y'].astype(str), hue='y',
             palette=['#78BC61', '#FF7F50'],)

plt.title('Difference between target class')
plt.xlabel('Class')
plt.ylabel('Count')
# %%


def plot_continous(features, ax):
    temp = total_df.copy()
    sns.histplot(data=temp, x=features,
                 hue='set', ax=ax, hue_order=hue_labels,
                 common_norm=False, palette=custom_palette,
                 **histplot_hyperparams)

    ax_2 = ax.twinx()
    ax_2 = plot_dot_continous(
        temp.query('set=="train"'),
        features, target, ax_2,
        color='#48703A', df_set='train'
    )

    ax_2 = plot_dot_continous(
        temp.query('set=="origin"'),
        features, origin_target, ax_2,
        color='#FF7F50', df_set='origin'
    )

    return ax_2


def plot_dot_continous(
    df, column, target, ax_2,
    show_yticks=False, color='green',
    df_set='train'
):

    bins = pd.cut(df[column], bins=n_bins)
    bins = pd.IntervalIndex(bins)
    bins = (bins.left + bins.right) / 2
    target = target.groupby(bins).mean()
    ax_2.plot(
        target.index,
        target, linestyle='',
        marker='.', color=color,
        label=f'Mean {df_set} {target.name}'
    )
    ax_2.grid(visible=False)

    if not show_yticks:
        ax_2.get_yaxis().set_ticks([])

    return ax_2


total_df = pd.concat([
    train.assign(set='train'),
    test.assign(set='test'),
    origin.assign(set='origin'),
], ignore_index=True)

total_df.reset_index(drop=True, inplace=True)
hue_labels = ['train', 'test', 'origin']

numeric_features = ['age', 'balance', 'campaign',
                    'day', 'duration', 'pdays', 'previous']


n_bins = 50
histplot_hyperparams = {
    'kde': True,
    # 'alpha': 0.8,
    'stat': 'percent',
    'bins': n_bins
}
# line_style = '--'

custom_palette = {
    'train': '#78BC61',    # зеленый для train
    'test': '#3498DB',     # синий для test
    'origin': '#FF7F50'    # оранжевый для origin
}

columns = num_col
n_cols = 3
n_rows = math.ceil(len(columns)/n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*5))
ax = ax.flatten()

for i, column in enumerate(columns):
    ax2 = plot_continous(column, ax[i])
    # titles
    ax[i].set_title(f'{column} Distribution', pad=60)
    ax[i].set_xlabel(None)

    handles, labels = [], []
    plot_axes = [ax[i]]

handles, labels = ax[i].get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles.extend(handles2)
labels.extend(labels2)
for i in range(i+1, len(ax)):
    ax[i].axis('off')

plt.tight_layout()
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
