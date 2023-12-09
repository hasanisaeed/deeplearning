import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

csfont = {'fontname': 'Times New Roman'}
sns.set()
sns.set(rc={"figure.figsize": (10, 8)})

resr = np.array([[0, 2], [3, 4]], dtype=int)

u = np.unique(resr)
bounds = np.concatenate(([resr.min() - 1], u[:-1] + np.diff(u) / 2., [resr.max() + 1]))

norm = colors.BoundaryNorm(bounds, len(bounds) - 1)

_files = [
    'output_matrix____raw_signal',
    'output_matrix____dropout_1',
    'output_matrix____lstm_1',
    'output_matrix____global_average_pooling1d_1',
    'output_matrix____concatenate_1',
    'output_matrix____dense_1',
]
_plot_title = [
    'Raw Signal',
    'LSTM Layer (a)',
    'Dropout Layer (b)',
    'FCN Layer (c) ',
    'Concatenate Layer (d)',
    'Softmax Layer 5 (e)',
]
_colors = [
    "#000066",
    "#4c0080",
    "#990033",
    # "#ff0000",
    "#cc6600",
    # "#cccc00",
    "#cccc00",
    "#00b300",
    "#669999",
    "#00ffcc",
    "#0086b3",
    "#0000ff",
    # "#ccccff",
    "#cc00ff",
    "#ff33cc"
    # "#ff99c2",
]
PALETTE = sns.set_palette(sns.color_palette(_colors))
# PALETTE = sns.color_palette('deep', n_colors=18)
CMAP = colors.ListedColormap(_colors)

RANDOM_STATE = 42

file_name = 'concatenate_1'
fig = plt.figure()

# df = pd.read_csv('weights/output_matrix____dense_1.csv')
# # Delete first
# df = df.drop([df.columns[0]], axis=1)

# dataset = load_iris()
# m_features = list(df.columns.values)[:-1]
# features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
m_target = 'class'

# target = 'species'
# iris = pd.DataFrame(
#     dataset.data,
#     columns=features)

# iris[target] = dataset.target


# print(df[m_features])
# print(iris[features])

fig.tight_layout(pad=1)


def plot_iris_2d(x, y, title, xlabel="", ylabel="", index=1):
    print(f'>> INDEX IS: {str(index)}')
    plt.rcParams['font.family'] = ['serif']
    # plt.rcParams.update({'font.size': 22})
    # sns.set_style("darkgrid")
    plt.subplot(2, 3, index)

    # print(df['class'])
    plt.scatter(x, y,
                c=df['class'],
                cmap=CMAP,
                s=50)

    plt.title(title, y=-0.2)

    # plt.xlabel(xlabel, fontsize=16)
    # plt.ylabel(ylabel, fontsize=16)
    # plt.show()
    # plt.savefig('2D___tSNE_' + file_name + '___.png',
    #             format='png')


def plot_iris_3d(x, y, z, title, name=''):
    sns.set_style('whitegrid')
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)

    ax.scatter(x, y, z,
               c=df['class'],
               cmap=CMAP,
               s=70)

    ax.set_title(title, y=-0.2)
    #
    # fsize = 14
    # ax.set_xlabel("1st feature", fontsize=fsize)
    # ax.set_ylabel("2nd feature", fontsize=fsize)
    # ax.set_zlabel("3rd feature", fontsize=fsize)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    # plt.show()
    fig.savefig(name + '.png',
                format='png')


if __name__ == '__main__':
    for i in range(0, _files.__len__()):
        _name = _plot_title[i]
        print('>> FILE: ' + _files[i] )
        df = pd.read_csv('weights/' + _files[i] + '.csv')
        # Delete first_files[i]
        df = df.drop([df.columns[0]], axis=1)
        m_features = list(df.columns.values)[:-1]

        tsne = TSNE(n_components=2, n_iter=1000, random_state=RANDOM_STATE,
                    learning_rate=200, angle=.45)
        points = tsne.fit_transform(df[m_features])
        plot_iris_2d(
            x=points[:, 0],
            y=points[:, 1],
            title=_name,
            index=i + 1)

        # fig = matplotlib.pyplot.gcf()
        # fig.set_size_inches(15, 10)
        fig.savefig('2D3_NEW_V4.png', dpi=200, format='png')

        # tsne3 = TSNE(n_components=3, n_iter=1000, random_state=RANDOM_STATE)
        # points3 = tsne3.fit_transform(df[m_features])
        #
        # plot_iris_3d(
        #     x=points3[:, 0],
        #     y=points3[:, 1],
        #     z=points3[:, 2],
        #     title=_name,
        #     name=_name)
