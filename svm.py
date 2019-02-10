from dataset import load_data
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn import svm

from scrapers import scrape

# UNCOMMENT TO NOT DOWNLOAD IMAGES
# scrape(0.1, rescrape=True)

PCA_SIZE = 1
kernel = 'linear'


def normalize_set(X, mean = None):
    if mean is not None:
        X = X - mean
        return X

    # X = X / 255.
    # print(X.shape)
    a = X.mean(axis=0)
    X = X-a
    # print(a.shape)
    # b = max(np.abs(np.min(X, axis=0)), np.max(X, axis=0))
    # X = X / b
    return X-a, a


def normalize_pca(X, scale=None):
    if scale is not None:
        X = X / scale
        return X
    b = X.max(axis=0)

    X = X / b
    return X, b


base_dir = './data'
datasets = ['mtb, city, road']

X_inputs, Y_inputs, data = load_data(base_dir+'/training', as_array=True)

print(X_inputs.shape, Y_inputs.shape)

X_inputs = X_inputs.reshape((-1, 64*64))

print(X_inputs.shape, Y_inputs.shape)
pca = PCA(n_components=max(9, PCA_SIZE))

inp, mean= normalize_set(X_inputs)

pca = pca.fit(inp)
out, scaling_factor = normalize_pca(pca.transform(inp))



# Show eigenbike

eigenbike = pca.components_.reshape((-1, 64, 64))

plt.close('all')

# w=10
# h=10
# fig=plt.figure(figsize=(16, 16))
# columns = 3
# rows = 3
# for i in range(1, columns*rows +1):
#     img = eigenbike[i-1]
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img, cmap='seismic')
# plt.savefig('text/eigenbike.png')
# plt.show()

# train SVM

Y_classes = np.array([np.where(i)[0][0] for i in Y_inputs])

clf = svm.SVC(gamma='scale', kernel=kernel)
clf.fit(out[:,:PCA_SIZE], Y_classes)


def make_meshgrid(x, y, h=.01):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    buff = 0.5
    # x_min, x_max = x.min() - buff, x.max() + buff
    # y_min, y_max = y.min() - buff, y.max() + buff
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel()])
    Z = Z.reshape(xx.shape)
    print(Z.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out




# Plot PCA

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_title('One-component SVM', fontsize=20)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
# ax.set_title('2 component PCA', fontsize=20)
colors = np.array(['r', 'b', 'g'][:Y_inputs.shape[1]])
shapes = ['.', 's']

# cm = LinearSegmentedColormap.from_list(
#         'my_cmap', colors, N=3)
cm = ListedColormap(colors)


class_names = [x[0] for x in sorted(data.class_indices.items(), key=lambda y:y[1])]

print(f'out.shape: {out.shape}')

# print(out.shape)
# print(Y_classes)

X_val, Y_val, _ = load_data(base_dir+'/validation', as_array=True)

X_val = X_val.reshape((-1, 64*64))

X_val = normalize_set(X_val, mean=mean)

X_val_out = normalize_pca(pca.transform(X_val), scaling_factor)

Y_val_classes = np.array([np.where(i)[0][0] for i in Y_val])

print('training score:', clf.score(out[:,:PCA_SIZE], Y_classes))
print('validation score:', clf.score(X_val_out[:,:PCA_SIZE], Y_val_classes))

if PCA_SIZE == 1:

    plot_x = out
    plot_y = Y_classes
    points = 'training'

    class_names = [class_names[0], class_names[1], 'Val-'+class_names[0], 'Val-'+class_names[1]]

    for y in range(Y_inputs.shape[1]):
        idx = (plot_y == y)
        # print(idx.shape, plot_y.shape)
        x = plot_x[idx, :]
        # for x, y in zip(out, Y_inputs):
        # print(x)
        i = 0
        col = colors[y]
        ax.scatter(x[:, 0]
                   , x[:, 1]
                   # , x[2]
                   , c = col
                   , marker = shapes[0]
                   # , s = 5
                   )
    plot_x = X_val_out
    plot_y = Y_val_classes
    points='validation'

    for y in range(Y_inputs.shape[1]):
        idx = (plot_y == y)
        # print(idx.shape, plot_y.shape)
        x = plot_x[idx, :]
        # for x, y in zip(out, Y_inputs):
        # print(x)
        i = 0
        col = colors[y]
        ax.scatter(x[:, 0]
                   , x[:, 1]
                   # , x[2]
                   , c = col
                   , marker = shapes[1]
                   # , s = 50
                   )


    ax.legend(labels=class_names)
    xx, yy = make_meshgrid(plot_x[:, 0], plot_x[:, 1])
    plot_contours(ax, clf, xx, yy, cmap=cm, alpha=0.3)
    ax.grid()
    # print('training score:', clf.score(out[:,:PCA_SIZE], Y_classes))
    # print(Y_classes)
    # plt.savefig(f'text/SVM-one-component.png')
    plt.show()

