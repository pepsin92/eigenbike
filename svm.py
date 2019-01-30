from dataset import load_data
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn import svm

PCA_SIZE = 32

def normalize_set(X):
    # X = X / 255.
    # print(X.shape)
    a = X.mean(axis=0)
    # X = X-a
    # print(a.shape)

    return X-a, a

base_dir = './data'
datasets = ['mtb, city, road']

X_inputs, Y_inputs, data = load_data(base_dir, as_array=True)

print(X_inputs.shape, Y_inputs.shape)

X_inputs = X_inputs.reshape((-1, 64*64))

print(X_inputs.shape, Y_inputs.shape)
pca = PCA(n_components=PCA_SIZE)

inp, mean= normalize_set(X_inputs)

pca = pca.fit(inp)
out = pca.transform(inp)



# Show eigenbike

eigenbike = pca.components_.reshape((-1, 64, 64))

# print(pca.components_[0])
# print(eigenbike)

plt.close('all')

# w=10
# h=10
# fig=plt.figure(figsize=(64, 64))
# columns = 4
# rows = 3
# for i in range(1, columns*rows +1):
#     img = eigenbike[i-1]
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img, cmap='gray')
# plt.show()

Y_classes = np.array([np.where(i)[0][0] for i in Y_inputs])

clf = svm.SVC(gamma='scale', kernel='rbf')
clf.fit(out[:,:PCA_SIZE], Y_classes)


def make_meshgrid(x, y, h=.02):
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
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
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
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# xx, yy = make_meshgrid(out[:, 0], out[:, 1])


# Plot PCA

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_title(clf.kernel)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
colors = np.array(['r', 'b', 'g'][:Y_inputs.shape[1]])

# cm = LinearSegmentedColormap.from_list(
#         'my_cmap', colors, N=3)
cm = ListedColormap(colors)

# plot_contours(ax, clf, xx, yy, cmap=cm, alpha=0.5)

class_names = [x[0] for x in sorted(data.class_indices.items(), key=lambda y:y[1])]

print(out.shape)
for x, y in zip(out, Y_inputs):
    # print(x)
    i = 0
    col = colors[y == 1][0]
    ax.scatter(x[0]
               , x[1]
               , c = col
               , s = 50)
# ax.legend(labels=class_names)
ax.grid()
print('training score:', clf.score(out[:,:PCA_SIZE], Y_classes))

plt.show()
