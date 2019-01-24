from dataset import load_data
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn import datasets

def normalize_set(X):
    X = X / 255.
    print(X.shape)
    # a = X.mean(axis=0)
    # X = X-a
    # print(a.shape)

    return X, 0, 255.

base_dir = './data'
datasets = ['mtb, city, road']

X_inputs, Y_inputs, data = load_data(base_dir, as_array=True)

print(X_inputs.shape, Y_inputs.shape)

X_inputs = X_inputs.reshape((-1, 64*64))

print(X_inputs.shape, Y_inputs.shape)
pca = PCA(n_components=32)

inp, mean, std = normalize_set(X_inputs)

pca = pca.fit(inp)
out = pca.transform(inp)

# Plot PCA

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
colors = np.array(['r', 'g', 'b'])

class_names = [x[0] for x in sorted(data.class_indices.items(), key=lambda y:y[1])]


print(out.shape)
for x, y in zip(out, Y_inputs):
    # print(x)
    i = 0
    col = colors[y == 1][0]
    ax.scatter(x[0]
               , x[1]
               , x[2]
               , c = col
               , s = 50)
ax.legend(labels=class_names)
ax.grid()
plt.show()

# Show eigenbike

eigenbike = pca.components_.reshape((-1, 64, 64))

print(pca.components_[0])
print(eigenbike)

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


