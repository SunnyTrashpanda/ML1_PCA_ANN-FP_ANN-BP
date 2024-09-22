import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import datasets

wine = datasets.load_wine()

X = wine.data
y = wine.target

nrComponents = 13
pca = decomposition.PCA(n_components=nrComponents, svd_solver='full')
pca.fit(X)
X = pca.transform(X)


# 2D Eigenvector plotting
eigenVectors = pca.components_[:3]

#print("EigenVectors (PC1, PC2, PC3):")
#print(eigenVectors)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot PC1 vs PC2
axs[0].quiver(np.zeros(eigenVectors.shape[1]), np.zeros(eigenVectors.shape[1]),
              eigenVectors[0], eigenVectors[1],
              angles='xy', scale_units='xy', scale=1, color='r')
axs[0].set_xlim(-1, 1)
axs[0].set_ylim(-1, 1)
axs[0].grid(True)
axs[0].set_title('PC1 vs PC2')
axs[0].set_xlabel('Principal Component 1')
axs[0].set_ylabel('Principal Component 2')

# Plot PC2 vs PC3
axs[1].quiver(np.zeros(eigenVectors.shape[1]), np.zeros(eigenVectors.shape[1]),
              eigenVectors[1], eigenVectors[2],
              angles='xy', scale_units='xy', scale=1, color='g')
axs[1].set_xlim(-1, 1)
axs[1].set_ylim(-1, 1)
axs[1].grid(True)
axs[1].set_title('PC2 vs PC3')
axs[1].set_xlabel('Principal Component 2')
axs[1].set_ylabel('Principal Component 3')

# Plot PC1 vs PC3
axs[2].quiver(np.zeros(eigenVectors.shape[1]), np.zeros(eigenVectors.shape[1]),
              eigenVectors[0], eigenVectors[2],
              angles='xy', scale_units='xy', scale=1, color='b')
axs[2].set_xlim(-1, 1)
axs[2].set_ylim(-1, 1)
axs[2].grid(True)
axs[2].set_title('PC1 vs PC3')
axs[2].set_xlabel('Principal Component 1')
axs[2].set_ylabel('Principal Component 3')
plt.tight_layout()

# Scree plotting
x = np.arange(pca.n_components_) + 1
print(pca.explained_variance_ratio_)
plt.figure()
plt.plot(x, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel("principal components")
plt.ylabel("explained variance ratio")

# Load score plotting
fig, axes = plt.subplots(nrows=round(nrComponents/2)+1, ncols=1, figsize=(8, 9), sharey=True, sharex=True)
fs = 9

axes[0].bar(x, pca.components_[0])
axes[0].set_title("loadings (components) of PC1", fontsize=fs)
axes[1].bar(x, pca.components_[1])
axes[1].set_title("loadings (components) of PC2", fontsize=fs)
axes[2].bar(x, pca.components_[2])
axes[2].set_title("loadings (components) of PC3", fontsize=fs)
axes[3].bar(x, pca.components_[3])
axes[3].set_title("loadings (components) of PC4", fontsize=fs)
axes[4].bar(x, pca.components_[4])
axes[4].set_title("loadings (components) of PC5", fontsize=fs)
axes[5].bar(x, pca.components_[5])
axes[5].set_title("loadings (components) of PC6", fontsize=fs)
axes[6].bar(x, pca.components_[6])
axes[6].set_title("loadings (components) of PC7", fontsize=fs)
axes[6].set_xticks(x)
axes[6].set_xticklabels(wine.feature_names)


fig, axes = plt.subplots(nrows=round(nrComponents/2), ncols=1, figsize=(8, 9), sharey=True, sharex=True)

axes[0].bar(x, pca.components_[7])
axes[0].set_title("loadings (components) of PC8", fontsize=fs)
axes[1].bar(x, pca.components_[8])
axes[1].set_title("loadings (components) of PC9", fontsize=fs)
axes[2].bar(x, pca.components_[9])
axes[2].set_title("loadings (components) of PC10", fontsize=fs)
axes[3].bar(x, pca.components_[10])
axes[3].set_title("loadings (components) of PC11", fontsize=fs)
axes[4].bar(x, pca.components_[11])
axes[4].set_title("loadings (components) of PC12", fontsize=fs)
axes[5].bar(x, pca.components_[12])
axes[5].set_title("loadings (components) of PC13", fontsize=fs)
axes[5].set_xticks(x)
axes[5].set_xticklabels(wine.feature_names)
plt.show()