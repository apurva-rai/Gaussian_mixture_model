if __name__ == '__main__':
    from pandas import read_csv
    import gmm

iris = load_iris()
X = iris.data
np.random.seed(42)
gmm = GMM(k=3, max_iter=10)
gmm.fit(X)

def jitter(x):
    return x + np.random.uniform(low=-0.05, high=0.05, size=x.shape)

def plot_axis_pairs(X, axis_pairs, clusters, classes):
    n_rows = len(axis_pairs) // 2
    n_cols = 2
    plt.figure(figsize=(16, 10))
    for index, (x_axis, y_axis) in enumerate(axis_pairs):
        plt.subplot(n_rows, n_cols, index+1)
        plt.title('GMM Clusters')
        plt.xlabel(iris.feature_names[x_axis])
        plt.ylabel(iris.feature_names[y_axis])
        plt.scatter(
            jitter(X[:, x_axis]),
            jitter(X[:, y_axis]),
            #c=clusters,
            cmap=plt.cm.get_cmap('brg'),
            marker='x')
    plt.tight_layout()

plot_axis_pairs(
    X=X,
    axis_pairs=[
        (0,1), (2,3),
        (0,2), (1,3) ],
    clusters=permuted_prediction,
    classes=iris.target)
