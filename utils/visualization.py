from tsnecuda import TSNE

def tsne(x, label):
    X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(x)

    pass