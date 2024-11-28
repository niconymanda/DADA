from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
from io import BytesIO


def fig2img(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img


def get_tsne_fig(feats, labels, title):
    if feats.shape[-1] > 50:
        feats = PCA(n_components=50).fit_transform(feats)
    x_embedded = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=3
    ).fit_transform(feats)
    authors = np.unique(labels)
    author_ids = np.arange(len(authors))
    author_id_map = dict(zip(authors, author_ids))
    label_ids = np.array([author_id_map[author] for author in labels])

    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(
        x_embedded[:, 0], x_embedded[:, 1], c=label_ids, cmap="viridis"
    )
    plt.title(title)
    handles, _ = scatter.legend_elements()
    labels_unique = [authors[i] for i in np.unique(label_ids)]
    plt.legend(handles, labels_unique, title="Labels")
    return plt.gcf()


def get_tsne_img(X, y, title):
    fig = get_tsne_fig(X, y, title)
    img = fig2img(fig)
    return img
