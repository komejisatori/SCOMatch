import wandb
import random
wandb.login()
import matplotlib.pyplot as plot
import torch
from sklearn.manifold import TSNE

def tsne(x, label):
    X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(x)
    plot.scatter(X_embedded[:,0], X_embedded[:,1], color=plot.cm.Set1(label))

def init(cfg):

    wandb.init(
        # set the wandb project where this run will be logged
        project="OpenSSL",
        # track hyperparameters and run metadata
        config=cfg
    )
    '''
    {
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    }
    '''

def log(metrics):
    wandb.log(metrics)

def log_tsne(x, label, epoch, num_classes):
    colorlist = ["r", "g", "b", "c", "m", "y", "k"]
    index_list = []
    for i in range(num_classes+1):
        index = torch.nonzero(label == i, as_tuple=False)
        index_list.append(index[:50])
    indexes = torch.cat(index_list).squeeze()
    X_embedded = TSNE().fit_transform(x[indexes]) # perplexity=20, init='pca', learning_rate='auto', n_iter=1500, verbose=False
    fig = plot.scatter(X_embedded[:, 0], X_embedded[:, 1], color=[colorlist[i] for i in label[indexes]])
    wandb.log({'tsne/{}'.format(epoch): wandb.Image(fig)})
    #plot.savefig('tsne_{}.png'.format(epoch))
