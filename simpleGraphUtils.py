import numpy as np
from matplotlib import pyplot as plt
import simpleGraph as sG
from matplotlib.colors import LogNorm

def plotGraphAsMatrix(G, colorMap='BuGn', filename=None):
    """
    Assumes that G is a simple graph.
    Will plot G as matrix with imshow
    """
    mat = [[0 for i in range(G.getNumVertices())]
            for _ in range(G.getNumVertices())]
    for v in G.getVertexIDs():
        for neighbor, intensity in G.getWeightedNeighbors(v):
            mat[v][neighbor] += intensity
            mat[neighbor][v] += intensity
    plt.imshow(mat, norm=LogNorm(), cmap = colorMap)
    if not filename == None:
        plt.savefig(filename, format='pdf')
    else:
        plt.show()

