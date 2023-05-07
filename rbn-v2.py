# utils
import csv
import time
import random
from dataclasses import dataclass

# linalg
import numpy as np
import numpy.linalg as L

# plotting
import seaborn as sns
import matplotlib.animation
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

CONNECTOME_FILE = "./net/01.net"

@dataclass
class Node:
    x: float
    y: float
    z: float
    name: str
    act: int

class Network:

    def __init__(self,
                 connectome_file, # input connectome file
                 node_quantization_thresholds=[1,5], # a,b threshold used for quantization of nodes/update
                 edge_quantization_threshold=0.8, # threshold to quantize adj. matrix value from 0 into 1
                 edge_quantization_epsilon=0.01, # the epsilon +0.01 of the above, see note below
                 edge_init_params=[0.51, 0.288]): # mean, std of raw connection edges

        with open(connectome_file, 'r') as df:
            reader = csv.reader(df, delimiter=" ")
            data = list(reader)

        # seperate vertice parsing and edge parsing
        num_verticies = int(data[0][1])
        nodes = data[1:num_verticies+1]
        vertices = data[num_verticies+2:]

        # parse the nodes
        self.nodes = []
        for i in nodes:
            self.nodes.append(Node(
                float(i[2]), # x
                float(i[3]), # y
                float(i[4]), # z
                i[1], # name
                random.choice([0,1]) # activity; TODO random choice for now?
            ))

        # and now, construct the raw connection matrix
        raw_edges = np.zeros((num_verticies,
                            num_verticies))

        # create a list of vertices
        vertices = [(int(i[0])-1, int(i[1])-1) for i in vertices]

        # register a raw conection matrix
        for a,b in vertices:
            # set raw base weights to 1

            # "Areas have been connected by an adjacency matrix
            #  of 82 rows and 82 columns, whose values are comprised
            #  between 0 and 1, with an average connection value of
            #  0.51 and a standard deviation of 0.288 for each node."

            # recall we write matricies (row, col)
            # so the output dimension is first
            # also :music: one indexingggg
            raw_edges[b][a] = np.random.normal(0.51, 0.288)

        # descritize matrix by with 
        # "Starting from a threshold value of 0.8, it is possible to increase
        #  this value by 0.01 every row of the matrix, thus having fluctuating
        #  values (which correspond to different values between the nodes)."
        edge_thresholds = np.full((num_verticies,
                                   num_verticies), edge_quantization_threshold)
        epsilon = 0
        for i in range(num_verticies):
            for j in range(num_verticies):
                edge_thresholds[i][j] += epsilon
            epsilon += edge_quantization_epsilon

        # compute the final adjacency matrix
        adjacency_matrix = (raw_edges > edge_thresholds).astype(int)

        # store the adjacency matrix
        self.adjacency = adjacency_matrix
        self.vertices = vertices

    # node activation vector
    @property
    def nodevec(self):
        node_acts = [i.act for i in self.nodes]
        return np.array(node_acts)

    # node location vector
    @property
    def nodeloc(self):
        locs = [np.array([i.x,
                          i.y,
                          i.z]) for i in self.nodes]
        return np.array(locs)

    # set the network to a certain vector state
    def set(self, nodevec):
        for (node, i) in zip(self.nodes, nodevec):
            node.act = int(i)

    # synchronous update
    def step(self):
        next = self.adjacency@self.nodevec
        self.set(next)

    # animation tooling
    def run(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        cmap = colors.ListedColormap(['red', 'green'])
        fig.add_axes(ax)

        # transpose to seperate x,y,z
        x,y,z = self.nodeloc.transpose()

        sc = ax.scatter(x,y,z, cmap=cmap)

        # draw edges
        for src,dest in self.vertices:
            x,y,z = self.nodeloc[[src,dest]].transpose()
            ax.plot3D(x,y,z, color='gray',linewidth=0.4)

        # draw legend
        plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

        # text = {}

        # define plot tick function
        def tick(_):
            self.step()

            # for i, node in enumerate(self.nodes):
            #     # annotate every 5 nodes
            #     if node.act == 1:
            #         pass
            #         # text[i] = ax.text(node.x, node.y, node.z, node.label, size=5)
            #     elif text.get(i):
            #         text[i].remove()
            #         text[i] = None

            # sc.set_color([cmap(i.act) for i in rbn.nodes])
            sc.set_color(cmap(self.nodevec))

            return sc,

        ani = matplotlib.animation.FuncAnimation(fig, tick, 
                                                 interval=500, blit=False)

        plt.show()
        ani.pause()
        time.sleep(5)
        ani.resume()

n = Network(CONNECTOME_FILE)
n.run()

