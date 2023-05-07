import csv
from dataclasses import dataclass
import warnings
from typing import Annotated, Union, List, NewType, Callable, Optional, Tuple
from annotated_types import Gt, MultipleOf
import os
import random
import pyvis.network
from enum import Enum
from bitarray import bitarray
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import matplotlib.animation
import random
from matplotlib import colors
import time

import numpy as np

def pairwise(iterable, last=None):
    it = iter(iterable)
    a = next(it, last)
    for b in it:
        yield (a, b)
        a = b
    yield (a, last)    

Bit = NewType("Bit", Annotated[int, MultipleOf(1)])
BitArray = NewType("BitArray", bitarray)

@dataclass
class Node:
    act: Bit # TODO should really be pulled out into its own array
    label: str
    x: float
    y: float
    z: float

class NetParseError(Exception):
    pass

class NetParseWarning(Warning):
    pass

class RBN:
    """A brain boolean network parser

    Parameters
    ----------
    path : os.PathLike
        The path to the .net file to parse containing a brain map.
    threshold : Tuple[float, float]
        The integer threshold for update; 0.8 in paper

    """

    def __init__(self,
                 path: Union[str, os.PathLike],
                 threshold: Tuple[float, float],
                 connection_eps: float = 0.01,
                 connection_threshold: float = 0.8,
                 weight_seeds: Tuple[float, float] = [0.51, 0.288]):
        """Initializes an RBN from a .net file.

        NOTE: Assumes there are no spaces in the quoted labels.
        NOTE: Assumes edges are sorted.
        NOTE: .net files are one-indexed, yet this function converts everything to 0-indexed
        """

        self.edges = []
        self.threshold_lower = threshold[0]
        self.threshold_upper = threshold[1]

        with open(path, "r") as f:
            num_vtxs = f.readline()
            if num_vtxs[:10] != "*Vertices ":
                raise NetParseError("Vertices header not found.")
            num_vtxs = int(num_vtxs[10:]) # HACK kinda

            self.nodes = [None] * num_vtxs
                
            for i in range(num_vtxs):
                vals = f.readline().strip().split(" ")
                if len(vals) != 5:
                    raise NetParseError(f"Node {i} line is incomplete")
                
                self.nodes[i] = Node(
                    random.getrandbits(1),
                    vals[1].strip("\""), # HACK we're not handling quotes correctly
                    *map(float, vals[2:])
                )

            self.offsets = []
            self.edges = []
            if f.readline() != "*Edges\n": # NOTE Could be more permissive.
                warnings.warn(
                    "Edges header not found, assuming rest of lines are edges",
                    NetParseWarning
                )

            prev = None
            while line := f.readline():
                vals = line.split(" ")
                if len(vals) < 2:
                    raise NetParseError(f"Found invalid edge line")

                dst, src = int(vals[0])-1, int(vals[1])-1
                if dst < 0 or src < 0:
                    raise ValueError("Invalid edge specified in file")
                self.edges.append(src)

                if prev != dst:
                    self.offsets.append(len(self.edges)-1)
                    prev = dst

            # seed random weights for the connection edges
            self.weights = [np.random.normal(*weight_seeds)
                            for _ in range(len(self.edges))]
            # we descritize the edge seeds:
            # "Starting from a [edge] threshold value of 0.8, it is possible to increase
            #  this value by 0.01 every row of the matrix, thus having fluctuating
            #  values (which correspond to different values between the nodes)." -- the paper
            epsilon = 0
            for index, i in enumerate(self.weights):
                if index % 83 == 0 and index != 0:
                    epsilon += connection_eps
                self.weights[index] = 1 if i+epsilon >= connection_threshold else 0

            breakpoint()

    def sync_update(self):
        new_nodes = self.nodes.copy()
        threshold_lower = self.threshold_lower
        threshold_upper = self.threshold_upper
        for i, locs in enumerate(pairwise(self.offsets, len(self.edges))):
            inc_acts = []
            for j in range(*locs):
                inc_acts.append(self.nodes[self.edges[j]].act*self.weights[self.edges[j]])
            new_nodes[i].act = 1 if (sum(inc_acts) > threshold_lower and
                                     sum(inc_acts) < threshold_upper) else 0
        self.nodes = new_nodes
        
    def show_pyvis(self, name="rbn"):
        net = pyvis.network.Network()
        for i, n in enumerate(self.nodes):
            net.add_node(i, label=n.label, value=n.act)

        for i, locs in enumerate(pairwise(self.offsets, len(self.edges))):
            for idx in range(*locs):
                net.add_edge(i, self.edges[idx])        
            
        net.toggle_physics(True)
        net.show(f"{name}.html")

    def load_state_file(self, f):
        with open(f, 'r') as df:
            reader = csv.reader(df, delimiter=" ")
            data = list(reader)

        node_activations = [[i[0], int(i[1])] for i in data[:83]]
        edge_activations = [[float(j) for j in i] for i in data[83:]]

        self.weights = [i[2] for i in edge_activations]
        for i, n in enumerate(self.nodes):
            assert n.label == node_activations[i][0]
            n.act = node_activations[i][1]

    def run(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        # cmap = ListedColormap(sns.color_palette("Spectral", 2).as_hex())
        cmap = colors.ListedColormap(['red', 'green'])
        fig.add_axes(ax)

        def get_data():
            x = [i.x for i in rbn.nodes]
            y = [i.y for i in rbn.nodes]
            z = [i.z for i in rbn.nodes]

            return x,y,z
        
        x,y,z = get_data()

        sc = ax.scatter(x,y,z, cmap=cmap)

        edges_src = []
        edges_dest = []

        get = lambda node:(self.nodes[node].x, self.nodes[node].y, self.nodes[node].z)
        for i, locs in enumerate(pairwise(self.offsets, len(self.edges))):
            for idx in range(*locs):
                # x,y,z = zip(get(i), get(idx))
                x,y,z = zip(get(i), get(self.edges[idx]))
                ax.plot3D(x,y,z, color='gray',linewidth=0.4)


        plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)


        text = {}

        def tick(_):
            self.sync_update()

            for i, node in enumerate(rbn.nodes):
                # annotate every 5 nodes
                if node.act == 1:
                    pass
                    # text[i] = ax.text(node.x, node.y, node.z, node.label, size=5)
                elif text.get(i):
                    text[i].remove()
                    text[i] = None

            # sc.set_color([cmap(i.act) for i in rbn.nodes])
            sc.set_color([cmap(i.act) for i in rbn.nodes])

            return sc,

        ani = matplotlib.animation.FuncAnimation(fig, tick, 
                                                 interval=500, blit=False)

        plt.show()
        ani.pause()
        time.sleep(5)
        ani.resume()

rbn = RBN("./net/01.net", threshold=[0,3],
          connection_eps=0.01, connection_threshold=0.8,
          weight_seeds=[0.51, 0.288])
# rbn.load_state_file("./net/01.weights")
rbn.run()
