import csv
from dataclasses import dataclass
import warnings
from typing import Annotated, Union, List, NewType, Callable, Optional
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
    threshold : float
        The integer threshold for update; 0.8 in paper
    epsilon : float, optional
        How much to fluctuate increase the threshold per node
        "it is possible to increase this value by 0.01 every row of the matrix,
         thus having fluctuating values (which correspond to different values between
         the nodes)."
    log_sistories : bool, optional
        IDFK
    """

    def __init__(self,
                 path: Union[str, os.PathLike],
                 threshold: float,
                 epsilon: float = 0,
                 log_histories: bool = False):
        """Initializes an RBN from a .net file.

        NOTE: Assumes there are no spaces in the quoted labels.
        NOTE: Assumes edges are sorted.
        NOTE: .net files are one-indexed, yet this function converts everything to 0-indexed
        """

        self.threshold = threshold
        self.epsilon = epsilon

        with open(path, "r") as f:
            num_vtxs = f.readline()
            if num_vtxs[:10] != "*Vertices ":
                raise NetParseError("Vertices header not found.")
            num_vtxs = int(num_vtxs[10:]) # HACK kinda

            self.nodes = [None] * num_vtxs
            if log_histories:
                self.hists = [[]] * num_vtxs
                
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
            self.weights = [random.uniform(0,1) for _ in range(len(self.edges))]

    def sync_update(self):
        new_nodes = self.nodes.copy()
        threshold = self.threshold
        for i, locs in enumerate(pairwise(self.offsets, len(self.edges))):
            inc_acts = []
            for j in range(*locs):
                inc_acts.append(self.nodes[self.edges[j]].act*self.weights[self.edges[j]])
            new_nodes[i].act = 1 if sum(inc_acts) >= threshold else 0
            threshold += self.epsilon
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

    def run(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        cmap = ListedColormap(sns.color_palette("rocket", 2).as_hex())
        fig.add_axes(ax)

        def get_data():
            x = [i.x for i in rbn.nodes]
            y = [i.y for i in rbn.nodes]
            z = [i.z for i in rbn.nodes]

            return x,y,z
        
        x,y,z = get_data()

        sc = ax.scatter(x,y,z,c=[i.act for i in rbn.nodes], cmap=cmap)

        for i, node in enumerate(rbn.nodes):
            # annotate every 5 nodes
            if i % 20 == 0:
                ax.text(node.x, node.y, node.z, node.label)

        plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)


        def tick(_):
            self.sync_update()

            sc.set_facecolor([cmap(i.act) for i in rbn.nodes])

        ani = matplotlib.animation.FuncAnimation(fig, tick, 
                            interval=40, blit=False)

        plt.show()

# cmap = ListedColormap(sns.color_palette("rocket", 2).as_hex())
# cmap(1)

rbn = RBN("./net/01.net", 0.8)
rbn.run()

rbn.nodes[0]
# rbn.show_pyvis()


# sc.set_xdata

# plt.show()

# ax.scatter(


# rbn.nodes[0]
# sns.sc
# # rbn.edges[:10]
# # rbn.offsets[:10]

# # next(pairwise(rbn.offsets, len(rbn.edges)))


# # t = bitarray()
