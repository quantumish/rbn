import csv
from dataclasses import dataclass
import warnings
from typing import Annotated, Union
from annotated_types import Gt, MultipleOf
import os
import random
import pyvis.network

def pairwise(iterable):
    it = iter(iterable)
    a = next(it, None)
    for b in it:
        yield (a, b)
        a = b
    yield (a, None)

@dataclass
class Node:
    act: Annotated[int, MultipleOf(1)]
    label: str
    x: float
    y: float
    z: float

class NetParseError(Exception):
    pass

class NetParseWarning(Warning):
    pass

class RBN:
    def __init__(self, path: Union[str, os.PathLike]):
        """Initializes an RBN from a .net file.

        NOTE: Assumes there are no spaces in the quoted labels.
        NOTE: Assumes edges are sorted.
        NOTE: .net files are one-indexed, yet this function converts everything to 0-indexed
        """
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

    def show_pyvis(self):
        net = pyvis.network.Network()
        for i, n in enumerate(self.nodes):
            net.add_node(i, label=n.label, value=n.act)

        for i, locs in enumerate(pairwise(self.offsets)):
            idxs = range(locs[0], locs[1] if locs[1] else len(self.edges))
            for idx in idxs:
                net.add_edge(i, self.edges[idx])
            
        net.toggle_physics(True)
        net.show("rbn.html", notebook=False)

rbn = RBN("./net/01.net")
rbn.show_pyvis()
