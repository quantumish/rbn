import csv
from dataclasses import dataclass
import warnings
from typing import Annotated, Union, List, NewType
from annotated_types import Gt, MultipleOf
import os
import random
import pyvis.network
from enum import Enum
from bitarray import bitarray

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
    rule: BitArray
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
                    bitarray(),
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

        for i, locs in enumerate(pairwise(self.offsets, len(self.edges))):
            num_in = locs[1]-locs[0]            
            bytelen = (2**num_in)//8 if num_in > 2 else 1            
            self.nodes[i].rule.frombytes(os.urandom(bytelen))           

    def async_update(self):
        node_idx = random.randint(0, len(self.nodes))

        start = self.offsets[node_idx]
        stop = self.offsets[node_idx+1] if node_idx != len(self.offsets)-1 else len(self.edges)

        inc_acts = bitarray()
        for i in range(start, stop):
            inc_acts.append(self.nodes[i].act)

        rule_idx = int(inc_acts.to01(), base=2) # HACK HACK HACK
        self.nodes[node_idx].act = self.nodes[node_idx].rule[rule_idx]            

    def sync_update(self):
        new_nodes = self.nodes.copy()
        for i, locs in enumerate(pairwise(self.offsets, len(self.edges))):
            inc_acts = bitarray()
            for j in range(*locs):
                inc_acts.append(self.nodes[self.edges[j]].act)
            rule_idx = int(inc_acts.to01(), base=2) # HACK HACK HACK
            new_nodes[i].act = self.nodes[i].rule[rule_idx]            
        self.nodes = new_nodes
        
    def show_pyvis(self):
        net = pyvis.network.Network()
        for i, n in enumerate(self.nodes):
            net.add_node(i, label=n.label, value=n.act)

        for i, locs in enumerate(pairwise(self.offsets, len(self.edges))):
            for idx in range(*locs):
                net.add_edge(i, self.edges[idx])        
            
        net.toggle_physics(True)
        net.show("rbn.html", notebook=False)

rbn = RBN("./net/01.net")
rbn.show_pyvis()
rbn.sync_update()
rbn.show_pyvis()
