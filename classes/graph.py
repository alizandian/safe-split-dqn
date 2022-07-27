from __future__ import annotations
from typing import List, Tuple, Dict
from .node import Node, Region
from classes.visualization import draw_graph, draw_table, draw_graph_grid
import numpy as np

class Graph:
    cells: List[List[int]] = None

    def __init__(self, dimention, mins, maxes, transitions = None) -> None:
        self.dimention = dimention
        self.mins = mins
        self.maxes = maxes
        self.len = maxes[0] - mins[0]
        self.teil = self.len / dimention
        self.nodes: List[Node] = []
        self.transition_count = 0
        self.transitions = {} # a dictionary, key is a tuple of locations, originating cell and target cell, value is a list of bool, indicating violation
        self.transitions_full = {}
        self.transitions_from = {}
        self.transitions_from_full = {}
        self.cells = [[0 for i in range(dimention)] for j in range(dimention)] # first x then y
        self.graphs: Dict[Tuple, Graph] = {} # a dictionary, key is location, value would be another grid 

        if transitions != None:
            for t in transitions:
                self.add_transitions(t[0], t[1], t[2])


    def clamp(self, n, smallest, largest): return max(smallest, min(n, largest))


    def feed_neural_network_feedback(self):
        pass

    def get_loc(self, state):
        x = self.clamp(int((state[0] - self.mins[0]) / self.len * self.dimention), 0, self.dimention-1)
        y = self.clamp(int((state[1] - self.mins[1]) / self.len * self.dimention), 0, self.dimention-1)

        return (x, y)

    def add_transitions(self, state1, state2, violation):
        l1 = self.get_loc(state1)
        l2 = self.get_loc(state2)
        t = (l1, l2) 

        if t not in self.transitions: self.transitions[t] = []
        if t not in self.transitions_full: self.transitions_full[t] = []
        if l1 not in self.transitions_from: self.transitions_from[l1] = []
        if l1 not in self.transitions_from_full: self.transitions_from_full[l1] = []

        transition = (state1, state2, violation)

        self.transitions[t].append(violation)
        self.transitions_full[t].append(transition)
        self.transitions_from[l1].append(violation)
        self.transitions_from_full[l1].append(transition)
        self.transition_count += 1

        l = len(self.transitions_from[l1])
        s = sum(self.transitions_from[l1])

        if l > 30 and s != 0 and s != l:
            self.cells[l1[0]][l1[1]] = -1
            if l1 not in self.graphs: 
                mins = ((l1[0] * self.teil) + self.mins[0], (l1[1] * self.teil) + self.mins[1])
                maxes = (((l1[0] + 1) * self.teil) + self.mins[0], ((l1[1] + 1) * self.teil) + self.mins[1])
                self.graphs[l1] = Graph(5, mins, maxes, self.transitions_from_full[l1])
            self.graphs[l1].add_transitions(state1, state2, violation)
        elif s == 0:
            self.cells[l1[0]][l1[1]] = 0
        else:
            self.cells[l1[0]][l1[1]] = 1

    def proximity_to_nearest_unsafe_state(self, transition):
        d = 1
        s, _, _, n, d = transition
        if d == True:
            d = 0
        else:
            p = np.mean(np.array([s, n]), axis=0)
            l = self.get_loc(p)

            for i in range(1, self.dimention, 1):
                k = [self.cells[x][y] for x,y in self.get_neighburs(l, i)]
                if -1 in k or 1 in k:
                    d = i
                    break

        return float(d/self.dimention)

    def visualize(self):
        #draw_table(self.cells)
        nodes = self.get_nodes()
        draw_graph(nodes)
        draw_graph_grid(nodes, (self.dimention, self.dimention))

    def exists(self, location) -> bool:
        if location[0] >= 0 and location[0] < self.dimention and location[1] >= 0 and location[1] < self.dimention:
            return True
        return False

    def get_neighburs(self, location, radius=1) -> List[Tuple]:
        ns = []

        x, y = location
        # right and left sides
        for n in range((radius * 2) + 1):
            offset = n - radius
            l1 = (x + radius, y + offset)
            l2 = (x - radius, y + offset)
            if self.exists(l1): ns.append(l1)
            if self.exists(l2): ns.append(l2)
        
        # top and bottom sides
        for n in range((radius * 2) - 1):
            offset = n - radius + 1
            l1 = (x + offset, y + radius)
            l2 = (x + offset, y - radius)
            if self.exists(l1): ns.append(l1)
            if self.exists(l2): ns.append(l2)

        return ns

    def get_nodes(self) -> List[Node]:
        Node.index = 0
        nodes: List[Node] = []
        assigned = []
        for i in range(self.dimention):
            for j in range(self.dimention):
                l = (i, j)
                if l in assigned: continue
                assigned.append(l)
                value = self.cells[i][j]
                node = Node(value)
                nodes.append(node)
                node.add_region(Region(j, j+1, i, i+1))

                ns = self.get_neighburs(l)
                while len(ns) != 0:
                    ns = list(dict.fromkeys(ns))
                    ns = [(x,y) for x,y in ns if value == self.cells[x][y]]

                    new_ns = []
                    for (x, y) in ns:
                        if (x,y) not in assigned:
                            node.add_region(Region(y, y+1, x, x+1))
                            assigned.append((x,y))

                        neighburs = self.get_neighburs((x, y))
                        neighburs = [n for n in neighburs if n not in assigned]
                        new_ns.extend(neighburs)
                    ns = new_ns

        for n1 in nodes:
            for n2 in nodes:
                if n1 == n2: continue
                if n1.is_adjacent(n2):
                    n1.add_node(n2)
                    
        return nodes








