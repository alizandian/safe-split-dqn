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
        
        self.update()


    def calculate_conflux_force(self, transition, x, y):
        """
        TODO: Not yet dimention free
        returning a value between -1 and 1. 0
        """
        s, _, _, n, d = transition
        p = np.mean(np.array([s, n]), axis=0)
        l = self.get_loc(p)
        dx = n[0] - s[0]
        dy = n[1] - s[1]
        Dx = x - l[0]
        Dy = y - l[1]

        xforce = self.clamp(float(dx / Dx), -1, 1)
        yforce = self.clamp(float(dy / Dy), -1, 1)

        force = xforce + yforce
        return self.clamp(force, -1, 1)

    def proximity_to_unsafe_states(self, transition, max_depth = 5):
        """
        TODO: Not yet dimention free
        calculates the significance of the transition regarded to unsafe states. 1 max, 0 min
        Only considering the first layer of accuracy of the grid, (not considering increase resolutions) for performance sake

        Optimizatino potential here. Should we consider all cells or just a rough position of nodes is enough?
        """
        s, _, _, n, d = transition
        if d == True:
            return 1
        else:
            p = np.mean(np.array([s, n]), axis=0)
            l = self.get_loc(p)
            forces = []

            m = min(self.dimention, max_depth)
            for i in range(1, m, 1):
                for x,y in self.get_neighburs(l, i):
                    if self.cells[x][y] == -1 or self.cells[x][y] == 1:
                        forces.append(self.calculate_conflux_force(transition, x, y) * (m-i+1) / m)

            if len(forces) == 0:
                return 0
            else:
                return float(sum(forces)/len(forces))

    def visualize(self):
        #draw_table(self.cells)
        draw_graph(self.nodes)
        draw_graph_grid(self.nodes, (self.dimention, self.dimention))

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

    def update(self):
        """
        Updating node representation of the model.
        A lot of optimisation potential here. Perhaps increamental updates?
        """
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
                    
        self.nodes = nodes








