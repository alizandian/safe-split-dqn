from __future__ import annotations
from typing import List, Tuple
from .graph import Node, Region

class Grid:

    cells: List[List[int]] = None

    def __init__(self, x = 5, y = 5) -> None:
        self.x = x
        self.y = y

        self.cells = [[0 for i in range(self.x)] for j in range(self.y)]
        self.grids = {} # a dictionary, key is location, value would be another grid 
        self.transitions = {} # a dictionary, key is a tuple of locations, originating cell and target cell, value is a list of bool, indicating violation
        self.transitions_full = {}

        # for i in range(self.x):
        #     for j in range(self.y):
        #         if (i <= 2 and j <= 2) or (i >= 7 and j >= 7):
        #             self.cells[j][i] = 1
        #         # if (i <= 2):
        #         #     self.cells[j][i] = 1

        #print(self.cells)
    def clamp(self, n, smallest, largest): return max(smallest, min(n, largest))

    def add_transitions(self, state1, state2, violation):
        """
        expecting normalized coordinates, between -1 and +1 in each axis
        violation is a bool, true if it is a system failure, end of trajectory
        """

        xl = 2.0 / self.x
        yl = 2.0 / self.y

        x1i = self.clamp(int((state1[0] + 1) / xl), 0, self.x-1)
        y1i = self.clamp(int((state1[1] + 1) / yl), 0, self.y-1)

        x2i = self.clamp(int((state2[0] + 1) / xl), 0, self.x-1)
        y2i = self.clamp(int((state2[1] + 1) / yl), 0, self.y-1)

        t = ((x1i, y1i), (x2i, y2i)) 

        if violation: self.cells[x2i][y2i] = 1

        if t not in self.transitions: self.transitions[t] = []
        if t not in self.transitions_full: self.transitions_full[t] = []
        self.transitions[t].append(violation)
        self.transitions_full[t].append((state1, state2, violation))

        if x1i == x2i and y1i == y2i:
            l = len(self.transitions[t])
            s = sum(self.transitions[t])
            

            if len(self.transitions[t]) > 50 and s != 0 and s != l:
                self.increase_resolution(x1i, y1i)

    def increase_resolution(self, x, y):
        print(f"INCREASE RESOLUTION at {x}, {y}")


    def exists(self, location) -> bool:
        if location[0] >= 0 and location[0] < self.x and location[1] >= 0 and location[1] < self.y:
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

    def grid_to_graph(self) -> List[Node]:
        nodes: List[Node] = []
        assigned = []
        for i in range(self.x):
            for j in range(self.y):
                l = (i, j)
                if l in assigned: continue
                assigned.append(l)
                value = self.cells[i][j]
                node = Node()
                nodes.append(node)
                if len(nodes) > 1:
                    previous_node = nodes[-2]
                    current_node = nodes[-1]
                    previous_node.add_node(current_node)
                    current_node.add_node(previous_node)
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
                    
        return nodes








