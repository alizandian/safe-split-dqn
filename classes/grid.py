from typing import List, Tuple
from .graph import Node, Region

class Grid:

    cells: List[List[int]] = None

    def __init__(self, x = 10, y = 10) -> None:
        self.x = x
        self.y = y

        self.cells = [[0 for i in range(self.x)] for j in range(self.y)]

        for i in range(self.x):
            for j in range(self.y):
                if (i <= 2 and j <= 2) or (i >= 7 and j >= 7):
                    self.cells[j][i] = 1
                # if (i <= 2):
                #     self.cells[j][i] = 1

        #print(self.cells)

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








