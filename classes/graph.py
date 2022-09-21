from __future__ import annotations
from typing import List, Tuple, Dict
from .node import Node, Region
from classes.visualization import draw_graph, draw_graph_grid
import numpy as np

class Graph:
    cells: List[List[int]] = None

    def __init__(self, actions_count, dimention, mins, maxes, parent: Graph = None, location = None, experiences = None) -> None:
        self.parent = parent if parent != None else self
        self.dimention = dimention
        self.parent_dimention = parent.dimention
        self.location = location
        self.mins = mins
        self.maxes = maxes
        self.len = (maxes[0] - mins[0], maxes[1] - mins[1])
        self.teil = (self.len[0] / dimention[0], self.len[1] / dimention[1])
        self.nodes: List[Node] = []
        self.experience_count = 0
        self.actions_count = actions_count
        self.origins = {}
        self.targets = {}
        self.experiences = [[[[] for _ in range(actions_count)] for _ in range(dimention[0])] for _ in range(dimention[1])]
        self.cells = [[[0 for _ in range(actions_count)] for _ in range(dimention[0])] for _ in range(dimention[1])]
        self.graphs: Dict[Tuple, Graph] = {} 

        if experiences != None:
            for t in experiences:
                self.add_experience(t)

    def clamp(self, n, smallest, largest): return max(smallest, min(n, largest))

    def get_safe_actions(self, state) -> list:
        location = self.get_loc(state)
        if location in self.graphs:
            return self.graphs[location].get_safe_actions(state)
        else:
            safe_actions = []
            for a, value in enumerate(self.cells[location[0]][location[1]]):
                if value != -1:
                    safe_actions.append(a)
            return safe_actions

    def feed_neural_network_feedback(self, values):
        unsafe_q_values = []
        for y in range(self.dimention[1]):
            for x in range(self.dimention[0]):
                for a in range(self.actions_count):
                    if self.cells[x][y][a] == -1:
                        unsafe_q_values.append(values[self.dimention[1]-y-1][x][a])

        if len(unsafe_q_values) == 0:
            return

        mean = np.mean(unsafe_q_values)
        min = np.min(values)
        max = np.max(values)

        if min > -0.8:
            return

        PERCENT = 0.01

        d = (mean - min) / (max - min)
        dmax = self.clamp(d + (d * PERCENT), 0, 1)
        dmin = self.clamp(d - (d * PERCENT), 0 ,1)

        new_min = min + ((max - min) * dmin)
        new_max = min + ((max - min) * dmax)

        for y in range(self.dimention[1]):
            for x in range(self.dimention[0]):
                for a in range(self.actions_count):
                    v = values[self.dimention[1]-y-1][x][a]
                    if v >= new_min and v <= new_max:
                        if self.cells[x][y][a] == 0:
                            self.cells[x][y][a] = -1
                            print("override")

    def get_loc(self, state):
        if self.location != None:
            if state[0] < self.mins[0] or state[0] > self.maxes[0] or state[1] < self.mins[1] or state[1] > self.maxes[1]:
                None

        x = self.clamp(int((state[0] - self.mins[0]) / self.len[0] * self.dimention[0]), 0, self.dimention[0]-1)
        y = self.clamp(int((state[1] - self.mins[1]) / self.len[1] * self.dimention[1]), 0, self.dimention[1]-1)

        return (x, y)

    def is_safe(self, location):
        safe_counts = 0
        unsafe_counts = 0
        unsure_counts = 0
        counts = 0
        for a in self.cells[location[0]][location[1]]:
            counts += 1
            if a == -1: unsafe_counts += 1
            elif a == 0: unsure_counts += 1
            elif a == 1: safe_counts += 1

        if safe_counts > 0:
            return 1
        elif unsafe_counts == counts:
            return -1
        else:
            return 0

    def update_location(self, location):
        if location in self.origins:
            for state, action, _, _, _ in self.origins[location]:
                self.parent.evaluate_state(state, action)

    def evaluate_state(self, state, action):
        l = self.get_loc(state)
        if l in self.graphs:
            l = self.graphs[l].get_loc(state)
            self.graphs[l].evaluate_location(l, action)
        else:
            self.evaluate_location(l, action)

    def evaluate_location(self, location, action):
        safe_counts = 0
        unsafe_counts = 0
        unsure_counts = 0
        for _, a, _, next_state, _ in self.targets[location]:
            if action == a:
                s = self.parent.is_state_safe(next_state)
                if s == -1: unsafe_counts += 1
                elif s == 1: safe_counts += 1
                else: unsure_counts += 1

        pv = self.is_safe(location)
        if self.cells[location[0]][location[1]][action] != -1:
            if unsafe_counts > 0:
                self.cells[location[0]][location[1]][action] = -1
            else:
                self.cells[location[0]][location[1]][action] = 1

        if pv != self.is_safe(location): self.update_location(location)
        
    def get_unsafe_bound_experiences(self):
        exs = []
        for y in range(self.dimention[1]):
            for x in range(self.dimention[0]): 
                if self.is_safe((x, y)) == 1:
                    for a in range(self.actions_count):
                        if self.cells[x][y][a] == -1:
                            unsafes = [(s,a,r,n,v) for s,a,r,n,v in self.refine_experiences(self.experiences[x][y][a]) if v == 1]
                            if len(unsafes) > 0:
                                exs.append(unsafes[-1])
        return exs

    def add_experience(self, experience):
        xp = self.refine_experiences([experience])[0]
        state, action, _, next_state, violation = xp
        origin = self.get_loc(state)
        target = self.get_loc(next_state)

        if target != None:
            if target not in self.origins: self.origins[target] = []
            self.origins[target].append(xp)
            if target in self.graphs: self.graphs[target].add_experience(xp)

        if origin != None: 
            self.experiences[origin[0]][origin[1]][action].append(xp)
            self.experience_count += 1
            if origin not in self.targets: self.targets[origin] = []
            self.targets[origin].append(xp)
            if origin in self.graphs: 
                self.graphs[origin].add_experience(xp)
            else:   
                xps = self.experiences[origin[0]][origin[1]][action]
                violating_xps = [x for x in xps if x[4] == 1]
                c = len(xps)
                v = len(violating_xps)
                if c > 2 and v != c and v != 0 and self.parent_dimention == None:
                    mins = ((origin[0] * self.teil[0]) + self.mins[0], (origin[1] * self.teil[1]) + self.mins[1])
                    maxes = (((origin[0] + 1) * self.teil[0]) + self.mins[0], ((origin[1] + 1) * self.teil[1]) + self.mins[1])
                    self.graphs[origin] = Graph(self.actions_count, (6,6), mins, maxes, self, origin, self.origins[origin] + self.targets[origin])

                pv = self.is_safe(origin)
                if self.cells[origin[0]][origin[1]][action] == -1:
                    if len(self.experiences[origin[0]][origin[1]][action]) == 1:
                        if violation == 0:
                            self.cells[origin[0]][origin[1]][action] = 1
                            if pv != self.is_safe(origin): self.update_location(origin)
                elif violation == 1:
                    self.cells[origin[0]][origin[1]][action] = -1
                    if pv != self.is_safe(origin): self.update_location(origin)
                else:
                    self.evaluate_location(origin, action)

    def calculate_conflux_force(self, experience, x, y):
        """
        TODO: Not yet dimention free
        returning a value between -1 and 1. 0
        """
        s, _, _, n, d = experience
        p = np.mean(np.array([s, n]), axis=0)
        l = self.get_loc(p)
        dx = n[0] - s[0]
        dy = n[1] - s[1]
        Dx = x - l[0]
        Dy = y - l[1]


        if Dx == 0: Dx = dx
        if Dy == 0: Dy = dx
        xforce = self.clamp(float(dx / Dx), -1, 1)
        yforce = self.clamp(float(dy / Dy), -1, 1)

        force = xforce + yforce
        return self.clamp(force, -1, 1)

    def get_action_value(self, state, action):
        l = self.get_loc(state)
        if l in self.graphs:
            return self.graphs[l].get_action_value(state, action)
        else:
            return self.cells[l[0]][l[1]][action]

    def is_state_safe(self, state):
        l = self.get_loc(state)
        if l in self.graphs:
            return self.graphs[l].is_state_safe(state)
        else:
            return self.is_safe(l)

    def refine_experiences(self, experiences):
        refined_experiences = []

        for s, a, r, n, d in experiences:
            reward = r
            done = d
            if d == True:
                reward = -1
            elif self.get_action_value(s, a) == -1:
                done = 1
                reward = -1
            elif self.is_state_safe(n) == -1:
                done = 1
                reward = -1
            else:
                reward = 1
            
            refined_experiences.append((s, a, reward, n, done))

        return refined_experiences

    def visualize(self):
        draw_graph(self.nodes)
        draw_graph_grid(self.nodes)

    def exists(self, location) -> bool:
        if location[0] >= 0 and location[0] < self.dimention[0] and location[1] >= 0 and location[1] < self.dimention[1]:
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
        for j in range(self.dimention[1]):
            for i in range(self.dimention[0]):
                l = (i, j)
                if l in assigned: continue
                assigned.append(l)
                if l in self.graphs:
                    self.graphs[l].update()
                    nodes.extend(self.graphs[l].nodes)
                    continue
                is_safe = self.is_safe(l)
                node = Node(is_safe)
                nodes.append(node)
                node.add_region(Region(i, i+1, j, j+1, self.dimention, self.location, self.parent_dimention))

                ns = self.get_neighburs(l)
                while len(ns) != 0:
                    ns = list(dict.fromkeys(ns))
                    ns = [(x,y) for x,y in ns if is_safe == self.is_safe((x,y))]

                    new_ns = []
                    for (x, y) in ns:
                        if (x,y) not in assigned and (x,y) not in self.graphs:
                            node.add_region(Region(x, x+1, y, y+1, self.dimention, self.location, self.parent_dimention))
                            assigned.append((x,y))

                        neighburs = self.get_neighburs((x, y))
                        neighburs = [n for n in neighburs if n not in assigned]
                        new_ns.extend(neighburs)
                    ns = new_ns

        merged = False
        while True:
            if not merged: break
            out = []
            to_be_checked: List[Node] = []
            to_be_checked.extend(nodes)

            while len(to_be_checked) != 0:
                n = to_be_checked.pop()
                m = []
                for nn in to_be_checked:
                    if n.is_adjacent(nn):
                        if n.value == nn.value:
                            n = n.merge(nn)
                            m.append(nn)
                            merged = True

                out.append(n)
                to_be_checked = [t for t in to_be_checked if t not in m]
            
            nodes = out

        for n1 in nodes:
            for n2 in nodes:
                if n1 == n2: continue
                if n1.is_adjacent(n2):
                    n1.add_node(n2)
                    
        self.nodes = nodes