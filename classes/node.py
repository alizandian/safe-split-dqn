from __future__ import annotations
from typing import List, Set

class Region:
    def __init__(self, x_min, x_max, y_min, y_max, dimention, location = None, parent_dimention = None) -> None:
        self.dimention = dimention
        self.parent_dimention = parent_dimention
        self.location = location
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def extend(self, region: Region):
        if self.location == region.location:
            if self.x_min == region.x_min and self.x_max == region.x_max:
                if self.y_min == region.y_max:
                    self.y_min = region.y_min
                    return True
                elif self.y_max == region.y_min:
                    self.y_max = region.y_max
                    return True
            elif self.y_min == region.y_min and self.y_max == region.y_max:
                if self.x_min == region.x_max:
                    self.x_min = region.x_min
                    return True
                elif self.x_max == region.x_min:
                    self.x_max = region.x_max
                    return True
        return False

    def windowed_values(self, width, height):
        dx = width / self.dimention[0]
        dy = height / self.dimention[1]
        offset = (0,0)
        if self.location != None:
            offset = (self.location[0] * dx, self.location[1] * dy)
            dx = dx / self.parent_dimention[0]
            dy = dy / self.parent_dimention[1]

        x_min = offset[0] + self.x_min * dx
        x_max = offset[0] + self.x_max * dx
        y_min = offset[1] + self.y_min * dy
        y_max = offset[1] + self.y_max * dy

        return x_min, x_max, y_min, y_max


    def oriented_values(self, other:Region):
        if self.location != None:
            x_offset = self.location[0] * self.dimention[0]
            y_offset = self.location[1] * self.dimention[1]
            self_x_min = x_offset + self.x_min
            self_x_max = x_offset + self.x_max
            self_y_min = y_offset + self.y_min
            self_y_max = y_offset + self.y_max
        else:
            self_x_min = self.x_min * other.dimention[0]
            self_x_max = self.x_max * other.dimention[0] 
            self_y_min = self.y_min * other.dimention[1]
            self_y_max = self.y_max * other.dimention[1] 

        return self_x_min, self_x_max, self_y_min, self_y_max


    def __is_adjacent(x1_min, x1_max, y1_min, y1_max, x2_min, x2_max, y2_min, y2_max):
        if (((x1_max == x2_min or x1_min == x2_max) and (y2_max >= y1_min and y2_min <= y1_max)) or 
            (y1_max == y2_min or y1_min == y2_max) and (x2_max >= x1_min and x2_min <= x1_max)):
            return True
        return False

    def is_adjacent(self, region: Region):
        if self.location == region.location:
            return Region.__is_adjacent(self.x_min, self.x_max, self.y_min, self.y_max, region.x_min, region.x_max, region.y_min, region.y_max)
        else:
            return Region.__is_adjacent(*self.oriented_values(region), *region.oriented_values(self))

class Node:
    def __init__(self, value) -> None:
        self.regions: List[Region] = []
        self.nodes: Set[Node] = []
        self.value = value

    def __str__(self):
        return self.region_formula_text()

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def add_region(self, region: Region):
        for r in self.regions:
            if r.extend(region) == True:
                self.prune_regions()
                return

        self.regions.append(region)

    def is_adjacent(self, node:Node):
        for r1 in self.regions:
            for r2 in node.regions:
                if r1.is_adjacent(r2):
                    return True
        return False


    def region_formula_text(self) -> str:
        text = ""
        for region in self.regions:
            t = f"{region.x_min} < x < {region.x_max} & {region.y_min} < y < {region.y_max}"
            if len(self.regions) > 1:
                t = f"({t})"
            if len(self.regions) - 1 != self.regions.index(region):
                t += " || \n"
            text += t
        return text

    def prune_regions(self):
        extention_happened = False
        new_regions = []
        while len(self.regions) != 0:
            region = self.regions.pop()

            for r in self.regions:
                if region.extend(r) == True:
                    self.regions.remove(r)
                    extention_happened = True
                    break

            new_regions.append(region)
            
        self.regions = new_regions

        if extention_happened:
            self.prune_regions()

    def add_node(self, node: Node):
        if node not in self.nodes:
            self.nodes.append(node)

    def merge(self, node: Node) -> Node:
        self.nodes.union(node.nodes)
        self.regions.extend(node.regions)
        self.prune_regions()
        return self
