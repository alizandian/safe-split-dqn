from __future__ import annotations
from typing import List, Set

class Region:
    def __init__(self, x_min, x_max, y_min, y_max) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def extend(self, region: Region):
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


class Node:
    index = 0
    def __init__(self, v) -> None:
        self.regions: List[Region] = []
        self.nodes: Set[Node] = []
        self.value = v
        self.i = Node.index
        Node.index += 1

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
