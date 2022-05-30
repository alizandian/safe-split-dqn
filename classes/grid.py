from typing import List

class Grid:

    cells: List[List[int]] = None

    x = 5
    y = 5

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

        cells = [[0 for j in range(self.x)] for i in range(self.y)]