import tkinter
from typing import List
from .graph import Node

def draw_table(coord: List[List[int]] = [[1,0,0,1], [1,1,1,1], [1,0,0,1]]):
    root = tkinter.Tk()
    min_x, min_y = 0, 0
    max_x, max_y = 500, 500
    canvas = tkinter.Canvas(root, bg="white", height=max_x, width=max_y)

    for y, rows in enumerate(coord):
        for x, column in enumerate(rows):
            lx = max_x - min_x
            ly = max_y - min_y
            x_unit = lx / len(rows)
            y_unit = ly / len(coord)
            color = "black" if column == 1 else "yellow"
            canvas.create_rectangle(x * x_unit, y * y_unit, (x * x_unit) + x_unit, ( y * y_unit) + y_unit, outline="#fb0", fill=color)

    canvas.pack()
    root.mainloop()

def draw_graph(nodes: List[Node]):
    root = tkinter.Tk()
    min_x, min_y = 0, 0
    max_x, max_y = 500, 500
    node_size = 20
    canvas = tkinter.Canvas(root, bg="white", height=max_x, width=max_y)
    
    lx = max_x - min_x
    ly = max_y - min_y
    x = lx / 3
    y_unit = ly / (len(nodes) + 1)


    node_positions = {}
    for i, node in enumerate(nodes):
        node_positions[node] = (x, y_unit * (i+1))
        canvas.create_oval(x, y_unit * (i+1), x + node_size, y_unit * (i+1) + node_size, fill="blue")
        canvas.create_text(x * 2, y_unit * (i+1), fill="darkblue",font="Times 20 italic bold", text=node.region_formula_text())

    for i, node in enumerate(nodes):
        for n in node.nodes:
            coord1 = node_positions[node]
            coord2 = node_positions[n]
            canvas.create_line(coord1[0] - (5 + i*5), coord1[1], coord2[0] - (5 + i*5), coord2[1])


    canvas.pack()
    root.mainloop()
