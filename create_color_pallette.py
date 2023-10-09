import numpy as np
from matplotlib import pyplot

def create_blank(size = (100, 100, 3), rgb_color=(0, 0, 0)):  
    image = np.zeros(size, np.uint8)   
    image[:] = rgb_color
    return image

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

size = (50, 50, 3)

hex_colors = {
    "#be601f": "apa inundatie",
    "#087715": "copaci verzi + vegetatie",
    "#633b08": "pamant",
    "#ff0000": "case",
    "#929292": "alee ciment",
    "#c6c6c6": "asfalt drum",
    "#fff600": "masini",
    "#515151": "drum de pamant",
    "#594a39": "alee de pamant"  
    }

color_position = 1
for key, value in hex_colors.items():
    pyplot.subplot(3, 3, color_position)
    pyplot.axis("off")
    pyplot.title(value, y=-0.23)
    pyplot.imshow(create_blank(size, hex_to_rgb(key)))
    color_position += 1

filename = "colors_plot.png"
pyplot.savefig(filename)
