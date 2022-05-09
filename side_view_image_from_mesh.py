"""
Saving Screenshots of mesh models
"""

import pyvista as pv
import matplotlib.pyplot as plt
import os

def get_side_view(mesh_file_name, folder_path_for_sideviews):
    # Read a mesh file
    # filename = "ex_normalized.obj"
    path_to_mesh_file = "./mugs/" + mesh_file_name
    mesh = pv.read(path_to_mesh_file)
    ###############################################################################
    # Take a screenshot without creating an interactive plot window
    # using the :class:`pyvista.Plotter`:
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color="white")
    plotter.set_background(color="black")

    sideview_image_name = folder_path_for_sideviews + "/" + mesh_file_name[:-4] + ".png"
    plotter.show(screenshot=sideview_image_name, cpos="zy") #save the image, cup(zy)
    image = plotter.image
    print(image) # image is an np array
    ##############################################################################
    # The ``img`` array can be used to plot the screenshot in ``matplotlib``:
    plt.imshow(image)
    plt.axis("off")
    # plt.savefig("ex_normalized_flipped_image.png")
    plt.show()

path_to_mesh_files = "./mugs"
# main script for sideviews from mesh
folder_to_save_sideview = path_to_mesh_files + "/sideview_images"
# mesh_file = "1a0bc9ab92c915167ae33d942430658c.obj"

from pathlib import Path
Path(folder_to_save_sideview).mkdir(parents=True, exist_ok=True)

i=1
for mesh_file in os.listdir(path_to_mesh_files):
    if mesh_file.endswith(".obj"):
        print(mesh_file)
        get_side_view(mesh_file, folder_to_save_sideview)
        print(f"The {i}th model has been processed.")
        i+=1
