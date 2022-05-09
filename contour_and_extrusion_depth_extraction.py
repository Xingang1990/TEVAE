'''
Extract contour points from side view images
'''

import numpy as np
import cv2
# from matplotlib import pyplot as plt
import os
import trimesh

def get_contour_and_extrusion_depth(sideview_image_name, folder_path_for_contour_csvs):
    # read image
    path_to_sideview_image = "./sideview_images/" + sideview_image_name
    # print(path_to_sideview_image)
    side_view = cv2.imread(path_to_sideview_image)
    # plt.imshow(side_view, 'gray')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    # change to grayscale image
    side_view_gray_scale = cv2.cvtColor(side_view, cv2.COLOR_BGR2GRAY)
    #plot gray scale image
    # plt.imshow(side_view_gray_scale, 'gray')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    # produce binary image
    ret, thresh1 = cv2.threshold(side_view_gray_scale, 50, 255, cv2.THRESH_BINARY) #the value of 50 is chosen by trial-and-error
    kernel = np.ones((5, 5), np.uint8) #square image kernel used for erosion
    erosion = cv2.erode(thresh1, kernel, iterations=1) #refines all edges in the binary image
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #removing small noises and holes in the image
    # plot binary image
    # plt.imshow(closing, 'gray')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    # contour extraction with simple approximation method
    # optimized points extraction
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    closing = np.expand_dims(closing, axis=2).repeat(3, axis=2)
    for k, _ in enumerate(contours):
        closing = cv2.drawContours(closing, contours, k, (0, 230, 255), 6)
    # plot contour
    # plt.imshow(closing)
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    # process the contour
    # print(type(contours)) #list contains contour (arrays)
    contours = contours[0]
    contours_array = [] # stores points of contours
    for contour in contours:
        contours_array.append(contour[0])

    contours_array = np.array(contours_array)
    contours_array[:, 1] *= -1
    number_of_points = contours_array.shape[0]
    # print(f"The number of points is {number_of_points}") # about 400 points

    # reduce the point number to about 100
    numElems = np.round(number_of_points // 4).astype(int)
    print(f"The number of points: {numElems}")
    idx = np.round(np.linspace(0, number_of_points - 1, numElems)).astype(int)
    selected_contour = contours_array[idx]

    # scale the coordinates of x and y
    # original coordinates are pixel position which needs scale-down
    final_contour = selected_contour.astype("float64")
    # print(final_contour)

    # move bounding box center to origin
    bbox = np.min(final_contour[:, 0]), np.max(final_contour[:, 0]), np.min(final_contour[:, 1]), np.max(final_contour[:, 1])
    bbox_center = np.array(((bbox[0] + bbox[1])/2, (bbox[2] + bbox[3])/2))
    # center the bounding box (in xy plane)
    final_contour -= bbox_center

    bbox_new = np.array((np.min(final_contour[:, 0]), np.max(final_contour[:, 0]), np.min(final_contour[:, 1]), np.max(final_contour[:, 1])))
    bbox_center_new = np.array(((bbox_new[0] + bbox_new[1])/2, (bbox_new[2] + bbox_new[3])/2)).round(2)
    bbox_new_diagonal = 2*np.sqrt(bbox_new[0]**2 + bbox_new[2]**2)
    # print(f"This is the bbox_new of the contour: {bbox_new}")
    # print(f"This is the diagonal of the bbox_new: {bbox_new_diagonal}")
    print(f"New bbox center: {bbox_center_new}")

    mesh_file = "./final_models/" + sideview_image_name[:-4] + ".obj"
    print(mesh_file)
    mesh = trimesh.load(mesh_file)
    bounds = mesh.bounds
    mesh_bbox_corners = trimesh.bounds.corners(bounds)
    # print(mesh_bbox_corners)
    mesh_bbox_diagonal_xy = 2*np.sqrt((bounds[0][0])**2 + (bounds[0][1])**2)
    # print(f"This is the diagonal of the mesh bbox: {mesh_bbox_diagonal_xy}")

    # scale down to the same size as Gao's model
    scale_factor = bbox_new_diagonal/mesh_bbox_diagonal_xy
    final_contour /= scale_factor

    # store the extrusion depth in "z" axis to the last row the final contour array
    extrusion_depth = 2*np.abs(bounds[0][2]).round(3)
    final_contour_last_row = np.ones((1,2))*extrusion_depth
    final_contour = np.concatenate((final_contour, final_contour_last_row), axis=0)
    # print(final_contour)

    contour_csv_file_name = folder_path_for_contour_csvs + "/" + sideview_image_name[:-4] + ".csv"
    np.savetxt(contour_csv_file_name, final_contour, delimiter=",", fmt="%.3e")


folder_to_save_contour_csv = "./contour_csv_files"
path_to_sideview_images = "./sideview_images"

i=1
for image in os.listdir(path_to_sideview_images):
    # print(image)
    get_contour_and_extrusion_depth(image, folder_to_save_contour_csv)
    print(f"{i} images have been processed.")
    print("===========================================================")
    i+=1

