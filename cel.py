import cv2
import numpy as np
import random
import argparse
import os
from os import path




sobel = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]], dtype=np.float64)

parser = argparse.ArgumentParser(description='Draw some images!')
parser.add_argument("image", help="Image path to process.")
parser.add_argument("-t","--threshold", help="Threshold value for the edge detection. Default value is 25", type=int, default=25)
parser.add_argument("-b","--background", help="Set this flag to use a white background for the image.", action="store_true")
parser.add_argument("-c", "--clamp", help="Set this flag to turn of edge clamping", action="store_false")
args = parser.parse_args()

white_back = args.background
threshold = args.threshold
input_file = args.image
clamp_edges =args.clamp
output_folder = path.join("output", path.splitext(path.split(input_file)[1])[0])
print(output_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
def main():
    
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    print("casThink")
    filtered_image = cv2.bilateralFilter(img, 15, 580,580)
    cv2.imwrite(path.join(output_folder, "1filter.jpg"), filtered_image)

    edges, theta = get_edges(filtered_image, threshold)
    if white_back:
        img = np.ones_like(filtered_image)*255
    else:
        img = filtered_image.copy()
    stroked_image = add_strokes(filtered_image, edges,theta, img, clamp_edges)

    cv2.imwrite(path.join(output_folder, "2edges.jpg"), edges)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(path.join(output_folder, "3nooutline.jpg"), stroked_image)
    toon = cv2.bitwise_and(stroked_image, edges)
    cv2.imwrite(path.join(output_folder, "4out.jpg"), toon)

def add_strokes(filtered_image, edges, theta, background, clamp):
    for i in xrange(0,filtered_image.shape[0],7):
        for j in xrange(0,filtered_image.shape[1], 7):
            if random.random() > 0.8:
                continue
            c = filtered_image[i,j].astype(np.int32) + (random.randint(0,10) - 5)
            #x, y = edgesx[i,j].astype(np.uint8)/10 ,edgesy[i,j].astype(np.uint8)/10
            scale = random.randint(20,40)
            #ioffset, joffset = int(np.cos(theta[i,j]) * scale) , int(np.sin(theta[i,j]) * scale)
            ang = (random.randint(30,60)/180.) * np.pi
            ioffset, joffset = int(np.cos(ang) * scale) , int(np.sin(ang) * scale)
            i0, j0 = min(i+ioffset,filtered_image.shape[0]-1), min(j+joffset,filtered_image.shape[1]-1)
            length = int(np.hypot(i0-i,j0-j))
            iVals, jVals = np.linspace(i, i0, length), np.linspace(j, j0, length)
            vals = edges[iVals.astype(np.int), jVals.astype(np.int)]
            if np.amin(vals) == 0 and clamp:
                continue
                #print(vals)
            
            r, g, b = c[0],c[1],c[2]
            cv2.line(background, (j,i), (j0, i0),color=(r,g,b) ,lineType=cv2.CV_AA, thickness=random.randint(1,4))
    return background

def get_edges(filtered_image, min_val):
    grey = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    edgesx = cv2.filter2D(grey,-1, sobel)
    edgesy = cv2.filter2D(grey,-1, sobel.T)
    #cv2.imwrite(path.join(output_folder, "2sobelx.jpg"), edgesx)
    #cv2.imwrite(path.join(output_folder, "3sobely.jpg"), edgesy)

    G = np.hypot(edgesy, edgesx)
    theta = np.arctan2(edgesy, edgesx) + np.pi
    #cv2.imwrite(path.join(output_folder, "4G.jpg"), G)
    #cv2.imwrite(path.join(output_folder, "5theta.jpg"), theta*50 )
    _, edges = cv2.threshold(G.astype('uint8'), min_val, 255, cv2.THRESH_BINARY_INV)
    return edges, theta


main()