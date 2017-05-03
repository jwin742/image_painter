import cv2
import numpy as np


from os import path

images = "photos"


sobel = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]], dtype=np.float64)

def main():

    img = cv2.imread(path.join(images,"source", "input.jpg"), cv2.IMREAD_COLOR)
    print("casThink")
    filtered_image = cv2.bilateralFilter(img, 15, 1580,1580)
    #filtered_image = cv2.bilateralFilter(filtered_image, 10, 380,380)
    cv2.imwrite(path.join(images,"output", "1filter.jpg"), filtered_image)
    grey = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    edgesx = cv2.filter2D(grey,-1, sobel)
    edgesy = cv2.filter2D(grey,-1, sobel.T)
    cv2.imwrite(path.join(images,"output", "2sobelx.jpg"), edgesx)
    cv2.imwrite(path.join(images,"output", "3sobely.jpg"), edgesy)

    G = np.hypot(edgesy, edgesx)
    cv2.imwrite(path.join(images,"output", "4G.jpg"), G)
    theta = np.arctan(edgesy, edgesx)
    cv2.imwrite(path.join(images,"output", "5theta.jpg"), theta)
    _, edges = cv2.threshold(G.astype('uint8'), 25, 255, cv2.THRESH_BINARY_INV)
    #edges = cv2 adaptiveThreshold(G.astype('uint8'), 255, )
    cv2.imwrite(path.join(images,"output", "6edges.jpg"), edges)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    toon = cv2.bitwise_and(filtered_image, edges)
    cv2.imwrite(path.join(images,"output", "7out.jpg"), toon)
    #get gaussian
    #smoothed = cv2.GaussianBlur(filtered_image, (5,5), 0)
    #cv2.imwrite(path.join(images,"output", "gauss.jpg"), smoothed)
    #smoothed = cv2.medianBlur(filtered_image, 7)
    #cv2.imwrite(path.join(images,"output", "median.jpg"), smoothed)

main()