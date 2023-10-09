import cv2

fileName = "../Imagini_Input/Train_Images/DSC05043r.jpg"

black_white = 0
red_green_blue = 1

def readImage(name, colorType):
    img = cv2.imread(name, colorType)
    return img

def writeImage(fileName, file):
    cv2.imwrite(fileName, file)

img = readImage(fileName, black_white)

prag_1 = 50
prag_2 = 100
prag_3 = 150
prag_4 = 200
prag_5 = 255

range_i, range_j = img.shape

for i in range(range_i):
    for j in range(range_j):
        if img[i][j] >= 0 and img[i][j] < prag_1:
            img[i][j] = 25
        if img[i][j] >= prag_1 and img[i][j] < prag_2:
            img[i][j] = 75
        if img[i][j] >= prag_2 and img[i][j] < prag_3:
            img[i][j] = 125
        if img[i][j] >= prag_3 and img[i][j] < prag_4:
            img[i][j] = 175
        if img[i][j] >= prag_4 and img[i][j] < prag_5:
            img[i][j] = 225
        
writeImage("binary_mask.bmp", img)


'''
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
fileName = "../Imagini_Input/Train_Images/DSC05043r.jpg"

pic = plt.imread(fileName)/255

pic_reshape = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])

kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_reshape)

output_img = kmeans.cluster_centers_[kmeans.labels_]

cluster_pic = output_img.reshape(pic.shape[0], pic.shape[1], pic.shape[2])

plt.imshow(cluster_pic)

plt.savefig("cluster_image.png", dpi = 300)

'''