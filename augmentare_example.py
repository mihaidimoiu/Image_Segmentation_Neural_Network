from imgaug import augmenters as iaa
from imgaug import parameters as iap
from os import listdir
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot
import numpy as np 

def LoadImages(path, size = (256, 256)):
    src_list = list()
    for filename in listdir(path):
        pixels = load_img(path + filename, target_size=size)
        pixels = img_to_array(pixels)
        src_list.append(pixels.astype(np.uint8))
    return src_list

def PlotImages(images, save = False):
    for i in range(len(images)):
        pyplot.subplot(3, 5, i+1)
        pyplot.axis('off')
        pyplot.imshow(images[i].astype('uint8'))
    if save:
        pyplot.savefig("testn.png", dpi = 1000)
        pyplot.close()


flip_oy_seq = iaa.Sequential([
    iaa.Fliplr(1),
])

flip_ox_seq = iaa.Sequential([
    iaa.Flipud(1),
])

filp_oy_rotate90 = iaa.Sequential([
    iaa.Fliplr(1),
    iaa.Affine(rotate=(90)),
])

scale = iaa.Sequential([
    iaa.Affine(scale=(1.5))
])


rotate90 = iaa.Sequential([
    iaa.Affine(rotate=(90))
])

perspective = iaa.Sequential([
    iaa.PerspectiveTransform(scale=(0.1))
])

rotateBlur90 = iaa.Sequential([
    iaa.pillike.FilterBlur(),
    iaa.Affine(rotate=(90))
])

filterSmooth = iaa.Sequential([
    iaa.pillike.FilterSmoothMore()
])

rotateSharpen90 = iaa.Sequential([
    iaa.pillike.FilterSharpen(),
    iaa.Affine(rotate=(90))
])

clouds = iaa.Sequential([
    iaa.Clouds()
])


fog = iaa.Sequential([
    iaa.Fog()
])


rain = iaa.Sequential([
    iaa.Rain(speed=(0.2))
])


rotateGauss90 = iaa.Sequential([
    iaa.GaussianBlur(sigma=1.5),
    iaa.Affine(rotate=(90))
])

rotateFog270 = iaa.Sequential([
    iaa.Fog(),
    iaa.Affine(rotate=(270))
])

color_enhance = iaa.Sequential([
    iaa.pillike.EnhanceColor()
])


size = (256, 256)
path_train = 'Train_Images/Aerial/'

src_images2 = LoadImages(path_train, size)
src_images = [src_images2[0]]

#src_images = LoadImages(path_train, size)
###################################################

images_aug = flip_oy_seq(images=src_images)
output = src_images + images_aug

###################################################

images_aug = flip_ox_seq(images=src_images)
output = output + images_aug
###################################################

images_aug = scale(images=src_images)
output = output + images_aug
###################################################

images_aug = rotate90(images=src_images)
output = output + images_aug
###################################################

images_aug = perspective(images=src_images)
output = output + images_aug
###################################################

images_aug = filterSmooth(images=src_images)
output = output + images_aug
###################################################

images_aug = clouds(images=src_images)
output = output + images_aug
###################################################

images_aug = fog(images=src_images)
output = output + images_aug
###################################################

images_aug = rain(images=src_images)
output = output + images_aug
###################################################

images_aug = filp_oy_rotate90(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateGauss90(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateSharpen90(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateFog270(images=src_images)
output = output + images_aug
###################################################

images_aug = color_enhance(images=src_images)
output = output + images_aug
###################################################

print(len(output))

PlotImages(np.asarray(output), True)
