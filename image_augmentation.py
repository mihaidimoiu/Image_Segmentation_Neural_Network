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
    for i in range(len(images) - 1):
        pyplot.subplot(50, 10, i+1)
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

filp_ox_rotate90 = iaa.Sequential([
    iaa.Flipud(1),
    iaa.Affine(rotate=(90)),
])

flip_oxoy_seq = iaa.Sequential([
    iaa.Fliplr(1),
    iaa.Flipud(1),
])

flip_oxoy_seq_rotate90 = iaa.Sequential([
    iaa.Fliplr(1),
    iaa.Flipud(1),
    iaa.Affine(rotate=(90)),
])

scaleFlip_ox = iaa.Sequential([
    iaa.Affine(scale=(1.5)),
    iaa.Flipud(1),
])

scaleFlip_oy = iaa.Sequential([
    iaa.Affine(scale=(1.5)),
    iaa.Fliplr(1),
])

scale = iaa.Sequential([
    iaa.Affine(scale=(1.5))
])

scaleRotate90 = iaa.Sequential([
    iaa.Affine(scale=(1.2)),
    iaa.Affine(rotate=(90))
])

scaleRotate180 = iaa.Sequential([
    iaa.Affine(scale=(1.3)),
    iaa.Affine(rotate=(180))
])

scaleRotate270 = iaa.Sequential([
    iaa.Affine(scale=(1.4)),
    iaa.Affine(rotate=(270))
])

rotate90 = iaa.Sequential([
    iaa.Affine(rotate=(90))
])

rotate180 = iaa.Sequential([
    iaa.Affine(rotate=(180))
])


rotate270 = iaa.Sequential([
    iaa.Affine(rotate=(270))
])

perspective = iaa.Sequential([
    iaa.PerspectiveTransform(scale=(0.1))
])


color_enhance = iaa.Sequential([
    iaa.pillike.EnhanceColor()
])

brightness_enhance = iaa.Sequential([
    iaa.pillike.EnhanceBrightness()
])

filterBlur = iaa.Sequential([
    iaa.pillike.FilterBlur()
])

rotateBlur90 = iaa.Sequential([
    iaa.pillike.FilterBlur(),
    iaa.Affine(rotate=(90))
])

rotateBlur180 = iaa.Sequential([
    iaa.pillike.FilterBlur(),
    iaa.Affine(rotate=(180))
])

rotateBlur270 = iaa.Sequential([
    iaa.pillike.FilterBlur(),
    iaa.Affine(rotate=(270))
])

filterSmooth = iaa.Sequential([
    iaa.pillike.FilterSmoothMore()
])

rotateSmooth90 = iaa.Sequential([
    iaa.pillike.FilterSmoothMore(),
    iaa.Affine(rotate=(90))
])

rotateSmooth180 = iaa.Sequential([
    iaa.pillike.FilterSmoothMore(),
    iaa.Affine(rotate=(180))
])

rotateSmooth270 = iaa.Sequential([
    iaa.pillike.FilterSmoothMore(),
    iaa.Affine(rotate=(270))
])

filterSharpen = iaa.Sequential([
    iaa.pillike.FilterSharpen()
])

rotateSharpen90 = iaa.Sequential([
    iaa.pillike.FilterSharpen(),
    iaa.Affine(rotate=(90))
])

rotateSharpen180 = iaa.Sequential([
    iaa.pillike.FilterSharpen(),
    iaa.Affine(rotate=(180))
])

rotateSharpen270 = iaa.Sequential([
    iaa.pillike.FilterSharpen(),
    iaa.Affine(rotate=(270))
])

filterDetail = iaa.Sequential([
    iaa.pillike.FilterDetail()
])

clouds = iaa.Sequential([
    iaa.Clouds()
])

rotateClouds90 = iaa.Sequential([
    iaa.Clouds(),
    iaa.Affine(rotate=(90))
])

rotateClouds180 = iaa.Sequential([
    iaa.Clouds(),
    iaa.Affine(rotate=(180))
])

rotateClouds270 = iaa.Sequential([
    iaa.Clouds(),
    iaa.Affine(rotate=(270))
])

fog = iaa.Sequential([
    iaa.Fog()
])

rotateFog90 = iaa.Sequential([
    iaa.Fog(),
    iaa.Affine(rotate=(90))
])

rotateFog180 = iaa.Sequential([
    iaa.Fog(),
    iaa.Affine(rotate=(180))
])

rotateFog270 = iaa.Sequential([
    iaa.Fog(),
    iaa.Affine(rotate=(270))
])

rain = iaa.Sequential([
    iaa.Rain(speed=(0.2))
])

adaptiveGaussian = iaa.Sequential([
    iaa.GaussianBlur(sigma=1.5)
])

rotateGauss90 = iaa.Sequential([
    iaa.GaussianBlur(sigma=1.5),
    iaa.Affine(rotate=(90))
])

rotateGauss180 = iaa.Sequential([
    iaa.GaussianBlur(sigma=1.5),
    iaa.Affine(rotate=(180))
])

rotateGauss270 = iaa.Sequential([
    iaa.GaussianBlur(sigma=1.5),
    iaa.Affine(rotate=(270))
])




size = (256, 256)
path_train = 'Train_Images_Patch/Aerial/'

src_images = LoadImages(path_train, size)
###################################################

images_aug = flip_oy_seq(images=src_images)
output = src_images + images_aug
###################################################

images_aug = flip_ox_seq(images=src_images)
output = output + images_aug
###################################################

images_aug = flip_oxoy_seq(images=src_images)
output = output + images_aug
###################################################

images_aug = scale(images=src_images)
output = output + images_aug
###################################################

images_aug = rotate90(images=src_images)
output = output + images_aug
###################################################

images_aug = rotate270(images=src_images)
output = output + images_aug
###################################################

images_aug = perspective(images=src_images)
output = output + images_aug
###################################################

images_aug = color_enhance(images=src_images)
output = output + images_aug
###################################################

images_aug = brightness_enhance(images=src_images)
output = output + images_aug
###################################################

images_aug = filterBlur(images=src_images)
output = output + images_aug
###################################################

images_aug = filterSmooth(images=src_images)
output = output + images_aug
###################################################

images_aug = filterSharpen(images=src_images)
output = output + images_aug
###################################################

images_aug = filterDetail(images=src_images)
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


images_aug = adaptiveGaussian(images=src_images)
output = output + images_aug
###################################################

images_aug = filp_oy_rotate90(images=src_images)
output = output + images_aug
###################################################

images_aug = filp_ox_rotate90(images=src_images)
output = output + images_aug
###################################################

images_aug = flip_oxoy_seq_rotate90(images=src_images)
output = output + images_aug
###################################################

images_aug = scaleFlip_ox(images=src_images)
output = output + images_aug
###################################################

images_aug = scaleFlip_oy(images=src_images)
output = output + images_aug
###################################################

images_aug = scaleRotate90(images=src_images)
output = output + images_aug
###################################################

images_aug = scaleRotate180(images=src_images)
output = output + images_aug
###################################################

images_aug = scaleRotate270(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateGauss90(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateGauss180(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateGauss270(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateFog90(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateFog180(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateFog270(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateClouds90(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateClouds180(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateClouds270(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateSharpen90(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateSharpen180(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateSharpen270(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateSmooth90(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateSmooth180(images=src_images)
output = output + images_aug
###################################################

images_aug = rotateSmooth270(images=src_images)
output = output + images_aug
###################################################

print(len(output))

#PlotImages(np.asarray(output), True)










path_train = 'Output_Images_Patch/Aerial/'

src_images = LoadImages(path_train, size)
###################################################

images_aug = flip_oy_seq(images=src_images)
output2 = src_images + images_aug
###################################################

images_aug = flip_ox_seq(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = flip_oxoy_seq(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = scale(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate90(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate270(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = perspective(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = color_enhance(images=src_images)
output2 = output2 + src_images
###################################################

images_aug = brightness_enhance(images=src_images)
output2 = output2 + src_images
###################################################

images_aug = filterBlur(images=src_images)
output2 = output2 + src_images
###################################################

images_aug = filterSmooth(images=src_images)
output2 = output2 + src_images
###################################################

images_aug = filterSharpen(images=src_images)
output2 = output2 + src_images
###################################################

images_aug = filterDetail(images=src_images)
output2 = output2 + src_images
###################################################

images_aug = clouds(images=src_images)
output2 = output2 + src_images
###################################################

images_aug = fog(images=src_images)
output2 = output2 + src_images
###################################################

images_aug = rain(images=src_images)
output2 = output2 + src_images
###################################################


images_aug = adaptiveGaussian(images=src_images)
output2 = output2 + src_images
###################################################

images_aug = filp_oy_rotate90(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = filp_ox_rotate90(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = flip_oxoy_seq_rotate90(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = scaleFlip_ox(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = scaleFlip_oy(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = scaleRotate90(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = scaleRotate180(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = scaleRotate270(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate90(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate180(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate270(images=src_images)
output2 = output2 + images_aug
###################################################


images_aug = rotate90(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate180(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate270(images=src_images)
output2 = output2 + images_aug
###################################################


images_aug = rotate90(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate180(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate270(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate90(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate180(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate270(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate90(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate180(images=src_images)
output2 = output2 + images_aug
###################################################

images_aug = rotate270(images=src_images)
output2 = output2 + images_aug
###################################################

print(len(output2))

from numpy import savez_compressed
filename = 'map_test_256_patch.npz'
savez_compressed(filename, np.asarray(output),
                 np.asarray(output2))
