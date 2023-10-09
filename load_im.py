# load, split and scale the maps dataset ready for training
'''from os import listdir
from numpy import asarray
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

# load all images in a directory into memory
def load_images(path, size=(256,256)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
		src_list.append(sat_img)
		tar_list.append(map_img)
	return [asarray(src_list), asarray(tar_list)]

def LoadImages(path, size = (256, 256)):
    src_list = list()
    for filename in listdir(path):
        pixels = load_img(path + filename, target_size=size)
        pixels = img_to_array(pixels)
        src_list.append(pixels)
    return asarray(src_list)

size = (1024, 1024)

# dataset path
path_train = 'Train_Images/Aerial/'
# load dataset
src_images = LoadImages(path_train, size)
print('Loaded: ', src_images.shape)
# save as compressed numpy array
#filename = 'train.npz'
#savez_compressed(filename, src_images)
#print('Saved dataset: ', filename)

path_output = 'Output_Images/Aerial/'
src_images2 = LoadImages(path_output, size)

print('Loaded: ', src_images2.shape)
# save as compressed numpy array
filename = 'map_test.npz'
savez_compressed(filename, src_images, src_images2)
print('Saved dataset: ', filename)
'''

from matplotlib import pyplot
from os import listdir
from numpy import asarray
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

size = (256, 256)

def LoadImages(path, size = (256, 256)):
    src_list = list()
    for filename in listdir(path):
        pixels = load_img(path + filename, target_size=size)
        pixels = img_to_array(pixels)
        src_list.append(pixels)
    return asarray(src_list)

def PlotImages(images, save = False):
    pyplot.figure()
    for i in range(9):
        pyplot.subplot(3, 3, i+1)
        pyplot.axis('off')
        pyplot.imshow(images[i].astype('uint8'))
    if save:
        pyplot.savefig("data_explore.png", dpi = 600)
        pyplot.close()
            
path_train = '../Masking_Images_Progress/'
src_images = LoadImages(path_train, size)
print('Loaded: ', src_images.shape)
PlotImages(src_images, True)

#path_train = 'Output_Images/Aerial/'
#src_images2 = LoadImages(path_train, size)
#print('Loaded: ', src_images2.shape)
#PlotImages(src_images2)

