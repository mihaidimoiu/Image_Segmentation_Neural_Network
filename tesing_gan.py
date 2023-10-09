# example of loading a pix2pix model and using it for one-off image translation
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import expand_dims
from matplotlib import pyplot
import image_slicer
import numpy as np
from keras.models import Model

def load_image(filename, size=(256,256)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels

def predict_reverse_image_256(filename = None, model = None, save = False):
    if filename == None:
        print("No file to predict")
        return
    
    if model == None:
        model = load_model('256x256_reverse/model_030000.h5')
    
    # Load image from disk
    image = load_image(filename, (256, 256))
    print("Image shape: ", (image.shape))
    
    # Predict image
    gen_image = model.predict(image)
    
    # Scale from [-1,1] to [0,1]
    gen_image = (gen_image + 1) / 2.0
    # Plot the image
    pyplot.imshow(gen_image[0])
    pyplot.axis('off')
    if save:
        pyplot.savefig("predicted_image.png")
    pyplot.show()
    return gen_image[0]

def predict_image_batch4(filename = None, model = None, save = False):
    if filename == None:
        print("No file to predict")
        return
    
    if model == None:
        model = load_model('256x256_4batch/model_010000.h5')
    
    # Load image from disk
    image = load_image(filename, (256, 256))
    print("Image shape: ", (image.shape))
    
    # Predict image
    gen_image = model.predict(image)
    
    # Scale from [-1,1] to [0,1]
    gen_image = (gen_image + 1) / 2.0
    # Plot the image
    pyplot.imshow(gen_image[0])
    pyplot.axis('off')
    if save:
        pyplot.savefig("predicted_image.png")
    pyplot.show()
    return gen_image[0]

def predict_image_batch16(filename = None, model = None, save = False):
    if filename == None:
        print("No file to predict")
        return
    
    if model == None:
        model = load_model('256x256_16batch/model_020000.h5')
    
    # Load image from disk
    image = load_image(filename, (256, 256))
    print("Image shape: ", (image.shape))
    
    # Predict image
    gen_image = model.predict(image)
    
    # Scale from [-1,1] to [0,1]
    gen_image = (gen_image + 1) / 2.0
    # Plot the image
    pyplot.imshow(gen_image[0])
    pyplot.axis('off')
    if save:
        pyplot.savefig("predicted_image.png")
    pyplot.show()
    return gen_image[0]

def predict_image_1layer_batch4(filename = None, model = None, save = False):
    if filename == None:
        print("No file to predict")
        return
    
    if model == None:
        model = load_model('512x512_1layer_4batch/model_010000.h5')
    
    # Load image from disk
    image = load_image(filename, (512, 512))
    print("Image shape: ", (image.shape))
    
    # Predict image
    gen_image = model.predict(image)
    
    # Scale from [-1,1] to [0,1]
    gen_image = (gen_image + 1) / 2.0
    # Plot the image
    pyplot.imshow(gen_image[0])
    pyplot.axis('off')
    if save:
        pyplot.savefig("predicted_image.png")
    pyplot.show()
    return gen_image[0]

def predict_single_image_256(filename = None, model = None, save = False):
    if filename == None:
        print("No file to predict")
        return
    
    if model == None:
        model = load_model('256x256_30epoci/model_030000.h5')
    
    # Load image from disk
    image = load_image(filename, (256, 256))
    print("Image shape: ", (image.shape))
    
    # Predict image
    gen_image = model.predict(image)
    
    # Scale from [-1,1] to [0,1]
    gen_image = (gen_image + 1) / 2.0
    # Plot the image
    pyplot.imshow(gen_image[0])
    pyplot.axis('off')
    if save:
        pyplot.savefig("predicted_image.png")
    pyplot.show()
    return gen_image[0]

def predict_single_image_512(filename = None, model = None, save = False):
    if filename == None:
        print("No image to predict")
        return
    
    if model == None:
        model = load_model('512x512_30epoci/model_030000.h5')
    
    # Load image from disk
    image = load_image(filename, (512, 512))
    print("Image shape: ", (image.shape))
    
    # Predict image
    gen_image = model.predict(image)
    
    # Scale from [-1,1] to [0,1]
    gen_image = (gen_image + 1) / 2.0
    # Plot the image
    pyplot.imshow(gen_image[0])
    pyplot.axis('off')
    if save:
        pyplot.savefig("predicted_image.png")
    pyplot.show()
    return gen_image[0]

def predict_single_image_512_plus_one_layer(filename = None, model = None, save = False):
    if filename == None:
        print("No image to predict")
        return
    
    if model == None:
        model = load_model('512x512_30epoci_plus_1_layer/model_030000.h5')
    
    # Load image from disk
    image = load_image(filename, (512, 512))
    print("Image shape: ", (image.shape))
    
    # Predict image
    gen_image = model.predict(image)
    
    # Scale from [-1,1] to [0,1]
    gen_image = (gen_image + 1) / 2.0
    # Plot the image
    pyplot.imshow(gen_image[0])
    pyplot.axis('off')
    if save:
        pyplot.savefig("predicted_image.png")
    pyplot.show()
    return gen_image[0]

def predict_patch_image_256(filename = None, model = None, save = False):
    
    if filename == None:
        print("No image to predict")
        return
    
    if model == None:
        model = load_model('256x256_30epoci_patch/model_030000.h5')

    src_images = image_slicer.slice(filename, 8, save = False)
    
    resized_images = []
    for img in src_images:
        temp = img.image
        temp_resize = temp.resize((256, 256))
        temp_resize = np.asarray(temp_resize)
        temp_resize = np.expand_dims(temp_resize, axis = 0)
        temp_resize = (temp_resize - 127.5) /127.5
        resized_images.append(np.asarray(temp_resize))
    
    gen_images = []
    for img in resized_images:
        temp = model.predict(img)
        temp = (temp+ + 1) / 2.0
        temp = temp[0]
        gen_images.append(temp)
    
    
    heights = [256, 256, 256]
    widths = [256, 256, 256]
    
    fig_width = 8.
    fig_height = fig_width * sum(heights) / sum(widths)
    
    f, axarr = pyplot.subplots(3,3, figsize=(fig_width, fig_height),
            gridspec_kw={'height_ratios':heights})
    
    index = 0 
    for i in range(3):
        for j in range(3):
            axarr[i, j].imshow(gen_images[index])
            axarr[i, j].axis('off')
            index += 1
    pyplot.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    if save:
        pyplot.savefig("predicted_image.png")
    pyplot.show()
    
def print_layers(model):
    for i in range(len(model.layers)):
        layer = model.layers[i]
        # check for convolutional layer
        #if 'conv' not in layer.name:
        #    continue
        # summarize output shape
        print(i, layer.name, layer.output.shape)
    
def visualize_model_one_layer(filename = None, model = None, layers = None, size = (256, 256), save = False):
    # load the model
    if filename == None:
        print("No image to predict")
        return

    if model == None:
        model = load_model('256x256_30epoci/model_030000.h5')
    #print_layers(model)
    # redefine model to output right after the first hidden layer
    model = Model(inputs=model.inputs, outputs=model.layers[1].output)
    model.summary()
    # load the image with the required shape
    img = load_image(filename, size = size)

    feature_maps = model.predict(img)
    # plot all 64 maps in an 8x8 squares
    square = 8
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    if save:
        pyplot.savefig("feature_map.png")
    pyplot.show()

def visualize_model_multiple_layers(filename = None, model = None, layers = None, size = (256, 256), save = False):
    if filename == None:
        print("No image to predict")
        return

    if model == None:
        model = load_model('256x256_30epoci/model_030000.h5')
        
    if layers == None:
        layers = [10, 15, 20, 40, 50]

    print_layers(model)

    img = load_image(filename, size = size)
    
    outputs = [model.layers[i].output for i in layers]
    model = Model(inputs=model.inputs, outputs=outputs)
    
    feature_maps = model.predict(img)
    # plot all 64 maps in an 8x8 squares
    square = 8
    pyplot.figure()
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps[4][0, :, :, ix-1], cmap='gray')
            ix += 1
    pyplot.savefig("features.png", dpi = 1000)
    pyplot.show()
    '''for fmap in feature_maps:
    # plot all 64 maps in an 8x8 squares
        ix = 1
        index = 1
        pyplot.figure()
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = pyplot.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
                ix += 1
        if save:
            pyplot.savefig("feature_map_" + str(index) + "_.png")
        pyplot.show()'''

#predict_image_batch4('Test_Images/Aerial/DSC04652.JPG', save = True)
#predict_image_batch16('Test_Images/Aerial/DSC04652.JPG', save = True)
#predict_image_1layer_batch4('Train_Images/Aerial/DSC05060.JPG', save = True)
#predict_reverse_image_256('Train_Images/Aerial/DSC05060.JPG', save = True)
#predict_single_image_256('Train_Images/Aerial/DSC05060.JPG', save = True)
#predict_single_image_512('Test_Images/Aerial/DSC04670.JPG', save = True)
#predict_single_image_512_plus_one_layer('Train_Images/Aerial/DSC05060.JPG', save = True)
#predict_patch_image_256('Test_Images/Aerial/DSC04670.JPG', save = True)
##visualize_model_one_layer('Test_Images/Aerial/DSC04670.JPG', save = True)
visualize_model_multiple_layers('Test_Images/Aerial/DSC04670.JPG')