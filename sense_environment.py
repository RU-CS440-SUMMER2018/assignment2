import numpy as np
import sys
sys.path.append("./AI_FINAL_ASS")
import random
import matplotlib.pyplot as plt
##IMPORT MODEL
#from tf_train_model_ASSIGNMENT_FILE import build_train
from tf_classify_sensed_image import evaluate_model
##TO EVALUATE
from PIL import Image


def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    img = Image.fromarray(two_d, 'L')
    return img

#############################################################################
#############################################################################
#############################################################################
#############################################################################
def predict(input_x):
    out = ""
    #if len(input_x) != (28*28):
    #print("The input image or input array is shaped incorrectly. Expecting a 28x28 image.")
    '''
    for i in xrange(0,28):
        out = out+"\n"
	    for( j in xrange(0,28)):
            if (input_x[(i*28)+j]>0.5):
	            out = out + "1"
	        else:
		        out = out + "0"

    '''
    #print( "Input image array: \n", out)
	
    E = evaluate_model()
    #E.__init__()
    prediction = E.evaluate_model(model_version = 20,input = np.float32(input_x.reshape(28,28)).flatten().reshape(1,784)) #FIRST ENTRY IS SAME AS ALL ENTRIES FOR CLASSIFIER
    #print("\nPREDICTION CLASSIFIED:",prediction[0])	
    #prediction = int(random.random()*9.9) #Current prediction is random

    return prediction[0]
'''
#############################################################################
#############################################################################
#############################################################################
#############################################################################
def predict(input_x):
    out = ""
    if len(input_x) != 28*28:
        print("The input image or input array is shaped incorrectly. Expecting a 28x28 image.")
        exit(-1)
    for i in xrange(0,28):
        out = out+"\n"
        for j in xrange(0,28):
            if input_x[(i*28)+j]>0.5:
                out = out+"1"
            else:
                out = out +"0"
    #print "Input image array: \n", out
	
    E = evaluate_model()
    #E.__init__()
    
    prediction = E.evaluate_model(6,input = input_x.flatten())[0][0] #FIRST ENTRY IS SAME AS ALL ENTRIES FOR CLASSIFIER

    print("PREDICTED:",prediction)
	
    #prediction = int(random.random()*9.9) #Current prediction is random

    return prediction            
'''

if len(sys.argv) < 1:
    print("The script should be passed the full path to the image location")

filename = sys.argv[1]
# full_image = Image.open('$PRACSYS_PATH/prx_output/images/_0.jpg')
full_image = Image.open(filename)
size = 28,28
image = full_image.resize(size, Image.ANTIALIAS)
width, height = image.size
pixels = image.load()
print (width, height)
fill = 1
array = [[fill for x in range(width)] for y in range(height)]

for y in range(height):
    for x in range(width):
        r, g, b = pixels[x,y]
        lum = 255-((r+g+b)/3)
        array[y][x] = float(lum/255)

image_array = []
for arr in array:
    for ar in arr:
    	image_array.append(ar)
im_array = np.array(image_array)
print (image_array)
print (im_array)
out = predict(im_array)

outfile = "/".join(filename.split("/")[:-1])+"/predict.ion"
outf = open(outfile, 'w')
outf.write(str(out))
