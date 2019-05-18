
# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
#from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from pandas_ml import ConfusionMatrix
import sys
import os
import PIL.Image,PIL.ImageFilter
from IPython.display import Image
from numpy import array
from ast import literal_eval
# In[2]:

#tf.__version__


# In[4]:

from IPython.display import Image
#Image('index.png')


# In[3]:

# Convolutional Layer 1.
filter_size1 = 4          # Convolution filters are 5 x 5 pixels.
num_filters1 = 32         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 4         # Convolution filters are 5 x 5 pixels.
num_filters2 = 64         # There are 36 of these filters.

# Convolutional Layer 3.
#filter_size3 = 5          # Convolution filters are 5 x 5 pixels.
#num_filters3 = 64         # There are 64 of these filters.

# Fully-connected layer.
fc_size = 512             # Number of neurons in fully-connected layer.


# In[4]:

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST', one_hot=True)
#data = read('./data/', one_hot=True)

# In[5]:

#print("Size of:")
#print("- Training-set:\t\t{}".format(len(data.train.labels)))
#print("- Test-set:\t\t{}".format(len(data.test.labels)))
#print("- Validation-set:\t{}".format(len(data.validation.labels)))


# In[6]:

data.test.cls = np.argmax(data.test.labels, axis=1)


# In[7]:

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 36


# In[8]:

def plot_images(images, cls_true, cls_pred=None):
 #   assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# In[9]:

# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)


# In[10]:

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05,name='new_weights'))


# In[11]:

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length],name='new_biases'))


# In[12]:

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


# In[13]:

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


# In[14]:

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# In[15]:

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')


# In[16]:

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])


# In[17]:

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')


# In[18]:

y_true_cls = tf.argmax(y_true, dimension=1)


# In[19]:

layer_conv1, weights_conv1 =     new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)


# In[20]:

layer_conv1


# In[21]:

layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)


# In[23]:

layer_conv2


#layer_conv3, weights_conv3 =     new_conv_layer(input=layer_conv2,
 #                  num_input_channels=num_filters2,
  #                 filter_size=filter_size3,
   #                num_filters=num_filters3,
    #               use_pooling=True)




#layer_conv3

# In[24]:

layer_flat, num_features = flatten_layer(layer_conv2)


# In[25]:

layer_flat


# In[26]:

num_features


# In[27]:

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)


# In[28]:

layer_fc1


keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(layer_fc1, keep_prob)


# In[29]:

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)


# In[30]:

layer_fc2

h_fc2_drop = tf.nn.dropout(layer_fc2, keep_prob=1.0)

# In[31]:

y_pred = tf.nn.softmax(h_fc2_drop,name="y_pred")


# In[32]:

y_pred_cls = tf.argmax(y_pred, dimension=1)


# In[34]:

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)


# In[35]:

cost = tf.reduce_mean(cross_entropy)


# In[36]:

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


# In[37]:

correct_prediction = tf.equal(y_pred_cls, y_true_cls)


# In[38]:

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[39]:

session = tf.Session()


# In[40]:

session.run(tf.global_variables_initializer())


# In[41]:

train_batch_size = 64


# In[42]:

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# In[43]:

def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    #plot_images(images=images[0:9],
     #           cls_true=cls_true[0:9],
      #          cls_pred=cls_pred[0:9])


# In[44]:
#confusion_matrix = ConfusionMatrix(y_true=cls_true,y_pred=cls_pred)
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls
    
    # Get the confusion matrix using sklearn.
   # cm = confusion_matrix(y_true=cls_true,
    #                   y_pred=cls_pred)
    cm = ConfusionMatrix(y_true=cls_true,y_pred=cls_pred)
    # Print the confusion matrix as text.
    #print(cm)
    cm.print_stats()

    # Plot the confusion matrix as an image.
    #plt.matshow(cm)

    # Make various adjustments to the plot.
   # plt.colorbar()
   # tick_marks = np.arange(num_classes)
   # plt.xticks(tick_marks, range(num_classes))
   # plt.yticks(tick_marks, range(num_classes))
   # plt.xlabel('Predicted')
 #   plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
  #  plt.matshow()


# In[45]:

# Split the test-set into smaller batches of this size.
test_batch_size = 64

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        #plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


# In[55]:

print_test_accuracy(show_example_errors=True)


# In[57]:

optimize(num_iterations=1)


# In[56]:

#print_test_accuracy(show_example_errors=True)
#print_test_accuracy(show_example_errors=True,show_confusion_matrix=True)


# In[58]:

optimize(num_iterations=99) # We already performed 1 iteration above.


# In[59]:

print_test_accuracy(show_example_errors=True,show_confusion_matrix=True)
#cm.print_stats()

# In[60]:

optimize(num_iterations=900) # We performed 100 iterations above.


# In[61]:

#print_test_accuracy(show_example_errors=True)
print_test_accuracy(show_example_errors=True,show_confusion_matrix=True)


# In[62]:

optimize(num_iterations=20000) # We performed 1000 iterations above.
print_test_accuracy(show_example_errors=True,show_confusion_matrix=True)
#print_test_accuracy(show_example_errors=True,show_confusion_matrix=True)

saver = tf.train.Saver()
save_path = saver.save(session, "./model2.ckpt")
graph_def = tf.GraphDef()
save_path = tf.train.write_graph(session.graph_def,"./model", "graph.pb",False)
#print ("Model saved in file: ", save_path)



def imageprepare(filename):

    im = PIL.Image.open(filename)
    #img = PIL.ImageOps.invert(im)
    #im = PIL.Image.open(filename)
   # imng = np.array(im).reshape(1,784)
    tv = list(im.getdata()) #get pixel values
#normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv]
    return tva





#def imageprepare(argv):
    #"""
   # This function returns the pixel values.
   # The imput is a png file location.
   # """
   # im = PIL.Image.open(argv).convert('L')
   # width = float(im.size[0])
   # height = float(im.size[1])
   # newImage = PIL.Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels

  #  if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        #nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        #if (nheight == 0): #rare case but minimum is 1 pixel
       #     nheight = 1
        # resize and sharpen
      #  img = im.resize((20,nheight), PIL.Image.ANTIALIAS).filter(PIL.ImageFilter.SHARPEN)
     #   wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
    #    newImage.paste(img, (4, wtop)) #paste resized image on white canvas
   # else:
        #Height is bigger. Heigth becomes 20 pixels. 
        #nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
       # if (nwidth == 0): #rare case but minimum is 1 pixel
      #      nwidth = 1
         # resize and sharpen
     #   img = im.resize((nwidth,20), PIL.Image.ANTIALIAS).filter(PIL.ImageFilter.SHARPEN)
    #    wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
   #     newImage.paste(img, (wleft, 4)) #paste resized image on white canvas

    #newImage.save("sample.png")

  #  tv = list(newImage.getdata()) #get pixel values

    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
 #   tva = [ (255-x)*1.0/255.0 for x in tv]
#    return tva
    #print(tva)


#def main(argv):
 #   """
  #  Main function.
   #"""
    #global imvalue
    #imvalue = imageprepare(argv)
    #predint = predictint(imvalue)
    #print (predint[0]) #first value in list
    

#if __name__ == "__main__":
#    main(sys.argv[1])


  #  """
  #  Main function.
  # """
    
#global imvalue

    #Names = ["argv"]
    #for name in Names:
FileList = []
for dirname in os.listdir(sys.argv[1]):
                        path = os.path.join(sys.argv[1],dirname)
                #       print("@@@@",os.listdir(argv))
                        for filename in os.listdir(path):
         #               print("....",filename)

                  			  if filename.endswith(".jpg"):
                                   		 FileList.append(os.path.join(sys.argv[1],dirname,filename))

    #shuffle(FileList)i
global filename
for filename in FileList:
        #predint[]
        print(filename)
        #imvalue = imageprepare(filename)
	saver.restore(session, "./model2.ckpt")
        tf.import_graph_def(graph_def,name="./graph.pb")
#        tf.reset_default_graph()	
        #print ("Model restored.")
#        plot_conv_weights(weights=weights_conv1)
 #       plot_conv_layer(layer=layer_conv1, image=imvalue)
  #      plot_conv_weights(weights=weights_conv2)
   #     plot_conv_layer(layer=layer_conv2, image=imvalue)
        imvalue = imageprepare(filename)
        prediction=tf.argmax(y_pred,1)
        predint=prediction.eval(feed_dict={x:[imvalue],keep_prob:1.0 }, session=session)

           # predint = predictint(imvalue)
         #   print(".....",filename)
        print (predint[0]) #first value in list
            #p= (predint[0]) #first value in list
	    #sum=0
            #for i in range(0,35):
	#	if (p%i) ==0
	#		sum+=1
	   # workbook = xlsxwriter.Workbook('demo.xlsx')
#	    worksheet = workbook.add_worksheet()
#	    worksheet.add_table('D6:AN41',predint[0])
	    #worksheet.write('D:',predint[0])
#	    workbook.close()
	text_file = open('new17.ods','a')
	text_file.write(filename +" ")
	text_file.write('{}'.format(predint[0]) + "\n")
	text_file.close()

#if __name__ == "__main__":
   # main(sys.argv[1])

     




