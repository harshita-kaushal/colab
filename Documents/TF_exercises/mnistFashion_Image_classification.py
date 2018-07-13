TF TFimport tensorflow.tensorflow
mnist = tf.keras.datasets.mnist

from tensorflow import keras

#Helper libraries 
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline  #for plotting the first 25 images in the dataset 

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#train_images and train_labels are the training set 
#test_images and test_labels are arrays for -- test set 

#there will be different classes for all the labels 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#each image is mapped to a single label. since class names are not included in the dataset

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)

#data must be preprocessed before training the network.

train_images = train_images /255.0 

test_images = test_images/255.0 

# for plotting the first twenty five images of the fashion dataset ( displays them as a class too )
plt.figure(fig_size = (10,10))
for i in range (25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i],cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])


#Flatten - changes format of the images from the 2D array (of 28x28 pixels) to a 1D array of 28*28 - 784 pixels 
    #unstacking rows of pixels in the image and is lining them up 
        #this layer is not learning any parameters, it's only reformatting the data 

#after flattening, the network = 2 tf.keras,layers.Dense layers 
    #densely- connected/fully-connected neural layers 
    #first dense layer has 128 nodes 
    #second is a 10 node softmax layer -- returns an array of 10 probability scores     
        #each node == score/probability of current image belonging to 1 of 10 digit classes 
            


model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)), 
    keras.layers.Dense(128, activation = tf.nn.relu), 
    keras.layers.Dense(10, activation = tf.nn.softmax)
])


#loss function - measures how accurate the model is during training  - want to minimize the function to "steer" the model in the right direction
#optimizer - how model is updated based on the data it sees and loss function 
#metrics - used to monitor the training and testing steps -- uses accuracy ( fraction of images that are correctly classifed)


model.compile(optimizer = tf.train.AdamOptimizer(), 
                loss = 'sparse_categorical_crossentropy', 
                metrics = ['accuracy'])


#then we'll train the neural network 
# feed the training data to the model 
# model associates the images and labels 
# ask the model to make predictions about a test set  ( the test_images array) 
# verify the predictions match the labels from the test_labels array        

    #call model.fit to training data 

model.fit(train_images, train_labels, epochs =5)


#Evaluating accuracy
 test_loss, test_acc = model.evaluate(test_images, test_labels)
 print('Test accuracy:', test_acc)

 #accuracy on test dataset is less than that of tge accuracy of training dataset 
    #it's overfitting 

#making a prediction about the images
predictions = model.predict(test_images)

#gives you an array of ten numbers -- describe the "confidence" of the model that the image corresponds to 
#each of the ten different articles of clothing 
     
predictions[0]
np.argmax(predictions[0]) # ---> gives you the one with the highest confidence value

#model is confident that the image is an ankle boot 
# ankle boot == class_names[9]

test_labels[0] # check 

#plot first 25 test images // precicted label and true label 


plt.figure(figsize=(10,10))
for i in range 25 
    plt.subplot95(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap = plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color == 'green'
    else:
        color = 'red'

    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                class_names[true_label]), 
                                color = color)


#using trained model to make a prediction about a single image 

img = test_images[0]
print(img.shape)

#even though we're using a single image --> we need to add to a list, because tf.keras models 
# are optimized to make predictions on a batch, or collection of examples at once 

#add img to batch where it's the only batch 
img = (np.expand_dims(img,0))
print(img.shape)


predictions = model.predict(img)
print(predictions)

prediction = predictions[0]
np.argmax(prediction)

