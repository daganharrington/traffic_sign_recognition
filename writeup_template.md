#**Traffic Sign Recognition** 

**Goals**
The goals of this project were to:
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

---

###Design and Test a Model Architecture

####1. Preprocessing steps

I applied a few preprocessing steps to each image, and created new images from some transformations.
I applied grayscale and pixel normalization to each image...the rationale for both of these choices is that it cuts down
on uninformative noise and focuses on features that will be useful during training. If I had more time, I had planned to increase the image contrast and zoom in on images (borders and background are, generally, not helpful for classification of a foreground image) for the same reasons.

I also increased the size of the data by providing a slight right-rotation and left-rotation of each image. This does 2 things: 1) just plainly gives me more data, which can't be bad and 2) makes the dataset more robust to variations in images.

####2. Final Model architecture
My final model consisted of the following layers:
```
def Net(x):
    """Convolutional neural net model."""
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x20.
    conv1 = conv_layer(x, 5, 3, 20)
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # Pooling. Input = 28x28x20. Output = 14x14x36.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x36.
    conv2 = conv_layer(conv1, 5, 20, 36)
    # Activation.
    conv2 = tf.nn.relu(conv2)
    # Pooling. Input = 10x10x36. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x36. Output = 900.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 900. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(900, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1    = tf.nn.relu(fc1)
    # But .. add dropout for this activation layer.
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation
    fc2    = tf.nn.relu(fc2)
    # But ... add dropout for this activation layer
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))    
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
 ```
 
 This is roughly a copy of the famous LeNet architecture..with the addition of dropout and max_pool layers.

####3. Model Training
The model was trained with the following hyperparameters:
rate = 0.0005  # Learning rate
EPOCHS = 17 # Number of training cycles
BATCH_SIZE = 128 # Number of images per batch

I also applied L2-norm regularization (tf.nn.l2_loss) across the dataset, which had a good effect on training accuracy.

My final model results were:
EPOCH 17 ...
Validation Accuracy = 0.944

After model done training:
Test Accuracy = 0.931

I chose the LeNet architecture as a starting point and it brought me close to the required training accuracy along with the image preprocessing steps I had coded up originally. Adding dropout helped raise the accuracy quite a bit. Ultimately, tuning hyperparameters was not near as useful as choosing a good, established architecture and preprocessing the images.


###Test a Model on New Images

The 5 german traffic signs I found were screenshots from walking around in Berlin with Google's streetview feature.

Ultimately, I was unable to classify new images. The issue was that I was getting the exact same prediction vectors for each new image. At first, I had assumed this was a problem with tensorflow not loading the saved session appopriately. However, getting a prediction for my new images in the same `with` (and resulting `tf.Session` bound variable) I was still not able to get any difference for each new image (this rules out the notion that tensorflow may not have been saving or restoring the model appropriately)
