#%% IMPORTS
import os
import cv2
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_examples as tfex
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from tensorflow_examples.models.pix2pix import pix2pix

# CONSTANTS
OUTPUT_CLASSES = 2
BATCH_SIZE = 32
BUFFER_SIZE = 1000
SEED = 123
IMG_SIZE = (128,128)

# PATHS
TRAIN_PATH = os.path.join(os.getcwd(),'data-science-bowl-2018','data-science-bowl-2018-2','train')
TEST_PATH = os.path.join(os.getcwd(),'data-science-bowl-2018','data-science-bowl-2018-2','test')
MODEL_PATH = os.path.join('model','natasha_model3.h5')


# CLASS
# C1. Augmentation Class
class Augment(keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    self.augment_inputs = keras.layers.RandomFlip(mode='horizontal', seed=seed)
    self.augment_labels = keras.layers.RandomFlip(mode='horizontal', seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels

# C2. DisplayCallback class to display predicted images during training
class DisplayCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))


# FUNCTIONS
# F1. Display function to display input images, true mask and predicted mask
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# F2. UNet Model Function
def unet(output_channels:int):
    inputs = keras.layers.Input(shape=[128,128,3]) # input layer
    #Contracting path
    skips = down_stack(inputs)
    x = skips[-1]       
    skips = reversed(skips[:-1])

    # Upsampling layer
    for up,skip in zip(up_stack,skips):
        x = up(x)
        concat = keras.layers.Concatenate()
        x = concat([x,skip])

    # Transpose convolution and output layer 
    last = keras.layers.Conv2DTranspose(output_channels,kernel_size=3,strides=2,padding='same')     #64x64 --> 128x128
    outputs = last(x) 
    model = keras.Model(inputs=inputs,outputs=outputs)
    return model

# F3. Create mask for predicted mask function
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]       
    return pred_mask[0]

# F4. Show prediction function to show input image, true mask and predicted mask
def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

# F5. Show test predictions from loaded model
def test_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = loaded_model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])


#%% 1. Data Loading
train_images = []
train_masks = []

for file_name in os.listdir(os.path.join(TRAIN_PATH,'inputs')):
  image_path = os.path.join(os.path.join(TRAIN_PATH,'inputs'),file_name)
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # Convert image to RGB
  image = cv2.resize(image,IMG_SIZE) # Resize image to (128,128)
  train_images.append(image)


for file_name in os.listdir(os.path.join(TRAIN_PATH,'masks')):
  mask_path = os.path.join(os.path.join(TRAIN_PATH,'masks'), file_name)
  mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Convert mask image to GRAYSCALE
  mask = cv2.resize(mask,IMG_SIZE) # Resize image to (128,128)
  mask = np.expand_dims(mask,axis=-1) # Expand dimension to (128,128,1)
  train_masks.append(mask)

# Convert images and masks arrays
train_images = np.array(train_images)
train_masks = np.array(train_masks)

# %% 2. EDA 
plt.figure(figsize=(10, 10))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.imshow(train_images[i])
    plt.axis("off")
    plt.title("Image " + str(i))
plt.show()

plt.figure(figsize=(10, 10))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.imshow(train_masks[i])
    plt.axis("off")
    plt.title("Mask " + str(i))
plt.show()


# %% 3. Data Preprocessing
# Normalization
train_images = train_images / 255.0
train_masks = train_masks / 255

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks, test_size=0.3, random_state=SEED)

# Build dataset
X_train_tensor = tf.data.Dataset.from_tensor_slices(X_train)
X_test_tensor = tf.data.Dataset.from_tensor_slices(X_test)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)

# Combine images and masks into a single dataset
train_dataset = tf.data.Dataset.zip((X_train_tensor, y_train_tensor))
test_dataset = tf.data.Dataset.zip((X_test_tensor, y_test_tensor))

train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = test_dataset.batch(BATCH_SIZE)

# Display sample using display function
for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])


#%% 4. Model Development
# 4.1 Create base model using transfer learning with MobileNetV2
base_model = keras.applications.MobileNetV2(input_shape=(128,128,3), include_top=False)
base_model.summary()

layer_names = [
    'block_1_expand_relu',      #64x64
    'block_3_expand_relu',      #32x32
    'block_6_expand_relu',      #16x16
    'block_13_expand_relu',     #8x8
    'block_16_project'          #4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# 4.2 Instantiate the feature extractor
down_stack = keras.Model(inputs=base_model.input,outputs=base_model_outputs)
down_stack.trainable = False

# 4.3 Define the upsampling path
up_stack = [
    pix2pix.upsample(512,3),        #4x4 --> 8x8
    pix2pix.upsample(256,3),        #8x8 --> 16x16
    pix2pix.upsample(128,3),        #16x16 --> 32x32
    pix2pix.upsample(64,3)          #32x32 --> 64x64
]

# 4.4 Create UNet model using unet function
model = unet(OUTPUT_CLASSES)
model.summary()

# 4.5 Compile model
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
keras.utils.plot_model(model, show_shapes=True)

# Show prediction before training model
show_predictions()


# %% 5. Model Training
# Hyperparameters for model
EPOCHS = 15
VAL_SUBSPLITS = 5
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
VALIDATION_STEPS = len(test_dataset) // BATCH_SIZE // VAL_SUBSPLITS

# Callbacks
log_path = os.path.join(os.getcwd(),'tb_logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = keras.callbacks.TensorBoard(log_path)
es = keras.callbacks.EarlyStopping(patience=5, monitor='val_accuracy')

# Fit model
history = model.fit(
    train_batches,
    validation_data=test_batches,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=[DisplayCallback(),tb, es]
)

#%% 6. Model Analysis
# Predicted test images using show prediction function
show_predictions(test_batches, 3)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# %% 7. Model Deployment
model.save(MODEL_PATH)

#%% 8. Load Model
loaded_model = keras.models.load_model(MODEL_PATH)
loaded_model.summary()

#%% 9. Test with seperate test dataset
# 9.1 Load test dataset
test_images = []
test_masks = []

for file_name in os.listdir(os.path.join(TEST_PATH,'inputs')):
  image_path = os.path.join(os.path.join(TEST_PATH,'inputs'),file_name)
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # Convert image to RGB
  image = cv2.resize(image,IMG_SIZE) # Resize image to (128,128)
  test_images.append(image)


for file_name in os.listdir(os.path.join(TEST_PATH,'masks')):
  mask_path = os.path.join(os.path.join(TEST_PATH,'masks'), file_name)
  mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Convert mask image to GRAYSCALE
  mask = cv2.resize(mask,IMG_SIZE) # Resize image to (128,128)
  mask = np.expand_dims(mask,axis=-1) # Expand dimension to (128,128,1)
  test_masks.append(mask)

test_images = np.array(test_images)
test_masks = np.array(test_masks)

plt.figure(figsize=(10, 10))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.imshow(test_images[i])
    plt.axis("off")
    plt.title("Image " + str(i))
plt.show()

plt.figure(figsize=(10, 10))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.imshow(test_masks[i])
    plt.axis("off")
    plt.title("Mask " + str(i))
plt.show()

#%% 9.2 Preprocess test dataset
# Normalize 
test_images = test_images / 255.0
test_masks = test_masks / 255
# Use the show_predictions method to visualize the predicted masks

#%% 9.3 Evaluate model with test dataset
test2_loss, test2_acc = loaded_model.evaluate(test_images, test_masks, verbose=2)
print('Test accuracy:', test2_acc)
print('Test loss:', test2_loss)

test_predictions(tf.data.Dataset.from_tensor_slices((test_images, test_masks)).batch(16),num=3)



