# Cell-Nuclei--Semantic-Segmentation

## Classes
1. `Augment()` <br/>
The Augment() class is used to augment the training images. It uses Tensorflow Keras Layers (tensorflow.keras.layers.Layer) to RandomFlip the images and masks. 

2.  `DisplayCallback()` <br/>
The DisplayCallback() class is used to display the original input, true mask and predicted mask for every epoch during training.

## Functions
1. `display()` <br/>
  - Argument: list to be displayed
  - Displays input image, true mask and predicted mask.

2. `unet()` 
  - Argument: Number of output classes
  - Creates U-Net architecture
  - Input layer
  - Contracting path - returns list of tensors that contain the output of each layer
  - Upsampling layer - returns list of transpose convolutional layers. It upsamples the current tensor using each transpose convolutional layer and concatenates it with the corresponding tensor from the contracting path using the Concatenate layer
  - Output layer - a Conv2DTranspose layer
  - Model creation - the Model is created. It takes the input layers as inputs and outputs the final tensor produces
3. `create_mask()`
  - Argument: the mask to predict
  - Creates mask for predicted mask
4.  `show_predictions()`
  - Argument: the dataset predicted
  - Displays the input image, true mask and predicted mask using the trained model
5. `test_predictions()`
  - Argument: the dataset predicted
  - Displays the input image, true mask and predicted mask of the test dataset using the loaded model

## 1. Data Loading
Data source:

### List of train images and masks
Once the dataset is loaded, its seperated into train images and mask lists. The list will obtain images and masks that are color converted using the cv2 library and resized into 128 by 128. For the mask, its dimensiona are expanded to (128,128,1)

### Converted into arrays
The train images and masks are then converted array using the `np.array()` method.

## 2. EDA
Before preprocessing the dataset any further, I displayed some of the images and masks.
 <p align="center">
  <img src="https://github.com/natashanazamil/Cell-Nuclei--Semantic-Segmentation/blob/main/images/eda1.png" alt="Loaded Images">
  <br>
  <em>Loaded Images</em>
</p>
 <p align="center">
  <img src="https://github.com/natashanazamil/Cell-Nuclei--Semantic-Segmentation/blob/main/images/eda2.png" alt="Loaded Masks">
  <br>
  <em>Loaded Masks</em>
</p>

## 3. Data Preprocessing
### Data Normalization
The value of the pixels are then normalized by dividing it with 255

### Split dataset into train and test data
Using sklearn model_selection module, I used its `train_test_split()` method to split the dataset into train and test datasets with a test size of 30% and 70% for training.

### Create dataset
The dataset using the input tensors are created using tensorflows `from_tensor_slices` method. Then the train images and train masks are combined into a single dataset using the `zip` method. Same goes for the test images and test masks.<br/>

For the training data, the resulting dataset is shuffled, batched, and repeated to create multiple epochs. The Augment method is applied to each batch to perform data augmentation, and the resulting dataset is cached and prefetched to improve performance.<br/>

For the testing data, the dataset is simply batched with the specified batch size.<br/>

Finally, the train_batches dataset is used to display a sample image and mask using the display function.<br/>

Here is a sample of the training images and its true mask: 

 <p align="center">
  <img src="https://github.com/natashanazamil/Cell-Nuclei--Semantic-Segmentation/blob/main/images/eda3.png" alt="Sample Image And Mask">
  <br>
  <em>Sample Image And Mask</em>
</p>

## 4. Model Development
* The base model uses transfer learning from tensorflow keras applications, **MobileNetV2**
* The output of the base_model is extracted and used as input for the upsampling path of the U-Net model. 
* The upsampling path of the U-Net model using a list of upsampling layers created using the `pix2pix` module.
* Using the `unet()` function, the U-Net model is created. 
* Finally, the model is compiled using the SparseCategoricalCrossentropy as its loss and Adam as its optimizer. 

Here's the mask perdiction before training the model displayed using the `show_predictions()` function:
<p align="center">
  <img src="https://github.com/natashanazamil/Cell-Nuclei--Semantic-Segmentation/blob/main/images/eval_bfr_training.png" alt="Prediction Before Training">
  <br>
  <em>Prediction Before Training</em>
</p>


## 5. Model Training
The model is trained with 15 epochs with an earlystopping callback to avoid overfitting. <br/>
Using tensorflows callback TensorBoard, heres the training and validation accuracy and loss graph shown throughout the training process.
<p align="center">
  <img src="https://github.com/natashanazamil/Cell-Nuclei--Semantic-Segmentation/blob/main/images/tb_acc.PNG" alt="TensorBoard - Training And Validation Accuracy">
  <br>
  <em>TensorBoard - Training And Validation Accuracy</em>
</p>
<p align="center">
  <img src="https://github.com/natashanazamil/Cell-Nuclei--Semantic-Segmentation/blob/main/images/tb_loss.PNG" alt="TensorBoard - Training And Validation Loss">
  <br>
  <em>TensorBoard - Training And Validation Loss</em>
</p>

## 6. Model Analysis

With the `evaluate()` method, here's the accuracy and loss of the model after training. 

<p align="center">
  <img src="https://github.com/natashanazamil/Cell-Nuclei--Semantic-Segmentation/blob/main/images/eval1.PNG" alt="Model Evaluation After Training">
  <br>
  <em>Model Evaluation After Training</em>
</p>


A further visualization is done using the `show_predictions()` function to see the predicted mask. Here are some samples
<p align="center">
  <img src="https://github.com/natashanazamil/Cell-Nuclei--Semantic-Segmentation/blob/main/images/eval_aft_train.png" alt="Prediction After Training">
  <br>
  <em>Prediction After Training</em>
</p>

## 7. Model Deployment
The model is saved to the **model** file using the `.save()` method. 

## 8. Model Loading
To further test the saved model, its loaded using the `.load_model()` method from tensorflow keras.

## 9. Test Model with Seperate test Dataset
The test images and masks were loaded and preprocessed as per the training dataset. The model achieved 93% accuracy of predicting the mask of the test dataset as shown here.

<p align="center">
  <img src="https://github.com/natashanazamil/Cell-Nuclei--Semantic-Segmentation/blob/main/images/eval2.PNG" alt="Model Evaluation On Test Dataset">
  <br>
  <em>Model Evaluation On Test Dataset</em>
</p>

Further visualization is done using the `test_predictions()` function to see the outcome of the model using the test dataset.

<p align="center">
  <img src="https://github.com/natashanazamil/Cell-Nuclei--Semantic-Segmentation/blob/main/images/test.png" alt="Predicted Test Mask">
  <br>
  <em>Predicted Test Mask</em>
</p>
