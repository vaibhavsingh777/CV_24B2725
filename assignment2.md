# **Assignment 2: CNNs for Image Classification**

In this assignment, I explored Convolutional Neural Networks (CNNs) for image classification using the CIFAR-10 dataset in PyTorch. I compared the performance of a simple Artificial Neural Network (ANN) against a CNN to see how they handle spatial image data.

## **Step 1: Importing Libraries**

First, I imported the necessary libraries. This included:

* torch, torch.nn, and torch.optim for building and training the models.  
* torchvision and transforms for dataset loading and image transformations.  
* DataLoader for batching the data.  
* sklearn.metrics for evaluating model performance (accuracy, precision, recall, F1, confusion matrix).  
* seaborn and matplotlib.pyplot for plotting results.  
* numpy for numerical operations and pandas for creating comparison tables.

## **Step 2: Setting Device and Transformations**

I set up the device to use CUDA (GPU) if available, otherwise defaulting to CPU. Then, I defined the image transformations:

* Converting images to PyTorch tensors (transforms.ToTensor()).  
* Normalizing the tensor values to the range \[-1, 1\] using a mean and standard deviation of 0.5 for each channel (transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))).

## **Step 3: Loading CIFAR-10 Dataset**

I loaded the CIFAR-10 dataset using torchvision.datasets.CIFAR10.

* I applied the defined transform to both the training and test sets.  
* I created DataLoader instances (train\_loader, test\_loader) to handle batching (batch size of 64). The training loader shuffles the data each epoch, while the test loader does not.  
* I printed the number of samples and the shape of a sample batch to verify.

## **Step 4: Defining the ANN Model**

I defined a simple ANN model (class ANN) with three fully connected layers (nn.Linear):

* Input layer: Flattened image size (32\*32\*3) to 1024 units.  
* Hidden layer: 1024 units to 256 units.  
* Output layer: 256 units to 10 units (for the 10 CIFAR-10 classes).  
* I used the ReLU activation function (nn.ReLU) after the first two layers.  
* The forward method flattens the input image and passes it through the layers.

## **Step 5: Initializing ANN Components**

I instantiated the ANN model, moved it to the selected device, and defined the loss function and optimizer:

* Loss: nn.CrossEntropyLoss, suitable for multi-class classification.  
* Optimizer: optim.Adam with a learning rate of 0.001.  
* I printed the model structure to check it.

## **Step 6: Training the ANN**

I trained the ANN model for 5 epochs. In each epoch:

* I iterated through the train\_loader.  
* Moved images and labels to the device.  
* Performed a forward pass to get outputs.  
* Calculated the loss using criterion\_ann.  
* Performed backpropagation (loss.backward()) and updated weights (optimizer\_ann.step()).  
* Cleared gradients (optimizer\_ann.zero\_grad()).  
* I printed the loss every 100 steps to monitor progress.

## **Step 7: Plotting ANN Training Loss**

After training, I plotted the loss values recorded during training (ls\_losses\_ann) against the batch index. This visualization helped me see if the loss decreased over time, indicating successful training.

## **Step 8: Evaluating the ANN**

I evaluated the trained ANN on the test set (test\_loader).

* I set the model to evaluation mode (model\_ann.eval()).  
* I iterated through the test loader *without* calculating gradients (torch.no\_grad()).  
* Collected all predictions and true labels.  
* Calculated accuracy, precision, recall, F1-score (using 'macro' averaging), and the confusion matrix using sklearn.metrics.  
* Printed the metrics and visualized the confusion matrix using seaborn.heatmap.

## **Step 9: Defining the CNN Model**

Next, I defined a simple CNN model (class CNN):

* Two convolutional layers (nn.Conv2d):  
  * conv1: 3 input channels to 32 output channels, kernel size 3, padding 1\.  
  * conv2: 32 input channels to 64 output channels, kernel size 3, padding 1\.  
* ReLU activation (nn.ReLU) after each convolutional layer.  
* Max pooling layer (nn.MaxPool2d) with a 2x2 kernel and stride 2 after each convolutional block, reducing spatial dimensions.  
* Two fully connected layers (nn.Linear):  
  * fc1: Takes the flattened output of the last pooling layer (64 \* 8 \* 8 features) to 512 units.  
  * fc2: 512 units to 10 output classes.  
* The forward method defines the sequence of operations.

## **Step 10: Initializing and Training the CNN**

Similar to the ANN, I instantiated the CNN model, moved it to the device, defined nn.CrossEntropyLoss as the criterion, and used optim.Adam (lr=0.001) as the optimizer. I then trained the CNN for 5 epochs using the same training loop structure as in Step 6\.

## **Step 11: Evaluating the CNN**

I evaluated the trained CNN on the test set, following the same procedure as in Step 8 (setting to eval mode, using torch.no\_grad(), collecting predictions/labels, calculating metrics, and plotting the confusion matrix).

## **Step 12: Comparing ANN and CNN Performance**

To directly compare the models, I created a Pandas DataFrame (df\_comparison) showing the calculated Accuracy, Precision, Recall, and F1-Score for both the ANN and CNN side-by-side.

## **Step 13: Visualizing CNN Filters (Advanced)**

To understand what the CNN learned, I visualized the filters (weights) from the first convolutional layer (model\_cnn.conv1).

* I extracted the weights and normalized them for display.  
* I plotted the first 16 filters as images using matplotlib. This gave me an idea of the low-level features (like edges or color blobs) the network detects.

## **Step 14: Data Augmentation and Retraining CNN (Advanced)**

To potentially improve CNN performance, I introduced data augmentation for the training set:

* Defined train\_transform\_augmented including RandomHorizontalFlip and RandomCrop (with padding).  
* Created a new augmented training dataset (train\_dataset\_aug) and loader (train\_loader\_aug). The test set remained unchanged.  
* I initialized a new CNN model (model\_cnn\_augmented) and trained it for 5 epochs using the augmented data.

## **Step 15: Visualizing Incorrect Predictions (Advanced)**

Finally, I wanted to see where the best model (the augmented CNN, if trained) was making mistakes.

* I set the chosen model to eval mode.  
* Iterated through the test\_loader, identifying images where the prediction didn't match the true label.  
* Stored some of these incorrect examples (image tensor, predicted label, true label).  
* Unnormalized the image tensors for visualization.  
* Plotted 16 incorrectly predicted images using matplotlib, showing both the true (T) and predicted (P) labels. This helped me understand the types of errors the model was prone to making.