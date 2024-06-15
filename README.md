# **Machine Learning Models for Momee.id** 👶🏻✨

## **Data Understanding**

The data or dataset used in this machine learning project is self-generated data or image scraping from *Google Image*, *Bing Image*, and *Amazon*. Which can be visited or downloaded at [kaggle](https://www.kaggle.com/datasets/sahrul59/momee-dataset/).

### **Count Data and Classes**

Figure 1. Number of Data in Train Path

![data_count_train](https://github.com/muhammadsahrul59/Movie-System-Recommendation/assets/101655285/5350f1da-96a6-4ea2-a630-64003fe3f0b4)

In Figure 1, there are **14 labels**, the number of each label is **1,200 images**, and has a total number of images of **16,800 images**.

Figure 2. Number of Data in Val Path

![data_count_val](https://github.com/muhammadsahrul59/Movie-System-Recommendation/assets/101655285/13e7cce2-7084-41c9-abd3-10906a224cfa)

In Figure 2, there are **14 labels**, the number of each label is **300 images**, and has a total number of images of **4,200 images**.

### **Data Generator**

Tensorflow model training can only work with numerical data. Therefore, images need to be processed in order to become input for the tensorflow model. There are two steps that need to be done, namely creating an ImageDataGenerator object and applying it to images in the training data, validation data and test data.

ImageDataGenerator is part of the tensorflow.keras.preprocessing.image module. This object is used to store the numerical representation of the image. ImageDataGenerator has several parameters that can be set, one of the main ones is rescale. This parameter is used to perform feature scaling on image data, for example by normalization, which is changing the value so that it is in the interval from 0 to 1.

In the training data we will also set some parameters for image augmentation. Augmentation is a technique to extend the training dataset by creating variations of existing images. With this technique, we can generate a new image by making small changes to the existing image, such as rotating, cropping, flipping or shifting the image. In this way, we can provide variety in the training data used and can prevent overfitting.

In this casptone project, image augmentation is performed on the training data by rotating the image by 20 degrees, making the image dark and bright, and zooming by 20%. In addition, we also flip the image horizontally.

Figure 3. Show Image (train_generator)

![show_image_traingenerator](https://github.com/muhammadsahrul59/Movie-System-Recommendation/assets/101655285/af188069-0b41-4f5a-9e18-9aecfd66570a)

In Figure 3, we can see the results of the ImageDataGenerator that has been done, namely there are dark and bright images, images that are rotated 20 degrees, images that are zoomed, and images that are horizontally flipped.

Figure 4. Show Image (val_generator)

![show_image_valgenerator](https://github.com/muhammadsahrul59/Movie-System-Recommendation/assets/101655285/a02a2b5a-8279-4692-b08e-0f3cccddc493)

In Figure 4, It can be seen that the images look normal because they are intended to validate the model's performance on real data. It is important to ensure that the model evaluation metrics reflect its ability to generalize to real-world data.

## **Modelling**

This stage discusses the machine learning model used to solve the problem. This process is done using the **MobileNetV2** algorithm. The final result expected from this image classification is that it can make it easier for users to find the desired category of baby items.

- **Pre_trained_model**: Uses MobileNetV2 pre-trained on the ImageNet dataset without its top layers. All layers are set to non-trainable.

- **Conv2D**: Adds 128 convolution filters of size 3x3 to extract high-level features such as edges and textures.

- **MaxPooling2D**: Reduces the spatial dimensions of feature maps with a pool size of 2x2, decreasing computational complexity and preventing overfitting.

- **Flatten**: Converts 2D feature maps into a 1D vector before feeding into the Dense layers.

- **Dropout**: Adds a 20% dropout to prevent overfitting by ignoring some neurons during training.

- **Dense**: Adds a Dense layer with 128 neurons and ReLU activation, followed by a final Dense layer with 14 neurons and softmax activation for classification into 14 categories.

To enhance the training process of our image classification model, we use several callbacks and optimization techniques. These components help in monitoring performance, saving the best model, and adjusting the learning rate dynamically.

- **Custome Callback**: We implement a custom callback myCallback to stop training early if the model reaches a specific accuracy threshold. If the accuracy exceeds 96% and val_accuracy exceeds 94%, training will be stopped, ensuring we don't over-train the model.

- **Optimizer**: The **Adam optimizer** is used with a learning rate of 0.001 to compile the model. Adam is known for its efficiency and effectiveness in training deep learning models.

- **Learning Rate Reduction**: We employ **ReduceLROnPlateau** to reduce the learning rate when the validation loss plateaus. This helps in fine-tuning the model and achieving better performance.

Table 1. Training Data Results
|No |loss     |accuracy |val_loss |val_accuracy | learning_rate|
|---|---------|---------|---------|-------------|--------------|
|32 |0.128906 |0.959048 |0.252282 |0.937143     | 0.000125     |
|33 |0.128531 |0.959643 |0.253884 |0.935714     | 0.000125     |
|34 |0.129603 |0.958750 |0.252996 |0.936429     | 0.000125     |
|35 |0.125382 |0.960893 |0.254300 |0.935952     | 0.000063     |
|36 |0.122497 |0.960714 |0.246770 |0.940238     | 0.000063     |

In Table 1, it shows that the desired accuracy and val_accuracy are achieved, which is an accuracy exceeding 96% and val_accuracy of 94%.

## **Model Evaluation**

Figure 5. Model Plot

![plot_image](https://github.com/muhammadsahrul59/Movie-System-Recommendation/assets/101655285/ccc91651-93f9-4026-b2ae-3191cdef0b70)

In Figure 5, it can be seen above that the results of the previous training model show good results which meet the desired accuracy, which is Accuracy exceeds 96% and Val_accuracy exceeds 94%, and Best Epoch for Accuracy is 37 and Best Epoch for Loss is 32.


Figure 6. Confution Matrix

![confution_matrix](https://github.com/muhammadsahrul59/Movie-System-Recommendation/assets/101655285/542f767a-7f5e-43ab-9f7b-eaa07aa3ca35)

The confusion matrix provided above is an essential tool for understanding the performance of a classification model. This matrix compares the actual classifications with the predicted classifications made by the model. Each row of the matrix represents the true labels, while each column represents the predicted labels. Below, we break down the components and insights derived from the matrix.

In Figure 6, the diagonal elements (from top-left to bottom-right) show the number of correct predictions for each class. For example:

- `baby_bed`: 286 correct predictions
- `baby_car_seat`: 287 correct predictions
- `baby_folding_fence`: 292 correct predictions
- `bathtub`: 261 correct predictions
- `booster_seats`: 260 correct predictions
- `bouncer`: 261 correct predictions
- `breast_pump`: 288 correct predictions
- `carrier`: 284 correct predictions
- `earmuffs`: 291 correct predictions
- `ride_ons`: 275 correct predictions
- `rocking_horse`: 280 correct predictions
- `sterilizer`: 294 correct predictions
- `stroller`: 300 correct predictions
- `walkers`: 290 correct predictions

Table 2. Classificaton Report

|                   |precision |recall |f1-score |support | 
|-------------------|----------|------ |-------- |------- | 
|baby_bed           |0.96      |0.95   |0.96     |900     | 
|baby_car_seat      |0.94      |0.96   |0.95     |900     | 
|baby_folding_fence |0.99      |0.97   |0.98     |900     | 
|bathtub            |0.96      |0.87   |0.91     |900     | 
|booster_seats      |0.91      |0.87   |0.89     |900     |
|bouncer            |0.93      |0.87   |0.90     |900     | 
|breast_pump        |0.94      |0.96   |0.95     |900     | 
|carrier            |0.95      |0.95   |0.95     |900     | 
|earmuffs           |0.95      |0.97   |0.96     |900     | 
|ride_ons           |0.95      |0.92   |0.93     |900     | 
|rocking_horse      |0.93      |0.93   |0.93     |900     | 
|sterilizer         |0.94      |0.98   |0.96     |900     | 
|stroller           |0.95      |1.00   |0.97     |900     | 
|walkers            |0.88      |0.97   |0.92     |900     | 
|accuracy           |          |       |0.94     |4200    |
|macro avg          |0.94      |0.94   |0.94     |4200    | 
|weighted avg       |0.94      |0.94   |0.94     |4200    | 

In Tabel 2, the classification report provided above presents a comprehensive analysis of the performance of a classification model across various classes. The report includes key metrics such as precision, recall, F1-score, and support for each class. Below, we break down the components and insights derived from the report.

- `Precision` is the ratio of correctly predicted positive observations to the total predicted positives. It is an indicator of the model's accuracy in classifying positive samples.
- `Recall` is the ratio of correctly predicted positive observations to all observations in the actual class. It measures the model's ability to identify all relevant instances.
- `F1-score` is the weighted average of precision and recall. It considers both false positives and false negatives and is especially useful for imbalanced datasets.
- `Support` is the number of actual occurrences of the class in the dataset. All classes have equal support, indicating a balanced dataset.

**Overall Performance**

- `Accuracy`: The overall accuracy of the model is 0.94, indicating that 94% of all predictions are correct.
- `Macro Average`: The macro average precision, recall, and F1-score are all 0.94. This average treats all classes equally, providing an unweighted mean of the metrics.
- `Weighted Average`: The weighted average precision, recall, and F1-score are all 0.94. This average takes into account the support of each class, providing a more balanced view of the model's performance across all classes.

**Insights and Observations**

- `High Performance`: The model performs exceptionally well for most classes, with precision, recall, and F1-scores all above 0.90 for the majority of classes.
- `Areas for Improvement`: The bathtub, booster_seats, and bouncer classes have slightly lower recall scores (0.87), indicating room for improvement in detecting these classes correctly.
- `Stroller Class`: The stroller class has a perfect recall of 1.00, showing that all actual stroller instances were correctly identified by the model.


## **Image Prediction Visualization**

Figure 7. Image Prediction Viz.

![image_pred_viz](https://github.com/muhammadsahrul59/Movie-System-Recommendation/assets/101655285/9532fdf2-5f1e-468c-a1bd-bca6babe198c)

In Figure 7, it can be seen above there are examples of 50 images which show the predicted image (pred) and if it is correct it will match what is predicted and show green color. If the prediction does not match, it will show a mismatch with the prediction and show red.


