# Parkinsons Disease Detection using Parkinsons Spiral Drawing
The project aims at predicting whether a person is suffering from Parkinson's Disease or not using Parkinson's Spiral Drawing test.
Prediction is made using a trained Convolutional Neural Network (CNN) trained on spiral drawings made by healthy people and people suffering from Parkinson's Disease.

### Parkinson's Disease
As mentioned on https://www.nia.nih.gov/health/parkinsons-disease - 
```
Parkinson's disease is a brain disorder that leads to shaking, stiffness, and difficulty with walking, balance, 
and coordination. Parkinson’s disease occurs when nerve cells in the basal ganglia, an area of the brain that 
controls movement, become impaired and/or die. Normally, these nerve cells, or neurons, produce an important 
brain chemical known as dopamine. When the neurons die or become impaired, they produce less dopamine, which 
causes the movement problems of Parkinson's. Scientists still do not know what causes cells that produce 
dopamine to die.
```

### Dataset
The dataset was taken from Kaggle - https://www.kaggle.com/kmader/parkinsons-drawings. The dataset on Kaggle consists of Spiral drawings and Wave Drawings for classfying people
as healthy or having Parkinson's disease. However in this project only Spiral Drawings are used. </br>
The dataset is also present in the 'dataset folder' as seperate '.npz' files for train and test set. </br>
Spiral Drawing made by a healthy person - </br>
![](https://github.com/Yuvnish017/Parkinsons_Disease_Detection_using_Parkinsons_Spiral_Drawing/blob/master/dataset/test_image_healthy.png?raw=True)

Spiral Drawing made by a person suffering from Parkinson's disease- </br> 
![](https://github.com/Yuvnish017/Parkinsons_Disease_Detection_using_Parkinsons_Spiral_Drawing/blob/master/dataset/test_image_parkinson.png?raw=True)

### Description of Files and Folders
* dataset folder - contains train and test set as well as two test images corresponding to each class for testing the model
* Parkinson's_Disease_Detection.ipynb - notebook containing code for dataset exploration, data augmentation, training model, visualization of results, and testing the model.
* parkinson's_disease_detection.py - python file corresponding to the notebook file.
* parkinson_disease_detection.h5 - trained model
* requirements.txt - contains list of libraries and packages needed for the project along with the used versions.
