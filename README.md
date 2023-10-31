# Smart Bin - [Work in Progress]
This project focuses on utilizing Convolutional Neural Networks (CNN) for waste classification, determining whether it is suitable for recycling or not. The process involves three key steps: initial data preparation and CNN model construction for waste recognition, enhancing the CNN model with data augmentation, and finally, implementing transfer learning with the pre-trained MobileNet V2 model to achieve optimal results.

![image](https://github.com/donaldheddesheimer/Smart-Bin/assets/119540065/3ee38c9e-36de-49c2-80df-b1c00b7e9915)


## Overview
The primary goal of this initiative is to classify waste using Deep Neural Networks. The dataset comprises images of both recyclable and organic waste. A predictive model has been developed to assess whether the waste is recyclable or not. The analysis incorporates a Convolutional Neural Network (CNN) model with data augmentation and employs transfer learning to enhance prediction accuracy, ensuring the selection of the most effective model.

### Some correct predictions:

![image](https://github.com/donaldheddesheimer/Smart-Bin/assets/119540065/3fef2f0c-2615-40f1-979f-e52da3c9808d)
![image](https://github.com/donaldheddesheimer/Smart-Bin/assets/119540065/08139702-da74-44da-844f-22e00ac0c6f8)


### Some incorrect predictions:

![image](https://github.com/donaldheddesheimer/Smart-Bin/assets/119540065/e4d0bd40-bf9a-405a-bfef-9d0c401c85d8)


## Data Collection
The dataset encompasses 22,500 images featuring organic and recyclable objects, sourced from Kaggle and accessible here.
Future iterations may customize the dataset to cater more specifically to distinct communities.

![image](https://github.com/donaldheddesheimer/Smart-Bin/assets/119540065/507846a0-24e9-4e08-a7fb-6cd55a4a5484)


## Executing the Program
This project is crafted using:

Python 3.8
Libraries: TensorFlow, Keras, Pillow, NumPy, Pandas, Seaborn.
To run the project, employ either Jupyter Notebook or Google Colab.
