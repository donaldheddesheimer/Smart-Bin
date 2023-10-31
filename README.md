# Smart Bin - [Work in Progress]
This project focuses on utilizing Convolutional Neural Networks (CNN) for waste classification, determining whether it is suitable for recycling or not. The process involves three key steps: initial data preparation and CNN model construction for waste recognition, enhancing the CNN model with data augmentation, and finally, implementing transfer learning with the pre-trained MobileNet V2 model to achieve optimal results.

![image](https://github.com/donaldheddesheimer/Smart-Bin/assets/119540065/cdc5091f-25ce-4854-8364-98309df95964)

## Overview
The primary goal of this initiative is to classify waste using Deep Neural Networks. The dataset comprises images of both recyclable and organic waste. A predictive model has been developed to assess whether the waste is recyclable or not. The analysis incorporates a Convolutional Neural Network (CNN) model with data augmentation and employs transfer learning to enhance prediction accuracy, ensuring the selection of the most effective model.

### Some correct predictions:

![image](https://github.com/donaldheddesheimer/Smart-Bin/assets/119540065/cf54bcd2-1834-40b2-8368-5b5216ef22d6)

### Some incorrect predictions:

![image](https://github.com/donaldheddesheimer/Smart-Bin/assets/119540065/943f0889-9da6-4e5b-9457-b79f7b0e252d)


## Data Collection
The dataset encompasses 22,500 images featuring organic and recyclable objects, sourced from Kaggle and accessible here.
Future iterations may customize the dataset to cater more specifically to distinct communities.

![image](https://github.com/donaldheddesheimer/Smart-Bin/assets/119540065/76a5bd1c-a7db-4426-9e8e-09e9a6a569f1)


## Executing the Program
This project is crafted using:

Python 3.8
Libraries: TensorFlow, Keras, Pillow, NumPy, Pandas, Seaborn.
To run the project, employ either Jupyter Notebook or Google Colab.
