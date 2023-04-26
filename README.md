### LEAF CLASSIFICATION USING DEEP LEARNING MODELS

In this project we have implemented 8 models namely 
* ResNet18
* EfficientNet
* MnasNet
* MobileNetV3
* ShuffleNetV2
* SqueezeNet
* AlexNet (not ready to use in the repository, but can be trained for usage)
* VGG19 (not ready to use in the repository, but can be trained for usage)


The dataset used to train these models is :-
    - https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset


To Train the for different dataset the files must be structured like this and in this format

    - Datasets
        - train
            - class_name1 
            - class_name2 
            - class_name3
            ...
        - test
            - class_name1 
            - class_name2 
            - class_name3
            ...


## To train with any of the built models,

Use this command in the root directory of this project

    - python train.py --MODEL [model_name]  --TRAINED  [True, False] --EPOCHS [num_of_epochs] --BATCH_SIZE [images_per_batch]

eg.

    - python train.py --MODEL alexnet --TRAINED True --EPOCHS 10 --BATCH_SIZE 16


here,
* --MODEL : name of the model to train
* --TRAINED : True if the model is already trained and re-training for better results, else False
* --EPOCHS : Number of epochs to be trained for
* --BATCH_SIZE : Number of images to be trained at one go.

## To predict with any of the built models,

Use this command in the root directory of this project

    - python predict.py --MODEL [model_name]  --IMAGE ["path/to/image"]

eg. 
    
    - python train.py --MODEL mobilenetv3 --IMAGE blueberry.jpg

here,

* --MODEL : name of the model to be used for prediction (can use 'all' to find prediction with all the trained models)
* --IMAGE : path to the image that will be used for the prediction.

    

## To visualize the history of the training process

Use this command in the root directory of this project 

    - python visualize.py --MODEL [model_name]

eg. 
    
    - python visualize.py --MODEL mobilenetv3

here,

* --MODEL : name of the model trained model

    