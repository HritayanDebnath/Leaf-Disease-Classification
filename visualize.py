import matplotlib.pyplot as plt
import numpy as np
import utils
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--MODEL")

    args = parser.parse_args()

    model_name = args.MODEL.strip().lower()

    if model_name == "vgg19":
        model_name = "VGG19"

    elif model_name == "resnet18":
        model_name = "ResNet18"

    elif model_name == "mnasnet":
        model_name = "MnasNet"
    
    elif model_name == "mobilenetv3":
        model_name = "MobileNetV3"

    elif model_name == "alexnet":
        model_name = "AlexNet"

    elif model_name == "shufflenetv2":
        model_name = "ShuffleNetv2"

    
    elif model_name == "squeezenet":
        model_name = "SqueezeNet"

    
    elif model_name == "efficientnet":
        model_name = "EfficientNet"

    else :
        raise Exception("Wrong Model.")


    data = utils.load_data(model_name)
    epoch = [float(e) for e in data["epoch"]]
    train_loss = [float(e) for e in data["train_loss"]]
    test_loss = [float(e) for e in data["test_loss"]]
    train_acc = [float(e) for e in data["train_acc"]]
    test_acc = [float(e) for e in data["test_acc"]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(epoch, train_loss, label='Train Loss', color="blue")
    ax1.plot(epoch, test_loss, label='Test Loss', color="red", linestyle="--")
    ax1.set_xlabel('Epochs')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(epoch, train_acc, label='Train Accuracy', color="blue")
    ax2.plot(epoch, test_acc, label='Test Accuracy', color="red", linestyle="--")
    ax2.set_xlabel('Epochs')
    ax2.set_title('Accuracy')
    
    ax2.legend()

    fig.suptitle(f"Model Summary : {model_name}")
    plt.show()