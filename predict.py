import torch
from torchvision import transforms
from PIL import Image
import utils
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--MODEL")
    parser.add_argument("--IMAGE")

    args = parser.parse_args()

    image = args.IMAGE
    model_name = args.MODEL.strip().lower()

    if model_name == "all":
        model_name = "all"
    elif model_name == "vgg19":
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

    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(image)
    image = transforms.Resize((224,224))(image)
    image_tensor = transforms.ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(dim=0).to(device)

    if model_name == "all":
        models =[model.split(".")[0] for model in os.listdir("models")]

    else:
        models = [model_name]

    for model_name in models:
        model = utils.load_model("models", f"{model_name}.pth")
        classes = model["class_names"]
        model = model["model"].to(device)
        pred = model(image_tensor)
        print(f"\nBy the model {model_name},", "\nThe image is predicted to be : " + classes[pred.argmax(dim=1)] + "\n")

