from PIL import Image
from imageai.Classification import ImageClassification
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import json
import os
 

def mobilnet_v2(img_dir, n=3):
    exec_path = os.getcwd()

    prediction = ImageClassification()
    prediction.setModelTypeAsMobileNetV2()
    prediction.setModelPath(os.path.join(exec_path, 'mobilenet_v2-b0353104.pth'))
    prediction.loadModel()
    
    predictions, probabilities = prediction.classifyImage(os.path.join(exec_path, img_dir), result_count=n)

    results = []
    for eachPred, eachProb in zip(predictions, probabilities):
        results.append(f'{eachPred.capitalize()}: {eachProb:.1f}% ')
    print("mobilenet_v2")

    return ', '.join(results)


def simple_CNN(img_dir, n=3):
    with open("names_dict.json") as f:
        class_translation = json.load(f)
        
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Linear(16 * 112 * 112, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    model = SimpleCNN(num_classes=len(class_translation))
    model.load_state_dict(torch.load("./cnn_model_trained.pth"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_image = Image.open(img_dir)
    input_tensor = transform(input_image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = nn.functional.softmax(output[0], dim=0)
    top_probabilities, top_class_indices = torch.topk(probabilities, k=n)

    results = []
    for prob, class_idx in zip(top_probabilities, top_class_indices):
        class_label = str(class_idx.item())
        class_name = class_translation.get(class_label, "Unknown")
        results.append(f"{class_name}: {prob*100:.1f}%")
    print("CNN")

    return ', '.join(results)


def resnet_50(img_dir, n=3):
    # Load the translation JSON file
    with open("names_dict.json") as f:
        class_translation = json.load(f)

    # Load the trained model
    model = tf.keras.models.load_model('./resnet_model_tf.h5')

    # Load and preprocess the image for prediction
    img_path = img_dir  # Replace with the path to your image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image data

    # Perform classification
    predictions = model.predict(img_array)
    top_n = n  # Number of top predictions to display

    # print(class_labels)
    # Get the indices of the top N predictions
    top_indices = np.argsort(predictions[0])[::-1][:top_n]

    results = []
    for i in range(top_n):
        class_label = class_translation.get(str(top_indices[i]), "Unknown")
        probability = predictions[0][top_indices[i]]
        results.append(f"{class_label}: {probability*100:.1f}%")

    return ', '.join(results)
