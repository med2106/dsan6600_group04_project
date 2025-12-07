import os
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from PIL import Image
import boto3
import io


# 1. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Get model state dictionary from S3 bucket
s3_client = boto3.client('s3')
bucket_name = 'med2106-neural-nets-hair-project'
object_key = "FINALMODEL_weights.pth"
try:
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    body = response["Body"].read()
    print("Downloaded .pth file from S3")
    state_dict = torch.load(io.BytesIO(body), map_location=device)
    print("Unpacked the state_dict from .pth file")
    if "classifier.1.fc.weight" in state_dict:
        state_dict["classifier.1.weight"] = state_dict.pop("classifier.1.fc.weight")
        state_dict["classifier.1.bias"] = state_dict.pop("classifier.1.fc.bias")
except Exception as e:
    print(f"Error downloading .pth: {e}")
    raise

# 3. Re-build the model
weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
model = efficientnet_v2_m(weights=weights)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 9)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Model Loaded Sucessfully!")

labels = ['1', '2a', '2b', '2c', '3a', '3b', '3c', '4a', '4b', '4c']

# def standardize_image(uploaded_file):
#     raw_image = uploaded_file.read().convert("RGB")
#     processed_image = transforms.Compose([
#     transforms.Resize((600, 600)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
#     ])
#     # transforms.CenterCrop(224),

#     return processed_image

def standardize_image(uploaded_file):
    pil_image = Image.open(uploaded_file).convert("RGB")
    processed_image = transforms.Compose([
        transforms.Resize((600, 600)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return processed_image(pil_image)

def predict_hair_type(logits):
    probs = torch.sigmoid(logits)
    class_idx = (probs > 0.5).sum(dim=1).item()
    return labels[class_idx]

# def predict_hair_type(logits):
#     probs = torch.softmax(logits, dim=1)
#     class_idx = torch.argmax(probs, dim=1).item()
#     return labels[class_idx]

def classify_image(image_path):
    # load and transform image
    image = standardize_image(image_path).unsqueeze(0).to(device)
    print("image transformed")
    # ensure model is ready to evaluate an image
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        # _, predicted = torch.max(outputs, 1)
        # print("Model Predicted something")
        # pred_class = predict_hair_type(predicted)
        pred_class = predict_hair_type(outputs)
    return pred_class

image = "application/streamlit_app/3B_square_2_480x480.jpg"

pred_class = classify_image(image)
print("Predicted Class:", pred_class)