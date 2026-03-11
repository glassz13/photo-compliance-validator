import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

ATTRS = ["Eyeglasses", "Wearing_Hat", "Narrow_Eyes", "Smiling", "Male"]

RULES = {
    "Eyeglasses"  : {"label": "Eyeglasses",   "pass_when": 0, "fail_msg": "Remove glasses for ID photos."},
    "Wearing_Hat" : {"label": "Headwear",      "pass_when": 0, "fail_msg": "Remove hat or headwear."},
    "Narrow_Eyes" : {"label": "Eyes Open",     "pass_when": 0, "fail_msg": "Keep your eyes fully open."},
    "Smiling"     : {"label": "Expression",    "pass_when": 0, "fail_msg": "Maintain a neutral expression."},
    "Male"        : {"label": "Gender",        "pass_when": None, "fail_msg": ""},
}

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_model(weights_path: str):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, len(ATTRS)),
    )
    checkpoint = torch.load(weights_path, map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def predict(model, image_path: str) -> dict:
    image  = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze().tolist()

    predictions = {}
    for attr, prob in zip(ATTRS, probs):
        predicted = 1 if prob >= 0.5 else 0
        rule      = RULES[attr]

        if rule["pass_when"] is None:
            # metadata only — no pass/fail
            predictions[attr] = {
                "label"  : rule["label"],
                "value"  : "Male" if predicted == 1 else "Female",
                "passed" : None,
                "prob"   : round(prob, 3),
                "msg"    : "",
            }
        else:
            passed = (predicted == rule["pass_when"])
            predictions[attr] = {
                "label"  : rule["label"],
                "value"  : "Yes" if predicted == 1 else "No",
                "passed" : passed,
                "prob"   : round(prob, 3),
                "msg"    : "" if passed else rule["fail_msg"],
            }

    return predictions
