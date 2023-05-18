from pathlib import Path

import os
import numpy as np
import onnxruntime as rt
import urllib.request
from PIL import Image
import torchvision.transforms as transforms

BASE_DIR = Path(__file__).resolve(strict=True).parent

thresholding_lambda = lambda x: 1 if x > 0.5 else 0

sessOptions = rt.SessionOptions()
sessOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL 
model = rt.InferenceSession(os.path.join(BASE_DIR,"model.onnx"), sessOptions)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4303, 0.4301, 0.4139], std=[0.2186, 0.2140, 0.2205])
])

def predict(url):
    try:
        with urllib.request.urlopen(url) as f:
            image = Image.open(f)
        tensor_image = transform(image)
        tensor_image = tensor_image.numpy().astype(np.float32).reshape((1, 3, 224, 224))
        prediction = model.run([], {'input': tensor_image})[0]
        prediction = np.vectorize(thresholding_lambda)(prediction)
    except Exception as e:
        print(str(e))
        return None

    return prediction[0]


