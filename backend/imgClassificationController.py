from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np
def build_imgclassification():
   processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
   model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
   return processor, model

from fastapi import HTTPException, UploadFile

def classification(image_path):
    try:
        processor, model = build_imgclassification()
        pil_img = Image.open(image_path).convert("RGB") 
        array_img = np.asarray(pil_img)
        inputs = processor(images=array_img, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]
        return {"text": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
