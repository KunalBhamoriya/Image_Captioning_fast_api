from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import io
import os


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = image_path
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds



app = FastAPI(title="image-captioning", description="image-captioning")
templates = Jinja2Templates(directory="templates")

class ImageCaption(BaseModel):
  caption: str



@app.post("/predict", response_model=ImageCaption)
async def predict(request: Request, image: UploadFile = File(...)):

  # print(image.filename)
  # filename = image.filename
  # filepath = os.path.join("images/", filename)                  
  # image.save(filepath)
  
  contents = image.file.read()
  result = predict_step([Image.open(io.BytesIO(contents))])
  return templates.TemplateResponse("index.html", {"request": request, "caption": result[0]})



@app.get("/")
async def index(request: Request):
  return templates.TemplateResponse("index.html", {"request": request})






