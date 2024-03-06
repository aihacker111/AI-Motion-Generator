import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)


def transform(init_image, textPrompt, strength=0.5, guidance_scale=15):
    init_image = Image.open(init_image).convert("RGB")
    init_image = init_image.resize((768, 512))
    images = pipe(prompt=textPrompt, image=init_image, strength=strength, guidance_scale=guidance_scale).images
    image = images[0]
    image.save("astronaut_rides_horse.png")
    return image

if __name__ == '__main__':
    image = transform(init_image='', textPrompt='A cute chinese girl with beauty dress')
