from fastapi import FastAPI
import json
import uvicorn
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
import torch, pickle
import base64
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import numpy as np
from torchvision import transforms

import torchvision
import os
import pandas as pd


from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel

from utils.model_input import model_input_img, model_input_lang, model_input_scrapper, model_input_upd, model_input_pick, model_input_home
from utils.imagen_fn import img_evaluate
from utils.lang_fn import lang_evaluate
from utils.utils import base642image
from utils.scrapper import PinterestImageScraper, Identity, ScrappedDataSet, make_embedding, cosine_sim, scrapper_fn

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

img_device = "cuda:1"
lang_device = "cuda:0"

model_id_or_path = "runwayml/stable-diffusion-v1-5"
img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float32)
img_pipe = img_pipe.to(img_device)

lang_tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

lang_model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map=lang_device,
)

lang_model = PeftModel.from_pretrained(lang_model, "tloen/alpaca-lora-7b")

lang_model = lang_model.to(lang_device)

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
)

#############Recommender###################

scraper = PinterestImageScraper()

model = torchvision.models.resnet50(weights= torchvision.models.resnet.ResNet50_Weights.DEFAULT)
model.fc = Identity()
for params in model.parameters():
    params.requires_grad = False

transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(255),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                                transforms.CenterCrop(224)])

########### Routes ######################
@app.post('/image_model')
def image_model(input_parameters : model_input_img):

    input_data = input_parameters.inp
    encoding = input_data[1]
    prompt = input_data[0]
    img = img_evaluate(encoding, prompt, img_pipe)
    return img

@app.post('/lang_model')
def lang_predd(input_parameters : model_input_lang):

    input_data = input_parameters.model_dump_json()
    input_dictionary = json.loads(input_data)

    inp = input_dictionary['inp']
    summary = inp[0]
    query = inp[-1]
    n = len(inp)
    chat = ""
    for i in range(1,n-1):
        if(i%2 == 1):
            chat += "Query: "
            chat += inp[i]
        else:
            chat += "Response: "
            chat += inp[i]
    info = "Previous conversation summary: " + summary + "Previous " + str((n//2)-1) + " chats: " + chat
    resp = lang_evaluate(lang_model, lang_tokenizer, generation_config, lang_device, instruction = query, input = "")
    summary_inp = "Summarize in 50 words: " + summary + chat + "Query: " + query + "Response: " + resp
    summary = lang_evaluate(lang_model, lang_tokenizer, generation_config, lang_device, summary_inp)
    return summary + "!?!?" + resp

@app.post('/scrapper')
def scrapper(input_parameters : model_input_scrapper):
    # keyword = "Jordon_university_blue_og"   
    input_data = input_parameters.model_dump_json()
    input_dictionary = json.loads(input_data)

    inp = input_dictionary['inp']
    user_embeddings = np.random.randn(2048) 
    # user_embeddings = inp[0]
    prompt = inp[1]

    scrapper_fn(keyword = prompt, scraper = scraper)
    embedding = make_embedding(model=model, path = f'searches/{prompt}', transform = transform)
    links = cosine_sim(user_embeddings,embedding,n=3)
    send = []
    for link in links:
        my_string = ""
        with open(os.path.join(r'./searches',prompt, link), 'rb') as img:
            my_string = base64.b64encode(img.read())
            send.append(my_string)
    return send

@app.post('/pick')
def pick(input_parameters : model_input_pick):
    input_data = input_parameters.inp
    img = input_data
    with open('catalog/embeds.pickle', 'rb') as handle:
        embeds = pickle.load(handle)
    img = base642image(img)
    img.save('./pick_option/output.jpg')
    pick_embed = make_embedding(model=model, path = './pick_option', transform = transform)
    f  = cosine_sim(pick_embed['output.jpg'], embeds,n=12)
    df = pd.read_csv('catalog/data.csv',header=None)
    g = [df[df[0]==idx].values.tolist() for idx in f]
    h = [i for idx in g for i in idx]
    send = []
    for i in range(len(h)):
        my_string = ""
        with open(os.path.join(r'catalog/products_images', h[i][0] + '.jpg'), 'rb') as img:
            my_string = base64.b64encode(img.read())
            send.append(my_string)
            for j in range(4):
                send.append(h[i][j])
    return send

@app.post('/recommend_homepage')
def pick_homepage(input_parameters : model_input_home):
    input_data = input_parameters.inp
    user_embeddings = input_data
    with open('catalog/embeds.pickle', 'rb') as handle:
        embeds = pickle.load(handle)

    f  = cosine_sim(user_embeddings, embedding = embeds,n=20)
    df = pd.read_csv('catalog/data.csv',header=None)
    g = [df[df[0]==idx].values.tolist() for idx in f]
    h = [i for idx in g for i in idx]
    send = []
    for i in range(len(h)):
        my_string = ""
        with open(os.path.join(r'catalog/products_images', h[i][0] + '.jpg'), 'rb') as img:
            my_string = base64.b64encode(img.read())
            send.append(my_string)
            for j in range(4):
                send.append(h[i][j])
    return send

@app.post('/update_encodings')
def update_encodings(input_parameters : model_input_upd):
    input_data = input_parameters.inp
    print(input_data)
    input_data = np.array(input_data)
    with open('catalog/embeds.pickle', 'rb') as handle:
        embeds = pickle.load(handle)
    purchase_emb = [embeds[item] for item in input_data]
    purchase_avg = np.mean(purchase_emb,axis=0).tolist()
    assert len(purchase_avg)==2048 
    return purchase_avg

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
with open("public_url.txt", mode="wt") as f:
    f.write(ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)