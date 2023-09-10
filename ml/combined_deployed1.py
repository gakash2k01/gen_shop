from fastapi import FastAPI
from pydantic import BaseModel
import json
import uvicorn
import random
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
import requests
import torch, pickle
from PIL import Image
from io import BytesIO
import base64
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import numpy as np
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
from torchvision.io import read_image
import os
from numpy.linalg import norm
import re
import cv2
import shutil
import pandas as pd

from requests import get
from bs4 import BeautifulSoup as soup
from concurrent.futures import ThreadPoolExecutor

from pydotmap import DotMap
while(True):
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
    except:
        print("Retrying importing diffusers.")
    else:
        break
while(True):
    try:
        from peft import PeftModel
    except:
        print("Retrying importing peft.")
    else:
        break


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input_img(BaseModel):
    inp : list

class model_input_lang(BaseModel):
    inp : list

class model_input_scrapper(BaseModel):
    inp : list

class model_input_upd(BaseModel):
    inp : list

class model_input_pick(BaseModel):
    inp : str

class model_input_home(BaseModel):
    inp : list

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

def img_generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

def lang_generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
)

def base642image(encoding):
    decoded_data = base64.b64decode(encoding)
    image_io = BytesIO(decoded_data)
    return Image.open(image_io).convert("RGB")

def img_evaluate(encodings, prompt):
    init_image = base642image(encodings)
    init_image = init_image.resize((768, 512))

    images = img_pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
    images[0].save("image_gen_output.png")
    my_string = ''
    with open('image_gen_output.png', 'rb') as img:
        my_string = base64.b64encode(img.read())
    return my_string

def lang_evaluate(instruction, input=None):
    prompt = lang_generate_prompt(instruction, input)
    inputs = lang_tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(lang_device)
    generation_output = lang_model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256
    )
    for s in generation_output.sequences:
        output = lang_tokenizer.decode(s)
        res = output.split("### Response:")[1].strip()
        print("Response:", res)
        return res

#############Recommender###################

class PinterestImageScraper:
    def __init__(self):
        self.json_data_list = []
        self.unique_img = []
    # ---------------------------------------- GET GOOGLE RESULTS ---------------------------------
    @staticmethod
    def get_pinterest_links(body, max_images):
        searched_urls = []
        html = soup(body, 'html.parser')
        links = html.select('#main > div > div > div > a')
        for link in links:
            link = link.get('href')
            link = re.sub(r'/url\?q=', '', link)
            if link[0] != "/" and "pinterest" in link:
                searched_urls.append(link)
                #stops adding links if the limit has been reached
                if max_images is not None and max_images == len(searched_urls):
                    break
        return searched_urls

    # -------------------------- save json data from source code of given pinterest url -------------
    def get_source(self, url, proxies):
        try:
            res = get(url, proxies=proxies)
        except Exception as e:
            return
        html = soup(res.text, 'html.parser')
        json_data = html.find_all("script", attrs={"id": "__PWS_DATA__"})
        for a in json_data:
            self.json_data_list.append(a.string)

    # --------------------------- READ JSON OF PINTEREST WEBSITE ----------------------
    def save_image_url(self, max_images):
        url_list = [i for i in self.json_data_list if i.strip()]
        if not len(url_list):
            return url_list
        url_list = []
        for js in self.json_data_list:
            try:
                data = DotMap(json.loads(js))
                urls = []
                for pin in data.props.initialReduxState.pins:
                    if isinstance(data.props.initialReduxState.pins[pin].images.get("orig"), list):
                        for i in data.props.initialReduxState.pins[pin].images.get("orig"):
                            urls.append(i.get("url"))
                    else:
                        urls.append(data.props.initialReduxState.pins[pin].images.get("orig").get("url"))

                for url in urls:
                    url_list.append(url)

                    #if the maximum has been achieved, return early
                    if max_images is not None and max_images == len(url_list):
                        return list(set(url_list))
                    

            except Exception as e:
                continue
        
        return list(set(url_list))

    # ------------------------------ image hash calculation -------------------------
    def dhash(self, image, hashSize=8):
        resized = cv2.resize(image, (hashSize + 1, hashSize))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

    # ------------------------------  save all downloaded images to folder ---------------------------
    def saving_op(self, var):
        url_list, folder_name = var
        if not os.path.exists(os.path.join(os.getcwd(), folder_name)):
                os.mkdir(os.path.join(os.getcwd(), folder_name))
        for img in url_list:
            result = get(img, stream=True).content
            file_name = img.split("/")[-1]
            file_path = os.path.join(os.getcwd(), folder_name, file_name)
            img_arr = np.asarray(bytearray(result), dtype="uint8")
            image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if not self.dhash(image) in self.unique_img:
                cv2.imwrite(file_path, image)
            self.unique_img.append(self.dhash(image))

    # ------------------------------  download images from image url list ----------------------------
    def download(self, url_list, num_of_workers, output_folder):
        idx = len(url_list) // num_of_workers if len(url_list) > 9 else len(url_list)
        param = []
        for i in range(num_of_workers):
            param.append((url_list[((i*idx)):(idx*(i+1))], output_folder))
        with ThreadPoolExecutor(max_workers=num_of_workers) as executor:
            executor.map(self.saving_op, param)

    # -------------------------- get user keyword and google search for that keywords ---------------------
    @staticmethod
    def start_scraping(max_images, key=None, proxies={}):
        assert key != None, "Please provide keyword for searching images"
        keyword = key + " pinterest"
        keyword = keyword.replace(" ","+")
        # keyword = keyword.replace("+", "%20")
        url = f'http://www.google.co.in/search?hl=en&q={keyword}'
        res = get(url, proxies=proxies,timeout=10)
        searched_urls = PinterestImageScraper.get_pinterest_links(res.content,max_images)
        return searched_urls, key.replace(" ", "_")


    def scrape(self, key=None, output_folder="", proxies={}, threads=10, max_images: int = None):
        extracted_urls, keyword = PinterestImageScraper.start_scraping(max_images,key, proxies)
        return_data = {}
        self.unique_img = []
        self.json_data_list = []

        for i in extracted_urls:
            self.get_source(i, proxies)

        # get all urls of images and save in a list
        url_list = self.save_image_url(max_images)

        return_data = {
            "isDownloaded": False,
            "url_list": url_list,
            "extracted_urls": extracted_urls,
            "keyword": key
        }

        # download images from saved images url
        if len(url_list):
            try:
                out_folder = output_folder if output_folder else key
                self.download(url_list, threads, out_folder)
            except KeyboardInterrupt:
                return return_data
            
            return_data["isDownloaded"] = True
            return return_data
        
        return return_data

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ScrappedDataSet(Dataset):
    
    def __init__(self,folder,transform=None,augment_transform=None):
        self.folder = folder
        self.transform = transform
        self.augment_trans = augment_transform
        self.items = os.listdir(self.folder)
        self.files = [item for item in self.items if os.path.isfile(os.path.join(self.folder, item))]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.folder,self.files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        if self.augment_trans:
            image = self.augment_transform(image)
           
        image = torch.mul(image, (1/255))
        return image,self.files[idx]

def make_embedding(model, path):
    dataset = ScrappedDataSet(path,transform)
    train_dataloader = DataLoader(dataset, batch_size=128)
    embedding = {}
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for inputs,fn in train_dataloader:
        inputs = inputs.to(device)
        inputs = inputs.type(torch.float)
        out = model(inputs)
        for idx in range(inputs.shape[0]):
            embedding[fn[idx]] = np.array(out[idx].cpu())
    return embedding

def cosine_sim(user_embedding,embedding,n=1):
    def sim(A,B):
        return np.dot(A,B)/(norm(A)*norm(B))
    sim_matrix = []
    for img in embedding:
        sim_matrix.append([sim(np.array(embedding[img]), user_embedding),img])
    sim_matrix.sort(reverse=True)
    return [sim_matrix[i][1] for i in range(n)]


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

def convert(item):
    item = item.strip()  # remove spaces at the end
    item = item[1:-1]    # remove `[ ]`
    item = np.fromstring(item, sep=' ')  # convert string to list
    return item

def scrapper_fn(keyword):
    # keyword = "Jordon_university_blue_og"   
    if(os.path.exists('./searches')):
        shutil.rmtree('./searches')
    max_images = 100
    if not os.path.exists(os.path.join(r'./searches',keyword)):
        os.makedirs(os.path.join(r'./searches',keyword))
    details = scraper.scrape(keyword, os.path.join(r'./searches',keyword),max_images=max_images)
    _, _, files = next(os.walk("/usr/lib"))
    file_count = len(files)
    for i in range(file_count, max_images):
        shutil.copyfile('./noise.png', f'./searches/{keyword}/noise.png')
        os.rename(f'./searches/{keyword}/noise.png', f'./searches/{keyword}/noise_{i}.png')

    if not details["isDownloaded"]:
        print("\nNothing to download !!") # return a blank image here



########### Recommender ends ######################
@app.post('/image_model')
def image_model(input_parameters : model_input_img):

    input_data = input_parameters.inp
    encoding = input_data[1]
    prompt = input_data[0]
    img = img_evaluate(encoding, prompt)
    return img

@app.post('/lang_model')
def lang_predd(input_parameters : model_input_lang):

    input_data = input_parameters.json()
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
    resp = lang_evaluate(instruction = query, input = "")
    summary_inp = "Summarize in 50 words: " + summary + chat + "Query: " + query + "Response: " + resp
    summary = lang_evaluate(summary_inp)
    return summary + "!?!?" + resp

@app.post('/scrapper')
def scrapper(input_parameters : model_input_scrapper):
    # keyword = "Jordon_university_blue_og"   
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    inp = input_dictionary['inp']
    user_embeddings = np.random.randn(2048) 
    # user_embeddings = inp[0]
    prompt = inp[1]

    scrapper_fn(keyword = prompt)
    embedding = make_embedding(model=model, path = f'searches/{prompt}')
    links = cosine_sim(user_embeddings,embedding,n=3)
    send = []
    for link in links:
        my_string = ""
        with open(os.path.join(r'./searches',prompt, link), 'rb') as img:
            my_string = base64.b64encode(img.read())
            send.append(my_string)
    return send

def recommend_picks(img, n = 10):
    send = []
    for i in range(n):
        send.append(img)
        send.append(f'image_id_{i+1}')
        send.append(f'image_{i+1}')
        send.append(random.randint(1000, 9999))
        send.append(random.randint(500, 999))
    return send

@app.post('/pick')
def pick(input_parameters : model_input_pick):
    input_data = input_parameters.inp
    img = input_data
    with open('catalog/embeds.pickle', 'rb') as handle:
        embeds = pickle.load(handle)
    img = base642image(img)
    img.save('./pick_option/output.jpg')
    pick_embed = make_embedding(model=model, path = './pick_option')
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