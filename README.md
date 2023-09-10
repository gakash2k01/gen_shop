# GEN SHOP 
<p id="readme-top"></p>

## Repository for Flipkart Grid 5.0 - Software Development Track

### Understanding of problem

Online shoppers grapple with finding products that match their preferences and staying updated with trends due to static images' limitations and the exhaustion of traditional keyword searches.
This project seeks to address these issues by leveraging Generative AI and conversational interactions to provide personalized recommendations, enhance visual representation, simplify product searches, and align with evolving trends and thus improving user satisfaction.

### Our Solution

- **Web Scraping** - For a given search, we’ll formulate the query for a specific topic and employ web scraping with Beautiful Soup to gather trending product data from the internet. This data will be fed into the model, ensuring that up-to-date trend information is integrated for improved recommendations.

- **Recommendation Engine** - User preferences stem from order and search history via fine-tuned ResNet50 model embeddings, augmented with ongoing trends, and ultimately informed by similarity metrics to offer tailored product recommendations.

- **Chatbot** - The interface is designed so the users can interact with the chatbot which will be then processed by the Alpaca-7b 16 bit quantised model. The model is supported by the recommendation system to generate responses tailored to user preferences.

- **Gen AI** - A stable diffusion model, fine tuned on e-commerce images along with captions, is used to generate images.  Trending data obtained through web scraping merged with the response from LLM is passed to the gen-AI model and is used  to create images that resonate with user preferences to create an effect of personalized visual search results.

_The designed web platform integrates the recommendation engine, LLM model and designed GenAI model using cloud virtual machine and fast API, to create an experience of simplified product search._

## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

#### Website
- node
```
Install latest node from "https://nodejs.org/en/"
```

- npm
 ```sh
 npm install npm@latest -g
```

#### ML Models

- python 3.10

```sh
https://www.python.org/downloads/
```

- cuda 12.1

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
```

- Atleast 30gb of GPU memory (18+12)

### Installation


1. Clone the repo

   ```sh
   git clone https://github.com/amRohitKumar/BalAsha-backend.git
   ```
2. Open project directory

    ```sh
    cd gen-shop
    ```
3. Install client packages

   ```sh
   cd server && npm install
   ```

4. Install server packages
   ```sh
   cd client && npm install
   ```

5. Install ML models 

    _make sure to crete new environment_

    ```sh
    cd models
    pip install -r requirements.txt
    ```

6. Dataset setup
- For the catalog we are using public Kaggle Dataset
    ```
    https://www.kaggle.com/datasets/dqmonn/zalando-store-crawl?sort=published
    ```
<!-- USAGE EXAMPLES -->

## Usage

<br>

1. To start website

    _Make sure you have nodemon package installed._
    
    - Client

   ```sh
   cd client
   npm run start-dev
   ```

    - Server

   ```sh
   cd server
   node app.js
   ```

   - Machine Learning
   ```sh
   python combined_deploy.py
   ```

   _use the cloud link in public_url.txt_
<br>

## UI and functionalities

The project encompasses several key components aimed at enhancing the user experience. Here's a detailed breakdown of its functionality:

- **Sign-In and Home Page:**
The project begins with a simple yet efficient sign-in page. Upon successfully logging in, users are greeted with a dynamic home page. This page offers a comprehensive overview of past interactions, showcasing previous conversations with an AI-driven chatbot. Additionally, the home page highlights a collection of personalized products meticulously curated based on the user's historical shopping activities.

- **Engaging Chat Experience:**
One of the project's focal points is its engaging chat functionality. Users can seamlessly initiate conversations with an advanced AI chatbot. This AI companion excels at generating images from textual commands, rendering the conversation not just text-based, but visually interactive. This interaction is further enhanced by the user's ability to tailor image features through written prompts. The dynamic "Update" button allows for real-time adjustments.

- **Image Customization and Suggestions:**
The user's involvement isn't confined to text alone. They possess the power to influence image generation by specifying desired attributes via written instructions. These customized prompts drive the AI to generate images that align more closely with the user's vision. Furthermore, if a generated image resonates with the user, they can seamlessly proceed to the "Pick" feature.

- **Product Matching and Personalization:**
The "Pick" feature acts as a bridge between the AI-generated content and the project's shopping aspect. Upon selecting a favored image, the system employs innovative technology to identify similar products within the catalog. This functionality draws connections between the user's aesthetic preferences and available merchandise, resulting in a curated selection that matches the user's distinct taste.

- **Shopping Integration and Personal Fashion Sense:**
Through the seamless integration of shopping capabilities, users can purchase items that align with their evolving fashion preferences. These selections are intricately linked to the user's interaction history, creating a dynamic loop where purchases continuously refine and enhance their personalized fashion sense.

In essence, the project revolves around creating a holistic user experience that seamlessly intertwines conversational AI, visual creativity, and personalized shopping. The user embarks on a journey of interactive conversations, image generation, product discovery, and fashion refinement, culminating in a uniquely tailored shopping experience that evolves alongside the user's preferences.

## Machine Learning Pipeline

- **Stable Diffusion**
The project utilizes the Stable Diffusion v1-5 model within an image-to-image pipeline. This approach allows trending outfits to be provided as input to the model, enabling it to generate images based on these trends. By incorporating trending fashion elements, the model can produce images that reflect the latest and most popular styles, enhancing the overall user experience.

- **Alpaca Lora 7b**
This repo uses a low-rank adapter for LLaMA-7b fit on the Stanford Alpaca dataset for genreating prompts.The model uses LLaMA, which is Meta’s large-scale language model. It uses OpenAI’s GPT (text-davinci-003) to fine-tune the 7B parameters-sized LLaMA model. It is free for academic and research purposes and has low computational requirements.Alpaca-LoRA uses Low-Rank Adaptation(LoRA) to accelerate the training of large models while consuming less memory.

- **Recommendation Engine:**
This repository employs ResNet-50 embeddings, which are utilized in a cosine similarity metric. This metric identifies the most visually similar images, enabling the recommendation engine to provide personalized product recommendations based on visual similarity. Resnet50 operates by passing an input image through multiple layers, known as residual blocks, which learn and extract features of increasing complexity.

## Screenshots

- Previous Chats
![Previous Chats](/public/image-1.png)

- Recommended products
![Recommended products](/public/image-2.png)

- Chat box
![Chat box](/public/image-3.png)

- Web Scrapping
![Web Scrapping](/public/image-4.png)

- Update image
![Update image](/public/image-5.png)

- Similar items
![Similar items](/public/image-6.png)

- Checkout page
![Checkout page](/public/image-7.png)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Devlopers:

1. Akash Gupta, B.Tech in Electronics and Communications Engineering
2. Apoorva Bhardwaj,B.Tech in Electrical Engineering
3. Rohit Kumar,B.Tech in Computer Science and Engineering
4. Tarun Shrivastava,M.Tech in Computer Science and Engineering 
