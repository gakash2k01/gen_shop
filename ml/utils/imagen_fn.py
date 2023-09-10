from utils.utils import base642image
import base64

def img_evaluate(encodings, prompt, img_pipe):
    init_image = base642image(encodings)
    init_image = init_image.resize((768, 512))

    images = img_pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
    images[0].save("image_gen_output.png")
    my_string = ''
    with open('image_gen_output.png', 'rb') as img:
        my_string = base64.b64encode(img.read())
    return my_string