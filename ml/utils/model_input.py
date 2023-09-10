from pydantic import BaseModel

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