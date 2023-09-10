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

def lang_evaluate(lang_model, lang_tokenizer, generation_config, lang_device, instruction, input=None):
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