import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

#本地模型路径
model_path = r"D:\github\llama3-local-use\Meta-Llama-3-8B-Instruct"
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
else:
    print('没有GPU')

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
if model_path.endswith("4bit"):
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map='auto'
        )
elif model_path.endswith("8bit"):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map='auto'
        )
else:
    model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

prompt = instruction.format("Hello, what the meaning of life？")
generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096, streamer=streamer)