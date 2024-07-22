import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
model_id = r"D:\Desktop\hf-llama3\LM_Studio_Community"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
else:
    device = torch.device("cpu")
    model.to(device)

# 输入文本
input_text = "Hey, how are you doing today?"

# 对输入文本进行分词
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 生成文本
with torch.no_grad():
    outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1)

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
