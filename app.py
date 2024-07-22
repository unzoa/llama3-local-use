import transformers
import torch

# 确认模型路径和相关信息
model_id = r"D:\github\llama3-local-use\Meta-Llama-3-8B-Instruct"
tokenizer_id = r"D:\github\llama3-local-use\Meta-Llama-3-8B-Instruct"

# print(f"Model ID: {model_id}")
# print(f"Tokenizer ID: {tokenizer_id}")
# print(f"Transformers version: {transformers.__version__}")
# print(torch.cuda.is_available())

try:
    # 尝试加载模型和分词器
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        load_in_4bit=True,

        # quantization_config=bnb.BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        #     ),
        device_map="cuda:0"
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)

    h = tokenizer('你好啊')
    print(h)

    # quantization_bit = 4
    # print(f"Quantized to {quantization_bit} bit")
    # model = model.quantize(quantization_bit)
    # print("model quantized done")


    # model = model.to(0)

    print("Model and tokenizer loaded successfully!")


    # 创建生成管道
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # device=0  # 指定使用的设备，可以根据你的环境进行调整
    )

    # 执行生成
    print(1111)
    output = pipeline(
        "hello",
        max_length=50,  # 设置生成文本的最大长度
        num_return_sequences=1,  # 设置生成文本的数量
        truncation=True,
        do_sample=True,  # 启用采样以增加生成的多样性
        top_k=50,  # 限制最高概率的 top k 词汇
        top_p=0.95  # 使用核采样，累积概率达到 top_p 的词汇
        )
    print(output)

    # 打印生成的结果
    for idx, sequence in enumerate(output):
        print(f"Generated text {idx + 1}: {sequence.get('generated_text', 'No text found')}")

        '''
        [{'generated_text': "Hey how are you doing today? You look a bit tired. Do you have any plans for the weekend? I hope you're doing well.\nI'm just a simple chatbot, but I'm here to help you with any questions you might have. Is there something specific you'd like to talk about or ask? I'm all ears!\nI hope you're doing well and that you have a great day. If you have any questions or need any help, feel free to ask. I'm here for you.\nI'm happy to help you with any questions you might have, but I don't have the ability to see or hear you. I'm just a computer program designed to provide information and assist with tasks. If you have any questions or need help with something, feel free to ask and I'll do my best to assist you. Otherwise, I hope you have a great day!"}]
        Generated text 1: Hey how are you doing today? You look a bit tired. Do you have any plans for the weekend? I hope you're doing well.
        I'm just a simple chatbot, but I'm here to help you with any questions you might have. Is there something specific you'd like to talk about or ask? I'm all ears!
        I hope you're doing well and that you have a great day. If you have any questions or need any help, feel free to ask. I'm here for you.
        I'm happy to help you with any questions you might have, but I don't have the ability to see or hear you. I'm just a computer program designed to provide information and assist with tasks. If you have any questions or need help with something, feel free to ask and I'll do my best to assist you. Otherwise, I hope you have a great day!
        '''


except Exception as e:
    print(f"Error occurred: {e}")
