from flask import Flask, request, jsonify
import transformers
import torch
import re

app = Flask(__name__)

# 全局变量，用于存储模型和分词器
global_model = None
global_tokenizer = None
global_encode = None

def load_model_and_tokenizer(model_id, tokenizer_id):
    global global_model, global_tokenizer

    # 检查是否已经加载过模型和分词器
    if global_model is None or global_tokenizer is None:
        print(f"Loading model and tokenizer from {model_id} and {tokenizer_id}...")
        global_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map=0
            )
        # print('=====================', global_model, '==================')
        global_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)
        # global_encode = transformers.PreTrainedTokenizerFast.from_pretrained(tokenizer_id)
        print("Model and tokenizer loaded successfully!")

@app.route('/generate', methods=['POST'])
def generate_text():
    global global_model, global_tokenizer

    # 确保模型和分词器已加载
    if global_model is None or global_tokenizer is None:
        return jsonify({"error": "Model and tokenizer are not loaded."}), 500

    print('模型和分词器已加载')

    data = request.json
    prompt = data.get("prompt", "")
    max_length = data.get("max_length", 50)
    num_return_sequences = data.get("num_return_sequences", 1)
    print('==============\n', data, prompt, max_length, '========\n')

    # 创建生成管道
    print('创建生成管道')
    pipeline = transformers.pipeline(
        "text-generation",
        model=global_model,
        tokenizer=global_tokenizer,
        # device=0  # 确保指定正确的 GPU 设备
    )

    print(pipeline.model.device)

    # 生成文本
    print('准备生成文本')
    output = pipeline(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        truncation=True,
        # temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        min_p=0.05,

        # max_tokens=-1,
        # presence_penalty=0,
        # frequency_penalty=0,
        # logit_bias={},
        # repeat_penalty=1.1,
        # seed=-1
    )

    print('生成完毕！')
    # print(output)

    return jsonify(output)

@app.route('/embedding', methods=['POST'])
def embedding ():
    global global_tokenizer, global_model

    data = request.json
    prompt = data.get("prompt", "")
    print(prompt)

    try:
        model_id = r"D:\github\llama3-local-use\Meta-Llama-3-8B-Instruct"
        tokenizer = transformers.LlamaTokenizerFast.from_pretrained(model_id)
        o = tokenizer.encode
        print(o)

        return jsonify(f'''{o}''')
    except Exception as e:
        return({"error": f"Error occurred: {e}"})


@app.route('/gg', methods=['POST'])
def gg():
    global global_model, global_tokenizer

    try:
        # 获取请求中的 Python 代码
        code = request.json.get('code')
        print('==========')
        print(code)
        print('==========')

        # 创建一个字典来保存局部变量
        local_vars = {}

        # 创建一个字典来保存全局变量
        global_vars = globals().copy()

        # 执行代码
        exec(code, global_vars, local_vars)

        # 获取执行结果
        result = local_vars.get('result', 'No result variable defined.')

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/haha', methods=['GET'])
def haha():
    return jsonify({ "code": 200, "message": '22222' })

if __name__ == '__main__':
    model_id = r"D:\github\llama3-local-use\Meta-Llama-3-8B-Instruct"
    tokenizer_id = r"D:\github\llama3-local-use\Meta-Llama-3-8B-Instruct"


    load_model_and_tokenizer(model_id, tokenizer_id)
    app.run(host='0.0.0.0', port=5000)
