import requests
import time
import threading
import json
import re

h = True

def generate_text(prompt, max_length=50, num_return_sequences=1):
    url = 'http://127.0.0.1:5000/generate'
    payload = {
        "prompt": prompt,
        "max_length": max_length,
        "num_return_sequences": num_return_sequences
    }
    response = requests.post(url, json=payload)

    h=False

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to generate text"}

def run_generate ():
    # prompt = "你叫什么名字"
    prompt = f"""
## 要求

- 内容前提，文本内容是一篇文章中的学员提取总结内容
- 需要你去理解并分析文本内容，整理其中的维度和内容信息
- 整理信息以json格式，数据格式以二维数组形式：
    - [["维度描述", "分数"，"答案内容，以一个段落描述"]]
- 回答的全部文本一定以中文语言返回

## 文本内容为

1、目标情况 :(10分)
某机构sss.com相关数据:

2、所在线路:(20分)
线路号1/2/3(秦材涉及4条线路);

3、数据分析情况:(30分)
1)组报还原了该目标邮件XX份，包括域名sss.com它
2)数据带有GRE外层封装:
外层IP: 用户侧111.222.33.x，通信侧:111.222.33.Y;
内层IP为 通信侧:444.555.66.*;
    """
    output = generate_text([{"role": "user", "content": prompt}], 400)
    # output = [{'generated_text': [{'role': 'user', 'content': '\n## 要求\n\n- 内容前提，文本内容是一篇文章中的学员提取总结内容\n- 需要你去理解并分析文本内容，整理其中的维度和内容信息\n- 整理信息以json格式，数据格式 以二维数组形式：\n    - [["维度描述", "维度的分数纯数字类型"，"答案内容，以一个段落描述"]]\n- 回答的 全部文本一定以中文语言返回\n\n## 文本内容为\n\n1、侦获目标情况 :(10分)\n侦获某机构sss.com相关数据:\n\n2、所在线路:(20分)\n线路号1/2/3(秦材涉及4条线路);\n\n3、数据分析情况:(30分)\n1)组报还原了该目标邮件XX份，包括域名sss.com它\n2)数据带有GRE外层封装:\n外层IP: 用户侧111.222.33.x，通信侧:111.222.33.Y;\n内 层IP为 通信侧:444.555.66.*;\n\n4、布控策路:(40分)\n(一)精控任务:\n1)邮件控守任务:\n布控条件: sss.com\n布控系统: 海网、327\n2)IP道采任务:\n布控条件:111.222.33.x、111.222.33.Y\n布控系统:海网\n(二)泛控任务:\n1)IP直采任务:\n布控条件:协议号47+邮件特征关键词;\n布控系统: 海网\n3) 落盘数据进一步筛选:\n筛选条件:sss.com\n    '}, {'role': 'assistant', 'content': 'Here is the analysis and summary of the text content in JSON format:\n\n```\n[\n  ["侦获目标情况", "integer", "检测到某机构sss.com相关数据"],\n  ["所 在线路", "integer", "线路号1/2/3，秦材涉及4条线路"],\n  ["数据分析情况", "integer", "组报还原了该目标邮件XX份，包括域名sss.com，数据带有GRE外层封装"],\n  ["布控策路", "integer", "布控任务包括精控任务和 泛控任务，精控任务包括邮件控守任务和IP道采任务，泛控任务包括IP直采任务和落盘数据进一步筛选"]\n]\n```\n\nNote that the "维度描述" column is a brief summary of the dimension, the "维度的分数纯数字类型" column specifies the data type of the dimension (in this case, an integer), and the "答案内容，以一个段落描述" column provides a paragraph-long description of the content.'}]}]
    an = process_api_response(output)[1]

    # print(an)
    print(an['content'])

    # 使用正则表达式提取 JSON 字符串
    try:
        # 针对 max_length: 8000
        json_string_match = re.search(r'```\n(.*?)\n```', an['content'], re.DOTALL)
    except:
        # 针对 max_length: 400
        json_string_match = re.search(r'\n\n(.*?)\n\n', an['content'], re.DOTALL)

    print(json_string_match)
    if json_string_match:
        json_string = json_string_match.group(1)
        # 解析 JSON 数据
        json_data = json.loads(json_string)
        print(json.dumps(json_data, indent=4, ensure_ascii=False))
    else:
        print("未找到 JSON 数据")

def loading ():
    while True:
        if (h):
            print("#", end="", flush=True)  # end="" 避免换行，flush=True 立即刷新输出
            time.sleep(1)  # 暂停1秒

def is_json(data):
    """
    判断数据是否为 JSON 格式。
    """
    try:
        json.loads(data)
    except (ValueError, TypeError):
        return False
    return True

def process_api_response(api_response):
    """
    处理 API 返回的数据，根据数据类型进行相应的处理。
    """
    processed_data = []

    for item in api_response:
        generated_texts = item.get('generated_text', [])
        for text_item in generated_texts:
            role = text_item.get('role')
            content = text_item.get('content', '').strip()

            # 判断 content 是否为 JSON 格式
            if is_json(content):
                content = json.loads(content)

            processed_data.append({
                'role': role,
                'content': content
            })

    return processed_data

def embedding (pp):
    url = 'http://127.0.0.1:5000/embedding'
    payload = {
        "prompt": pp,
    }
    response = requests.post(url, json=payload)

    h=False

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to generate text"}

def gg(code=''):
    url = 'http://127.0.0.1:5000/gg'
    payload = {
        "code": code,
    }
    response = requests.post(url, json=payload)

    h=False

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to generate text"}

if __name__ == "__main__":
    # 启动打印进度的线程
    progress_thread = threading.Thread(target=loading)
    progress_thread.daemon = True  # 设置为守护线程，主程序退出时该线程自动结束
    progress_thread.start()

    output = embedding('你好，今天你的心情怎么样？')
    print(output)
    # global_model, global_tokenizer
#     gg('''
# prompt = '你好，今天你的心情怎么样？'
# try:
#     model_id = r"D:\Desktop\hf-llama3\Meta-Llama-3-8B-Instruct"
#     tokenizer = transformers.LlamaTokenizerFast.from_pretrained(model_id)
#     o = tokenizer.encode("Hello this is a test")
#     print(type(o))


# except Exception as e:
#     print(e)
#     ''')




