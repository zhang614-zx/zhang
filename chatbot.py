import os
import requests

def ask_dashscope(question):
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        raise Exception('未设置DASHSCOPE_API_KEY环境变量')
    url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'qwen-turbo',
        'input': {'prompt': question},
        'parameters': {'result_format': 'text'}
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result.get('output', {}).get('text', '未获取到答案')
    else:
        raise Exception(f'API请求失败: {response.text}') 