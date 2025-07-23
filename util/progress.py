import json

def write_progress(task_id, step, finished=False):
    try:
        with open(f'progress_{task_id}.json', 'w', encoding='utf-8') as f:
            json.dump({'step': step, 'finished': finished}, f)
    except Exception as e:
        print('写入进度失败:', e) 