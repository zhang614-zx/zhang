from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for, send_file
import os
import uuid
from werkzeug.utils import secure_filename
from chatbot import ask_dashscope
# 假设模型推理函数在model.py中
from model import predict_all
import json
import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import time
from util.progress import write_progress

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mhd', 'raw', 'nii', 'nii.gz', 'dcm'}
HISTORY_FILE = 'history.json'
USERS_FILE = 'users.json'
PROGRESS_STATS_FILE = 'progress_stats.json'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key_here'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def root():
    if 'username' in session:
        return redirect('/predict')
    return redirect('/login')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# 登录态校验装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict')
@login_required
def predict_page():
    return render_template('predict.html', page='predict')

@app.route('/history')
@login_required
def history_page():
    return render_template('history.html', page='history')

@app.route('/stats')
@login_required
def stats_page():
    return render_template('stats.html', page='stats')

@app.route('/assistant')
@login_required
def assistant_page():
    return render_template('assistant.html', page='assistant')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '未检测到文件'}), 400
    files = request.files.getlist('file')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': '未选择文件'}), 400
    unique_id = str(uuid.uuid4())
    save_dir = os.path.join(app.config['UPLOAD_FOLDER'], unique_id)
    os.makedirs(save_dir, exist_ok=True)
    mhd_path = None
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(save_dir, filename)
            file.save(save_path)
            if filename.lower().endswith('.mhd'):
                mhd_path = save_path
        else:
            return jsonify({'error': f'文件格式不支持: {file.filename}'}), 400
    if not mhd_path:
        return jsonify({'error': '未上传.mhd文件'}), 400
    # 保存患者信息到session
    session['patient_info'] = {
        'name': request.form.get('patient_name', ''),
        'gender': request.form.get('patient_gender', ''),
        'age': request.form.get('patient_age', ''),
        'patient_id': request.form.get('patient_id', '')
    }
    return jsonify({'file_id': unique_id, 'path': mhd_path})

def save_history(record):
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        history.append(record)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print('保存历史记录失败:', e)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def update_progress_stats(stage_times):
    # stage_times: dict {stage: seconds}
    try:
        if os.path.exists(PROGRESS_STATS_FILE):
            with open(PROGRESS_STATS_FILE, 'r', encoding='utf-8') as f:
                stats = json.load(f)
        else:
            stats = {}
        for k, v in stage_times.items():
            if k not in stats:
                stats[k] = []
            stats[k].append(v)
            # 只保留最近50次
            stats[k] = stats[k][-50:]
        with open(PROGRESS_STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print('保存进度统计失败:', e)

def get_progress_stats():
    if os.path.exists(PROGRESS_STATS_FILE):
        with open(PROGRESS_STATS_FILE, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        avg = {k: (sum(v)/len(v) if v else 1) for k, v in stats.items()}
        return avg
    return {'seg': 10, 'nodule': 5, 'tumor': 5, 'ai': 2}  # 默认值


@app.route('/api/progress')
def api_progress():
    task_id = request.args.get('task_id')
    try:
        with open(f'progress_{task_id}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception:
        return jsonify({'step': 0, 'finished': False})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    file_path = data.get('path')
    file_id = data.get('file_id')  # 新增
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': '文件不存在'}), 400
    if not file_id:
        # 兼容老前端
        file_id = str(uuid.uuid4())
    try:
        patient_info = session.get('patient_info', {})
        # 传递 task_id 给 predict_all
        result, stage_times = predict_all(file_path, patient_info=patient_info, return_stage_times=True, task_id=file_id)
        record = {
            'id': str(uuid.uuid4()),
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'result': result,
            'patient_info': patient_info
        }
        save_history(record)
        session.pop('patient_info', None)
        update_progress_stats(stage_times)
        # 最后写 finished
        write_progress(file_id, 4, finished=True)
        return jsonify({'result': result, 'stage_times': stage_times})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': '问题不能为空'}), 400
    try:
        answer = ask_dashscope(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
def uploads_static(filename):
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    return send_from_directory(uploads_dir, filename)

@app.route('/vis/<path:filename>')
def vis_static(filename):
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    file_path = os.path.join(uploads_dir, filename)
    if os.path.exists(file_path):
        return send_from_directory(uploads_dir, filename)
    return 'Not Found', 404

@app.route('/api/history/list', methods=['GET'])
def history_list():
    history = load_history()
    # 只返回简要信息，并补充malignancy_label字段
    simple = [
        {
            'id': r['id'],
            'filename': r.get('filename', ''),
            'time': r.get('time', ''),
            'segmentation': r['result'].get('segmentation', ''),
            'malignancy': r['result'].get('malignancy', '未知'),
            'malignancy_label': r['result'].get('malignancy_label', None),
            'patient_info': r.get('patient_info', {})
        } for r in history
    ]
    return jsonify({'history': simple[::-1]})  # 新的在前

@app.route('/api/history/detail', methods=['GET'])
def history_detail():
    case_id = request.args.get('id')
    history = load_history()
    for r in history:
        if r['id'] == case_id:
            # 动态生成 nodule_type
            nodule_summary = r['result'].get('nodule_summary', {})
            high_prob = nodule_summary.get('high_prob_nodule_count', 0)
            total = nodule_summary.get('total', 0)
            if total == 0:
                nodule_type = '无结节'
            elif high_prob > 0:
                nodule_type = '高概率结节'
            else:
                nodule_type = '低概率结节'
            # 动态生成 malignancy
            malignancy_summary = r['result'].get('malignancy_summary', {})
            mal_num = malignancy_summary.get('恶性肿瘤数', 0)
            good_num = malignancy_summary.get('良性肿瘤数', 0)
            if mal_num > 0:
                malignancy = '恶性'
            elif good_num > 0:
                malignancy = '良性'
            else:
                malignancy = '未知'
            r['result']['nodule_type'] = r['result'].get('nodule_type', nodule_type)
            r['result']['malignancy'] = r['result'].get('malignancy', malignancy)
            return jsonify({'record': r})
    return jsonify({'error': '未找到该病例'}), 404

@app.route('/api/history/delete', methods=['POST'])
@login_required
def history_delete():
    data = request.json
    case_id = data.get('id')
    if not case_id:
        return jsonify({'error': '缺少id'}), 400
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = []
    new_history = [r for r in history if r.get('id') != case_id]
    if len(new_history) == len(history):
        return jsonify({'error': '未找到该病例'}), 404
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_history, f, ensure_ascii=False, indent=2)
    return jsonify({'success': True})

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    if not username or not password:
        return jsonify({'error': '用户名和密码不能为空'}), 400
    users = load_users()
    if any(u['username'] == username for u in users):
        return jsonify({'error': '用户名已存在'}), 400
    user = {
        'username': username,
        'password_hash': generate_password_hash(password),
        'register_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    users.append(user)
    save_users(users)
    return jsonify({'success': True})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    if not username or not password:
        return jsonify({'error': '用户名和密码不能为空'}), 400
    users = load_users()
    user = next((u for u in users if u['username'] == username), None)
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': '用户名或密码错误'}), 400
    session['username'] = username
    return jsonify({'success': True, 'username': username})

@app.route('/api/progress_stats')
def progress_stats():
    return jsonify(get_progress_stats())


@app.route('/api/preview', methods=['POST'])
def preview_ct():
    files = request.files.getlist('file')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': '未选择文件'}), 400

    import shutil
    temp_dir = os.path.join('uploads', 'preview_' + str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)
    mhd_path = None

    for file in files:
        filename = secure_filename(file.filename)
        save_path = os.path.join(temp_dir, filename)
        file.save(save_path)
        if filename.lower().endswith('.mhd'):
            mhd_path = save_path

    if not mhd_path:
        shutil.rmtree(temp_dir)
        return jsonify({'error': '未上传.mhd文件'}), 400

    try:
        import SimpleITK as sitk
        import matplotlib.pyplot as plt
        ct = sitk.ReadImage(mhd_path)
        arr = sitk.GetArrayFromImage(ct)
        img_urls = []

        for i in range(arr.shape[0]):
            png_path = os.path.join(temp_dir, f'ct_{i}.png')
            plt.imsave(png_path, arr[i], cmap='gray')
            rel_path = os.path.relpath(png_path, start=os.path.dirname(__file__))
            url_path = '/' + rel_path.replace('\\', '/').replace('uploads/', 'uploads/')
            img_urls.append(url_path)

        return jsonify({'imgs': img_urls})
    except Exception as e:
        shutil.rmtree(temp_dir)
        return jsonify({'error': f'灰度图像生成失败: {e}'})

@app.route('/api/report/preview', methods=['POST'])
@login_required
def preview_report():
    data = request.json
    case_id = data.get('id')
    if not case_id:
        return '缺少id', 400

    history = load_history()
    record = next((r for r in history if r['id'] == case_id), None)
    if not record:
        return '未找到该病例', 404

    html = generate_report_html(record, for_pdf=False)
    return html


def generate_report_html(record, for_pdf=False):
    patient_info = record.get('patient_info', {}) or {}
    result = record.get('result', {}) or {}
    nodule_summary = result.get('nodule_summary', {}) or {}
    malignancy_summary = result.get('malignancy_summary', {}) or {}
    time_str = record.get('time', '')
    vis_mask_list_key = result.get('vis_mask_list_key', []) or []
    vis_mask_list_all = result.get('vis_mask_list', []) or []
    all_slices = result.get('vis_all_slices', []) or []
    ai_diagnosis = result.get('ai_diagnosis', '')

    # 计算各种状态的数量
    nodule_results = result.get('nodule_results', []) or []
    benign_nodules = 0
    malignant_tumors = 0
    need_further_diagnosis = 0

    if nodule_results:
        nodule_threshold = nodule_summary.get('threshold', 0.5)
        malignant_threshold = malignancy_summary.get('threshold', 0.5)

        high_prob_nodules = [nodule for nodule in nodule_results
                             if nodule.get('prob_nodule', 0) >= nodule_threshold or
                             nodule.get('prob_malignant', 0) >= malignant_threshold]

        for nodule in high_prob_nodules:
            prob_nodule = nodule.get('prob_nodule', 0)
            prob_malignant = nodule.get('prob_malignant', 0)

            if prob_nodule >= 0.5 and prob_malignant < 0.5:
                benign_nodules += 1
            elif prob_nodule >= 0.5 and prob_malignant >= 0.5:
                malignant_tumors += 1
            elif prob_nodule < 0.5 and prob_malignant >= 0.5:
                need_further_diagnosis += 1

    # 关键切片索引集合
    key_slice_indices = set([item['slice'] for item in vis_mask_list_key]) if vis_mask_list_key else set()

    # 构建图片展示 - 智能选择策略
    ct_images = []

    if for_pdf:
        # PDF模式：智能选择切片
        selected_indices = set()

        # 1. 首先添加所有关键切片
        if vis_mask_list_key:
            for item in vis_mask_list_key:
                selected_indices.add(item['slice'])

        # 2. 如果关键切片少于20张，补充一些均匀分布的切片
        if all_slices and len(selected_indices) < 20:
            total_slices = len(all_slices)
            remaining_slots = min(20 - len(selected_indices), 15)  # 最多再加15张

            if remaining_slots > 0:
                # 均匀选择额外的切片
                step = max(1, total_slices // remaining_slots)
                for i in range(0, total_slices, step):
                    if len(selected_indices) < 20:
                        selected_indices.add(i)

        # 3. 按索引排序并生成图片列表
        sorted_indices = sorted(list(selected_indices))
        for idx in sorted_indices:
            if idx < len(all_slices):
                img_path = all_slices[idx]
                raw_path = img_path.replace("\\", "/").replace("uploads/", "")
                abs_path = os.path.abspath(os.path.join('uploads', raw_path))
                img_url = f'file:///{abs_path.replace(os.sep, "/")}'

                is_key = idx in key_slice_indices
                desc = f'切片{idx + 1}'
                if is_key:
                    desc += '（关键）'

                ct_images.append({
                    'url': img_url,
                    'slice': idx + 1,
                    'desc': desc,
                    'is_key': is_key
                })
    else:
        # 预览模式：显示所有切片
        if all_slices:
            for idx, img_path in enumerate(all_slices):
                raw_path = img_path.replace("\\", "/").replace("uploads/", "")
                img_url = '/vis/' + raw_path
                is_key = idx in key_slice_indices
                desc = f'切片号：{idx + 1}'
                if is_key:
                    desc += '（关键切片）'
                ct_images.append({
                    'url': img_url,
                    'slice': idx + 1,
                    'desc': desc,
                    'is_key': is_key
                })

    # 渲染图片HTML
    if ct_images:
        if for_pdf:
            # PDF模式：使用table布局，更兼容
            rows = []
            for i in range(0, len(ct_images), 4):  # 每行4个
                row_images = ct_images[i:i + 4]

                # 图片行
                img_cells = ''.join([
                    f'<td style="text-align:center;padding:5px;width:25%;"><img src="{img["url"]}" style="width:140px;height:auto;border:1px solid #ccc;border-radius:3px;"></td>'
                    for img in row_images
                ])
                # 补齐空单元格
                while len(row_images) < 4:
                    img_cells += '<td style="width:25%;"></td>'
                    row_images.append(None)

                # 标签行
                label_cells = ''.join([
                    f'<td style="text-align:center;padding:2px;font-size:9px;color:{"#e53935" if img and img["is_key"] else "#666"};{"font-weight:bold;" if img and img["is_key"] else ""}">{img["desc"] if img else ""}</td>'
                    for img in row_images
                ])

                rows.append(f'<tr>{img_cells}</tr><tr>{label_cells}</tr>')

            ct_images_html = f'<table style="width:100%;border-collapse:collapse;margin:15px 0;">{"".join(rows)}</table>'
        else:
            # 预览模式：正常布局
            ct_images_html = '<div class="ct-image-list">' + ''.join([
                f'<div class="ct-image-box"><img src="{img["url"]}"><div class="ct-image-desc"' +
                (" style=\"color:#e53935;\"" if img['is_key'] else "") +
                f'>{img["desc"]}</div></div>'
                for img in ct_images
            ]) + '</div>'
    else:
        ct_images_html = '<div class="ct-image-desc">无切片图像。</div>'

    # CSS样式
    if for_pdf:
        css_styles = '''
body { 
    font-family: 'Microsoft YaHei', 'SimSun', Arial, sans-serif; 
    background: #fff; 
    color: #222; 
    margin: 0; 
    padding: 15px; 
    font-size: 12px;
}
.report-container { max-width: 100%; margin: 0; }
.report-header { text-align: center; margin-bottom: 20px; }
.report-title { font-size: 18px; font-weight: bold; margin-bottom: 8px; }
.hospital-info { font-size: 14px; color: #1976D2; margin-bottom: 15px; }
.section-title { 
    font-size: 14px; font-weight: bold; color: #1976D2; 
    margin-top: 20px; margin-bottom: 10px; 
    border-left: 3px solid #1976D2; padding-left: 8px; 
}
.info-table, .result-table { 
    width: 100%; border-collapse: collapse; margin-bottom: 15px; font-size: 11px;
}
.info-table th, .info-table td, .result-table th, .result-table td { 
    border: 1px solid #333; padding: 4px 6px; 
}
.info-table th, .result-table th { background: #f0f0f0; font-weight: bold; }
.signature { margin-top: 25px; text-align: right; }
.signature-label { color: #888; font-size: 11px; }
'''
    else:
        css_styles = '''
body { 
    font-family: 'Microsoft YaHei', Arial, sans-serif; 
    background: #f8f9fa; color: #222; margin: 0; padding: 0; 
}
.report-container { 
    max-width: 900px; margin: 30px auto; background: #fff; 
    border-radius: 12px; box-shadow: 0 2px 16px #aaa; padding: 36px 48px;
}
.report-header { text-align: center; margin-bottom: 24px; }
.report-title { font-size: 2.2rem; font-weight: bold; letter-spacing: 2px; margin-bottom: 8px; }
.hospital-info { font-size: 1.1rem; color: #1976D2; margin-bottom: 18px; }
.section-title { 
    font-size: 1.2rem; font-weight: bold; color: #1976D2; 
    margin-top: 32px; margin-bottom: 12px; 
    border-left: 5px solid #1976D2; padding-left: 10px; 
}
.info-table, .result-table { width: 100%; border-collapse: collapse; margin-bottom: 18px; }
.info-table th, .info-table td, .result-table th, .result-table td { 
    border: 1px solid #bbb; padding: 8px 12px; 
}
.info-table th { background: #e3f2fd; font-weight: bold; }
.result-table th { background: #f5f7fa; }
.ct-image-list {
    display: flex; flex-wrap: wrap; justify-content: center;
    gap: 28px 18px; margin: 18px 0 0 0;
}
.ct-image-box { width: 180px; text-align: center; margin: 0; }
.ct-image-box img {
    max-width: 180px; border-radius: 8px; box-shadow: 0 2px 8px #ccc;
    border: 1.5px solid #1976D2; margin-bottom: 6px;
}
.ct-image-desc { color: #888; font-size: 0.98rem; margin-bottom: 12px; }
.signature { margin-top: 40px; display: flex; justify-content: flex-end; }
.signature-box { text-align: right; }
.signature-label { color: #888; }
'''

    # 动态设置切片部分标题
    if for_pdf:
        slice_title = f"关键及代表性CT切片（共{len(ct_images)}张）"
        slice_note = f"<p style='color: #666; font-style: italic; margin-top: 10px; font-size: 10px;'>注：包含所有关键切片及部分代表性切片，完整{len(all_slices) if all_slices else '所有'}张切片请查看系统在线报告。</p>" if ct_images else ""
    else:
        slice_title = f"全部CT切片（共{len(ct_images)}张）"
        slice_note = ""

    html = f'''
<html>
<head>
<meta charset="utf-8">
<title>肺部肿瘤CT分析报告</title>
<style>{css_styles}</style>
</head>
<body>
<div class="report-container">
  <div class="report-header">
    <div class="hospital-info">XX市人民医院 放射科</div>
    <div class="report-title">肺部肿瘤CT分析报告</div>
    <div style="color:#888;">报告编号：{record.get('id', '')} &nbsp; 检查日期：{time_str}</div>
  </div>

  <div class="section-title">患者信息</div>
  <table class="info-table">
    <tr><th>姓名</th><td>{patient_info.get('name', '')}</td><th>性别</th><td>{patient_info.get('gender', '')}</td></tr>
    <tr><th>年龄</th><td>{patient_info.get('age', '')}</td><th>住院号</th><td>{patient_info.get('patient_id', '')}</td></tr>
    <tr><th>检查时间</th><td colspan="3">{time_str}</td></tr>
  </table>

  <div class="section-title">影像所见</div>
  <p style="margin-bottom:15px;line-height:1.6;">{ai_diagnosis or '见AI分析结论'}</p>

  <div class="section-title">AI分析结论</div>
  <table class="result-table">
    <tr><th>结节数</th><td style="color:#42a5f5;font-weight:bold;">{benign_nodules}</td><th>恶性肿瘤数</th><td style="color:#ef5350;font-weight:bold;">{malignant_tumors}</td></tr>
    <tr><th>需进一步诊断数</th><td style="color:#4caf50;font-weight:bold;">{need_further_diagnosis}</td><th>总疑似结节数</th><td>{nodule_summary.get('total', 0)}</td></tr>
  </table>

  <p style="margin-bottom:15px;line-height:1.6;"><strong>诊断意见：</strong>本次AI分析共发现{nodule_summary.get('total', 0)}个疑似结节，其中结节{benign_nodules}个，恶性肿瘤{malignant_tumors}个，需进一步诊断{need_further_diagnosis}个，建议结合临床资料及影像随访。</p>

  <div class="section-title">{slice_title}</div>
  {ct_images_html}
  {slice_note}

  {"<div class='signature'>" if for_pdf else '<div class="signature"><div class="signature-box">'}
    <div class="signature-label">报告医生：{'_________________' if for_pdf else ''}</div>
    <div style="height:{'20px' if for_pdf else '32px'};"></div>
    <div class="signature-label">审核医生：{'_________________' if for_pdf else ''}</div>
    <div style="height:{'20px' if for_pdf else '32px'};"></div>
    <div class="signature-label">报告日期：{time_str}</div>
  {"</div>" if for_pdf else "</div></div>"}
</div>
</body>
</html>
'''
    return html


@app.route('/api/report/export', methods=['POST'])
@login_required
def export_report():
    import pdfkit
    import io

    data = request.json
    case_id = data.get('id')
    if not case_id:
        return jsonify({'error': '缺少id'}), 400

    history = load_history()
    record = next((r for r in history if r['id'] == case_id), None)
    if not record:
        return jsonify({'error': '未找到该病例'}), 404

    # 检查wkhtmltopdf路径
    wkhtmltopdf_path = r'D:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'

    if not os.path.exists(wkhtmltopdf_path):
        return jsonify({'error': 'wkhtmltopdf未找到，请检查安装路径'}), 500

    html = generate_report_html(record, for_pdf=True)

    try:
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None,
            'load-error-handling': 'ignore',
            'load-media-error-handling': 'ignore',
            'javascript-delay': 2000
        }

        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        pdf = pdfkit.from_string(html, False, options=options, configuration=config)

        return send_file(
            io.BytesIO(pdf),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='分析报告.pdf'
        )
    except Exception as e:
        print('PDF导出错误：', e)
        return jsonify({'error': f'PDF生成失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 