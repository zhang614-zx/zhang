import math
from torch import nn as nn
from util.logconf import logging
import os
import torch
import numpy as np
from TumorDatasets import getCt, CandidateInfoTuple
from segmentDsets import Luna2dSegmentationDataset
from segmentModel import UNetWrapper
import TumorModel
import scipy.ndimage.measurements as measurements
import scipy.ndimage.morphology as morphology
from torch.utils.data import DataLoader
from util.util import voxelCoord2patientCoord, PatientCoordTuple
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib
import time
from util.progress import write_progress
matplotlib.use('Agg')

log = logging.getLogger(__name__)
log.setLevel(logging.WARN)
log.setLevel(logging.INFO)


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)
        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        conv_flat = block_out.view(
            block_out.size(0),
            -1,
        )
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_softmax(linear_output)

class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)

# 模块中追加一个测试类，方便模块功能测试
class modelCheck:
    def __init__(self,arg):
        self.arg = arg
        log.info("init {}".format(type(self).__name__))
    def main(self):
        model = LunaModel()  # 实例化模型
        # 统计参数总数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数: {total_params:,}")
        # 查看各层参数形状
        for name, param in model.named_parameters():
            print(f"{name}: 形状={param.shape}, 可训练={param.requires_grad}")

def predict_all(file_path, patient_info=None, return_stage_times=False, task_id=None):
    """
    输入: file_path (str) - 单个CT影像文件路径（.mhd）
    输出: dict - {segmentation, nodule_type, malignancy, ai_diagnosis}
    """
    stage_times = {}
    t0 = time.time()
    # 1. 直接读取上传的CT影像文件
    ct_mhd = sitk.ReadImage(file_path)
    ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
    ct_a.clip(-1000, 1000, ct_a)
    origin_xyz = PatientCoordTuple(*ct_mhd.GetOrigin())
    vxSize_xyz = PatientCoordTuple(*ct_mhd.GetSpacing())
    direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
    class CtObj:
        def __init__(self, hu_a, origin_xyz, vxSize_xyz, direction_a):
            self.hu_a = hu_a
            self.origin_xyz = origin_xyz
            self.vxSize_xyz = vxSize_xyz
            self.direction_a = direction_a
    ct = CtObj(ct_a, origin_xyz, vxSize_xyz, direction_a)
    series_uid = os.path.splitext(os.path.basename(file_path))[0]
    # 3. 加载分割模型
    seg_model_path = 'data-unversioned/seg/models/seg/seg_2025-06-30_16.38.36_none.best.state'
    seg_dict = torch.load(seg_model_path, map_location='cpu')
    seg_model = UNetWrapper(
        in_channels=7,
        n_classes=1,
        depth=3,
        wf=4,
        padding=True,
        batch_norm=True,
        up_mode='upconv',
    )
    seg_model.load_state_dict(seg_dict['model_state'])
    seg_model.eval()
    # 4. 分割推理
    with torch.no_grad():
        output_a = np.zeros_like(ct.hu_a, dtype=np.float32)
        seg_ds = Luna2dSegmentationDataset(contextSlices_count=3, series_uid=series_uid, fullCt_bool=True)
        seg_dl = DataLoader(seg_ds, batch_size=4, num_workers=0)
        for input_t, _, _, slice_ndx_list in seg_dl:
            input_g = input_t
            prediction_g = seg_model(input_g)
            for i, slice_ndx in enumerate(slice_ndx_list):
                output_a[slice_ndx] = prediction_g[i].cpu().numpy()
        mask_a = output_a > 0.5
        mask_a = morphology.binary_erosion(mask_a, iterations=1)
    t1 = time.time(); stage_times['seg'] = t1 - t0
    if task_id is not None:
        write_progress(task_id, 1)
    # 5. 提取候选结节
    candidateLabel_a, candidate_count = measurements.label(mask_a)
    centerIrc_list = measurements.center_of_mass(
        ct.hu_a.clip(-1000, 1000) + 1001,
        labels=candidateLabel_a,
        index=np.arange(1, candidate_count+1),
    )
    candidateInfo_list = []
    for i, center_irc in enumerate(centerIrc_list):
        center_xyz = voxelCoord2patientCoord(
            center_irc,
            ct.origin_xyz,
            ct.vxSize_xyz,
            ct.direction_a,
        )
        candidateInfo_tup = CandidateInfoTuple(False, False, False, 0.0, series_uid, center_xyz)
        candidateInfo_list.append(candidateInfo_tup)
    t2 = time.time(); stage_times['nodule'] = t2 - t1
    if task_id is not None:
        write_progress(task_id, 2)
    # 6. 加载结节分类模型
    cls_model_path = 'data-unversioned/nodule/models/nodule-model/cls_2025-06-27_09.24.34_nodule-comment.best.state'
    cls_dict = torch.load(cls_model_path, map_location='cpu')
    model_cls = getattr(TumorModel, 'LunaModel')
    cls_model = model_cls()
    cls_model.load_state_dict(cls_dict['model_state'])
    cls_model.eval()
    # 7. 加载良恶性模型
    malignancy_model_path = 'data-unversioned/tumor/models/tumor_cls/tumor_2025-07-01_10.12.02_finetune-depth2.best.state'
    malignancy_dict = torch.load(malignancy_model_path, map_location='cpu')
    malignancy_model = model_cls()
    malignancy_model.load_state_dict(malignancy_dict['model_state'])
    malignancy_model.eval()
    # 8. 分类推理
    from TumorDatasets import LunaDataset
    cls_ds = LunaDataset(sortby_str='series_uid', candidateInfo_list=candidateInfo_list)
    cls_dl = DataLoader(cls_ds, batch_size=4, num_workers=0)
    nodule_results = []  # 每个结节的结节概率和肿瘤恶性概率
    for batch_ndx, batch_tup in enumerate(cls_dl):
        input_t, _, _, series_list, center_list = batch_tup
        input_g = input_t
        with torch.no_grad():
            _, probability_nodule_g = cls_model(input_g)
            _, probability_mal_g = malignancy_model(input_g)
        for i, center_irc in enumerate(center_list):
            prob_nodule = probability_nodule_g[i,1].item()
            prob_malignant = probability_mal_g[i,1].item()
            # 计算物理坐标
            if center_irc is not None and len(center_irc) >= 3:
                center_xyz = voxelCoord2patientCoord(
                    center_irc,
                    ct.origin_xyz,
                    ct.vxSize_xyz,
                    ct.direction_a,
                )
                nodule_results.append({
                    "prob_nodule": prob_nodule,
                    "prob_malignant": prob_malignant,
                    "center_irc": [int(center_irc[0]), int(center_irc[1]), int(center_irc[2])],  # 体素坐标
                    "center_xyz": [float(center_xyz.x), float(center_xyz.y), float(center_xyz.z)]  # 物理坐标
                })
            else:
                # 如果center_irc为None或长度不足，使用默认值
                nodule_results.append({
                    "prob_nodule": prob_nodule,
                    "prob_malignant": prob_malignant,
                    "center_irc": [0, 0, 0],  # 默认体素坐标
                    "center_xyz": [0.0, 0.0, 0.0]  # 默认物理坐标
                })
    # 统计summary
    nodule_threshold = 0.5
    malignant_threshold = 0.5
    nodule_total = len(nodule_results)
    high_prob_nodule_count = sum(1 for n in nodule_results if n["prob_nodule"] >= nodule_threshold)
    malignant_count = sum(1 for n in nodule_results if n["prob_malignant"] >= malignant_threshold)
    benign_count = sum(1 for n in nodule_results if n["prob_malignant"] < malignant_threshold)
    max_prob_nodule = max((n["prob_nodule"] for n in nodule_results), default=None)
    min_prob_nodule = min((n["prob_nodule"] for n in nodule_results), default=None)
    avg_prob_nodule = sum((n["prob_nodule"] for n in nodule_results), 0.0) / nodule_total if nodule_total else None
    max_prob_malignant = max((n["prob_malignant"] for n in nodule_results), default=None)
    min_prob_malignant = min((n["prob_malignant"] for n in nodule_results), default=None)
    avg_prob_malignant = sum((n["prob_malignant"] for n in nodule_results), 0.0) / nodule_total if nodule_total else None
    
    # 9. 双列表方案：保存所有采样切片和关键切片的图片
    vis_mask_list_all = []  # 所有步长采样切片
    vis_mask_list_key = []  # 关键结节/肿瘤切片
    vis_all_slices = []     # 新增：所有原始切片图片路径
    try:
        total_slices = ct.hu_a.shape[0]
        step = max(1, total_slices // 20)
        vis_dir = os.path.join(os.path.dirname(file_path), 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        # 0. 保存所有原始切片图片
        for i in range(total_slices):
            ct_slice = ct.hu_a[i]
            raw_path = os.path.join(vis_dir, f'{series_uid}_all_{i}.png')
            if not os.path.exists(raw_path):
                plt.imsave(raw_path, ct_slice, cmap='gray')
            vis_all_slices.append(os.path.relpath(raw_path, start=os.path.dirname(os.path.abspath(__file__))))
        # 1. 保存所有步长采样切片
        for i in range(0, total_slices, step):
            ct_slice = ct.hu_a[i]
            mask_slice = mask_a[i]
            # 保存原始灰度图像
            raw_path = os.path.join(vis_dir, f'{series_uid}_{i}.png')
            if not os.path.exists(raw_path):
                plt.imsave(raw_path, ct_slice, cmap='gray')
            # 保存掩码可视化图像
            plt.figure(figsize=(8, 8))
            plt.imshow(ct_slice, cmap='gray')
            plt.imshow(mask_slice, cmap='Reds', alpha=0.4)
            plt.axis('off')
            plt.title(f'切片 {i+1}', fontsize=12, color='white')
            vis_path = os.path.join(vis_dir, f'{series_uid}_vis_{i}.png')
            plt.savefig(vis_path, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()
            vis_path_rel = os.path.relpath(vis_path, start=os.path.dirname(os.path.abspath(__file__)))
            vis_mask_list_all.append(vis_path_rel)
        # 2. 保存关键切片（有高概率结节/恶性肿瘤）
        slice_indices = set()
        for nodule in nodule_results:
            if (nodule["prob_nodule"] >= nodule_threshold) or (nodule["prob_malignant"] >= malignant_threshold):
                if nodule["center_irc"] and len(nodule["center_irc"]) >= 1:
                    slice_indices.add(nodule["center_irc"][0])
        for i in sorted(slice_indices):
            ct_slice = ct.hu_a[i]
            mask_slice = mask_a[i]
            # 保存原始灰度图像
            raw_path = os.path.join(vis_dir, f'{series_uid}_{i}.png')
            if not os.path.exists(raw_path):
                plt.imsave(raw_path, ct_slice, cmap='gray')
            # 保存掩码可视化图像
            plt.figure(figsize=(8, 8))
            plt.imshow(ct_slice, cmap='gray')
            plt.imshow(mask_slice, cmap='Reds', alpha=0.4)
            # 标记当前切片上的高概率结节
            slice_nodules = [n for n in nodule_results if ((n["prob_nodule"] >= nodule_threshold) or (n["prob_malignant"] >= malignant_threshold)) and n["center_irc"][0] == i]
            for idx, nodule in enumerate(slice_nodules):
                if nodule["center_irc"] and len(nodule["center_irc"]) >= 3:
                    y, x = nodule["center_irc"][1], nodule["center_irc"][2]
                    color = 'red' if nodule["prob_malignant"] >= malignant_threshold else 'yellow'
                    marker = 'x' if nodule["prob_malignant"] >= malignant_threshold else 'o'
                    plt.scatter(x, y, c=color, marker=marker, s=100, linewidths=2, 
                               label=f'结节{idx+1}' if nodule["prob_malignant"] >= malignant_threshold else None)
                    # 添加文本标签
                    prob_text = f'{nodule["prob_nodule"]:.2f}'
                    if nodule["prob_malignant"] >= malignant_threshold:
                        prob_text += f'\n{nodule["prob_malignant"]:.2f}'
                    plt.text(x+10, y-10, prob_text, color='white', fontsize=8, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            plt.axis('off')
            plt.title(f'切片 {i+1}', fontsize=12, color='white')
            if any(n["prob_malignant"] >= malignant_threshold for n in slice_nodules):
                plt.legend(loc='upper right', fontsize=10)
            vis_path = os.path.join(vis_dir, f'{series_uid}_vis_{i}.png')
            plt.savefig(vis_path, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()
            vis_path_rel = os.path.relpath(vis_path, start=os.path.dirname(os.path.abspath(__file__)))
            vis_mask_list_key.append({"slice": i, "path": vis_path_rel})
    except Exception as e:
        print(f"可视化生成失败: {e}")
        vis_mask_list_all = []
        vis_mask_list_key = []
        vis_all_slices = []
    # 10. 返回结构化结果
    result = {
        'segmentation': f'检测到{candidate_count}个结节',
        'nodule_results': nodule_results,
        'nodule_summary': {
            'total': nodule_total,
            'high_prob_nodule_count': high_prob_nodule_count,
            'threshold': nodule_threshold,
            'max_prob_nodule': max_prob_nodule,
            'min_prob_nodule': min_prob_nodule,
            'avg_prob_nodule': avg_prob_nodule
        },
        'malignancy_summary': {
            '恶性肿瘤数': malignant_count,
            '良性肿瘤数': benign_count,
            'threshold': malignant_threshold,
            'max_prob_malignant': max_prob_malignant,
            'min_prob_malignant': min_prob_malignant,
            'avg_prob_malignant': avg_prob_malignant
        },
        'vis_mask_list': vis_mask_list_all,  # 兼容原有前端
        'vis_mask_list_key': vis_mask_list_key,  # 关键切片列表，含切片号和路径
        'vis_all_slices': vis_all_slices      # 新增：全部原始切片图片路径
    }
    # 11. 自动生成AI诊断建议
    try:
        from chatbot import ask_dashscope
        gender = (patient_info or {}).get('gender', '未知')
        age = (patient_info or {}).get('age', '未知')
        prompt = f"患者信息：\n性别：{gender} 年龄：{age}\n分析结果：{result['segmentation']}。结节预测：高概率结节{high_prob_nodule_count}个/共{nodule_total}个。肿瘤预测：恶性肿瘤{malignant_count}个/共{nodule_total}个。\n请用专业医学语言为医生生成一段肺部肿瘤影像AI辅助诊断建议，内容包括：1. 主要发现和结论；2. 风险分层或分级（如有）；3. 建议的随访或进一步检查方案。要求简明、专业、可直接用于报告。"
        ai_diagnosis = ask_dashscope(prompt)
        result['ai_diagnosis'] = ai_diagnosis
    except Exception as e:
        result['ai_diagnosis'] = f'AI诊断建议生成失败：{e}'
    t3 = time.time(); stage_times['tumor'] = t3 - t2
    if task_id is not None:
        write_progress(task_id, 3)
    t4 = time.time(); stage_times['ai'] = t4 - t3
    if task_id is not None:
        write_progress(task_id, 4)
    if return_stage_times:
        return result, stage_times
    return result

if __name__ == "__main__":
    checkmodel = modelCheck('参数检查').main()
