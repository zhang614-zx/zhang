import argparse
import glob
import os
import sys

import numpy as np
import scipy.ndimage.measurements as measurements
import scipy.ndimage.morphology as morphology

import torch
import torch.nn as nn
import torch.optim

from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from segmentDsets import Luna2dSegmentationDataset
from TumorDatasets import LunaDataset, getCt, getCandidateInfoDict, getCandidateInfoList, CandidateInfoTuple
from segmentModel import UNetWrapper

import TumorModel

from util.logconf import logging
from util.util import voxelCoord2patientCoord

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)
#logging.getLogger("nodule.detect").setLevel(logging.WARNING)
logging.getLogger("tumor.detect").setLevel(logging.WARNING)
def print_confusion(label, confusions, do_mal):
    row_labels = ['非结节', '良性', '恶性']
    if do_mal:
        col_labels = ['', '漏诊', '排除', '预测为良性', '预测为恶性']
    else:
        col_labels = ['', '漏诊', '排除', '预测为结节']
        confusions[:, -2] += confusions[:, -1]
        confusions = confusions[:, :-1]
    cell_width = 16
    f = '{:>' + str(cell_width) + '}'
    print(label)
    print(' | '.join([f.format(s) for s in col_labels]))
    for i, (l, r) in enumerate(zip(row_labels, confusions)):
        r = [l] + list(r)
        if i == 0:
            r[1] = ''
        print(' | '.join([f.format(i) for i in r]))

def match_and_score(detections, truth, threshold=0.5):
    # 提取真实结节信息
    true_nodules = [c for c in truth if c.isNodule_bool]
    # 结节直径
    truth_diams = np.array([c.diameter_mm for c in true_nodules])
    # 结节中心坐标
    truth_xyz = np.array([c.center_xyz for c in true_nodules])

    # 提取检测结果信息，检测出的结节坐标
    detected_xyz = np.array([n[2] for n in detections])
    # 确定检测结果的预测类别（1=良性, 2=恶性, 3=其他）
    detected_classes = np.array([1 if d[0] < threshold
                                 else (2 if d[1] < threshold
                                       else 3) for d in detections])
    # 初始化3x4混淆矩阵
    # 行: 0=非结节, 1=良性结节, 2=恶性结节
    # 列: 0=完全漏检, 1=过滤排除, 2=预测良性, 3=预测恶性
    confusion = np.zeros((3, 4), dtype=int)
    # 处理特殊情况：无检测结果或无真实结节
    if len(detected_xyz) == 0:
        for tn in true_nodules:
            confusion[2 if tn.isMal_bool else 1, 0] += 1
    elif len(truth_xyz) == 0:
        for dc in detected_classes:
            confusion[0, dc] += 1
    # 主要匹配逻辑：计算真实结节与检测结果的距离
    else:
        # 计算每个真实结节与每个检测结果的归一化距离（距离/结节直径）
        normalized_dists = np.linalg.norm(truth_xyz[:, None] - detected_xyz[None], ord=2, axis=-1) / truth_diams[:, None]
        # 距离阈值0.7
        matches = (normalized_dists < 0.7)
        # 标记未匹配的检测结果
        unmatched_detections = np.ones(len(detections), dtype=bool)
        # 记录每个真实结节匹配到的最高置信度类别
        matched_true_nodules = np.zeros(len(true_nodules), dtype=int)
        # 处理匹配成功的情况
        for i_tn, i_detection in zip(*matches.nonzero()):
            matched_true_nodules[i_tn] = max(matched_true_nodules[i_tn], detected_classes[i_detection])
            unmatched_detections[i_detection] = False
        # 处理未匹配的检测结果（假阳性）
        for ud, dc in zip(unmatched_detections, detected_classes):
            if ud:
                confusion[0, dc] += 1
        # 处理真实结节的分类结果
        for tn, dc in zip(true_nodules, matched_true_nodules):
            confusion[2 if tn.isMal_bool else 1, dc] += 1
    return confusion

class NoduleAnalysisApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.debug(sys.argv)
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='设定加载量',
            default=16,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='设定用于后台加载数据的工作进程数',
            default=4,
            type=int,
        )

        parser.add_argument('--run-validation',
            help='是否对所有的CT影像进行验证',
            action='store_true',
            default=True,
        )
        parser.add_argument('--include-train',
            help="是否包含训练集(默认只对验证集进行验证)",
            action='store_true',
            default=False,
        )

        parser.add_argument('--segmentation-path',
            help="指定语义分割模型的路径",
            nargs='?',
            default='data-unversioned/seg/models/seg/seg_2025-06-30_16.38.36_none.best.state',
        )

        parser.add_argument('--cls-model',
            help="指定分类器模型的类名.",
            action='store',
            default='LunaModel',
        )
        parser.add_argument('--classification-path',
            help="指定保存分类器模型文件的地址",
            nargs='?',
            default='data-unversioned/nodule/models/nodule-model/cls_2025-06-27_09.24.34_nodule-comment.best.state',
        )

        parser.add_argument('--malignancy-model',
            help="指定恶性肿瘤分类器模型的类名",
            action='store',
            default='LunaModel',
        )
        parser.add_argument('--malignancy-path',
            help="指定保存恶性肿瘤分类器模型文件的地址",
            nargs='?',
            default='data-unversioned/tumor/models/tumor_cls/tumor_2025-07-01_10.12.02_finetune-depth2.best.state',
        )

        parser.add_argument('series_uid',
            nargs='?',
            default=None,
            help="指定要使用的CT影像文件的标识",
        )

        self.cli_args = parser.parse_args(sys_argv)
        # self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        if not (bool(self.cli_args.series_uid) ^ self.cli_args.run_validation):
            raise Exception("One and only one of series_uid and --run-validation should be given")


        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        #if not self.cli_args.segmentation_path:
        self.cli_args.segmentation_path = self.cli_args.segmentation_path

        #if not self.cli_args.classification_path:
        self.cli_args.classification_path = self.cli_args.classification_path

        self.seg_model, self.cls_model, self.malignancy_model = self.initModels()
    def initModels(self):
        log.debug(self.cli_args.segmentation_path)
        seg_dict = torch.load(self.cli_args.segmentation_path)

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

        log.debug(self.cli_args.classification_path)
        cls_dict = torch.load(self.cli_args.classification_path)

        model_cls = getattr(TumorModel, self.cli_args.cls_model)
        cls_model = model_cls()
        cls_model.load_state_dict(cls_dict['model_state'])
        cls_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)

            seg_model.to(self.device)
            cls_model.to(self.device)

        if self.cli_args.malignancy_path:
            model_cls = getattr(TumorModel, self.cli_args.malignancy_model)
            malignancy_model = model_cls()
            malignancy_dict = torch.load(self.cli_args.malignancy_path)
            malignancy_model.load_state_dict(malignancy_dict['model_state'])
            malignancy_model.eval()
            if self.use_cuda:
                malignancy_model.to(self.device)
        else:
            malignancy_model = None
        return seg_model, cls_model, malignancy_model


    def initSegmentationDl(self, series_uid):
        seg_ds = Luna2dSegmentationDataset(
                contextSlices_count=3,
                series_uid=series_uid,
                fullCt_bool=True,
            )
        seg_dl = DataLoader(
            seg_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return seg_dl

    def initClassificationDl(self, candidateInfo_list):
        cls_ds = LunaDataset(
                sortby_str='series_uid',
                candidateInfo_list=candidateInfo_list,
            )
        cls_dl = DataLoader(
            cls_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return cls_dl


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
        )
        val_set = set(
            candidateInfo_tup.series_uid
            for candidateInfo_tup in val_ds.candidateInfo_list
        )

        if self.cli_args.series_uid:
            series_set = set(self.cli_args.series_uid.split(','))
        else:
            series_set = set(
                candidateInfo_tup.series_uid
                for candidateInfo_tup in getCandidateInfoList()
            )

        if self.cli_args.include_train:
            train_list = sorted(series_set - val_set)
        else:
            train_list = []
        val_list = sorted(series_set & val_set)


        candidateInfo_dict = getCandidateInfoDict()
        series_iter = enumerateWithEstimate(
            val_list + train_list,
            "Series",
        )
        all_confusion = np.zeros((3, 4), dtype=int)
        for _, series_uid in series_iter:
            ct = getCt(series_uid)
            mask_a = self.segmentCt(ct, series_uid)

            candidateInfo_list = self.groupSegmentationOutput(
                series_uid, ct, mask_a)
            classifications_list = self.classifyCandidates(
                ct, candidateInfo_list)

            if not self.cli_args.run_validation:
                print(f"found nodule candidates in {series_uid}:")
                for prob, prob_mal, center_xyz, center_irc in classifications_list:
                    if prob > 0.5:
                        s = f"nodule prob {prob:.3f}, "
                        if self.malignancy_model:
                            s += f"malignancy prob {prob_mal:.3f}, "
                        s += f"center xyz {center_xyz}"
                        print(s)

            if series_uid in candidateInfo_dict:
                one_confusion = match_and_score(
                    classifications_list, candidateInfo_dict[series_uid]
                )
                all_confusion += one_confusion
                print_confusion(
                    series_uid, one_confusion, self.malignancy_model is not None
                )

        print_confusion(
            "Total", all_confusion, self.malignancy_model is not None
        )


    def classifyCandidates(self, ct, candidateInfo_list):
        cls_dl = self.initClassificationDl(candidateInfo_list)
        classifications_list = []
        for batch_ndx, batch_tup in enumerate(cls_dl):
            input_t, _, _, series_list, center_list = batch_tup

            input_g = input_t.to(self.device)
            with torch.no_grad():
                _, probability_nodule_g = self.cls_model(input_g)
                if self.malignancy_model is not None:
                    _, probability_mal_g = self.malignancy_model(input_g)
                else:
                    probability_mal_g = torch.zeros_like(probability_nodule_g)

            zip_iter = zip(center_list,
                probability_nodule_g[:,1].tolist(),
                probability_mal_g[:,1].tolist())
            for center_irc, prob_nodule, prob_mal in zip_iter:
                center_xyz = voxelCoord2patientCoord(center_irc,
                    direction_a=ct.direction_a,
                    origin_xyz=ct.origin_xyz,
                    vxSize_xyz=ct.vxSize_xyz,
                )
                cls_tup = (prob_nodule, prob_mal, center_xyz, center_irc)
                classifications_list.append(cls_tup)
        return classifications_list

    def segmentCt(self, ct, series_uid):
        with torch.no_grad():
            output_a = np.zeros_like(ct.hu_a, dtype=np.float32)
            seg_dl = self.initSegmentationDl(series_uid)  #  <3>
            for input_t, _, _, slice_ndx_list in seg_dl:

                input_g = input_t.to(self.device)
                prediction_g = self.seg_model(input_g)

                for i, slice_ndx in enumerate(slice_ndx_list):
                    output_a[slice_ndx] = prediction_g[i].cpu().numpy()

            mask_a = output_a > 0.5
            mask_a = morphology.binary_erosion(mask_a, iterations=1)

        return mask_a

    def groupSegmentationOutput(self, series_uid,  ct, clean_a):
        candidateLabel_a, candidate_count = measurements.label(clean_a)
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
            assert np.all(np.isfinite(center_irc)), repr(['irc', center_irc, i, candidate_count])
            assert np.all(np.isfinite(center_xyz)), repr(['xyz', center_xyz])
            candidateInfo_tup = \
                CandidateInfoTuple(False, False, False, 0.0, series_uid, center_xyz)
            candidateInfo_list.append(candidateInfo_tup)

        return candidateInfo_list

if __name__ == '__main__':
    NoduleAnalysisApp().main()
