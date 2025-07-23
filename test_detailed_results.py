#!/usr/bin/env python3
"""
测试详细结节数据生成功能
"""

import os
import sys
import json
from model import predict_all

def test_detailed_results():
    """测试详细结节数据生成"""
    
    # 查找一个测试用的mhd文件
    test_file = None
    for root, dirs, files in os.walk('uploads'):
        for file in files:
            if file.endswith('.mhd'):
                test_file = os.path.join(root, file)
                break
        if test_file:
            break
    
    if not test_file:
        print("未找到测试用的.mhd文件，请先上传一个CT文件")
        return
    
    print(f"使用测试文件: {test_file}")
    
    try:
        # 调用预测函数
        result, stage_times = predict_all(test_file, return_stage_times=True)
        
        print("\n=== 预测结果 ===")
        print(f"分割结果: {result['segmentation']}")
        print(f"结节总数: {result['nodule_summary']['total']}")
        print(f"高概率结节数: {result['nodule_summary']['high_prob_nodule_count']}")
        print(f"恶性肿瘤数: {result['malignancy_summary']['恶性肿瘤数']}")
        
        print("\n=== 详细结节数据 ===")
        if result['nodule_results']:
            for i, nodule in enumerate(result['nodule_results']):
                print(f"结节 {i+1}:")
                print(f"  结节概率: {nodule['prob_nodule']:.3f} ({nodule['prob_nodule']*100:.1f}%)")
                print(f"  恶性概率: {nodule['prob_malignant']:.3f} ({nodule['prob_malignant']*100:.1f}%)")
                print(f"  体素坐标: {nodule['center_irc']}")
                print(f"  物理坐标: {nodule['center_xyz']}")
                print(f"  切片位置: {nodule['center_irc'][0] + 1 if nodule['center_irc'] else '未知'}")
                
                # 判断状态
                is_high_prob = nodule['prob_nodule'] >= 0.5
                is_malignant = nodule['prob_malignant'] >= 0.5
                status = '高概率恶性' if is_high_prob and is_malignant else '高概率良性' if is_high_prob else '低概率'
                print(f"  状态: {status}")
                print()
        else:
            print("未检测到结节")
        
        print("\n=== 可视化文件 ===")
        if result['vis_mask_list']:
            print(f"生成了 {len(result['vis_mask_list'])} 个可视化文件")
            for i, vis_path in enumerate(result['vis_mask_list'][:3]):  # 只显示前3个
                print(f"  {i+1}: {vis_path}")
        else:
            print("未生成可视化文件")
        
        print("\n=== 处理时间 ===")
        for stage, time_taken in stage_times.items():
            print(f"  {stage}: {time_taken:.2f}秒")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detailed_results() 