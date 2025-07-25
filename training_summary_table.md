# 4. 实训总结

## 4.1 实训内容的复杂性评价

本次实训的工程复杂性评价如表 4.1 所示：

**表 4.1 工程复杂性评价表**

| 复杂工程问题特征 | 问题描述及解决方案 (方法) |
|------------------|--------------------------|
| 必须运用深入的工程原理，经过分析才可能得到解决 | **问题描述：** 肺部肿瘤CT扫描影像的3D卷积神经网络模型设计需要深入理解深度学习原理、卷积神经网络架构和医学影像特征。<br><br>**解决方案：** 通过系统学习PyTorch框架、3D卷积神经网络原理，结合医学影像数据特点，设计适合肺部肿瘤检测的深度学习模型架构。 |
| 涉及多方面的技术、工程和其它因素，并可能相互有一定冲突 | **问题描述：** 项目涉及医学影像处理、深度学习算法、数据预处理、模型训练优化、可视化展示等多个技术领域，各模块间存在性能与精度的权衡。<br><br>**解决方案：** 采用模块化设计，平衡模型精度与计算效率，通过数据增强、缓存机制、可视化工具等综合技术手段解决冲突。 |
| 需要通过建立合适的抽象模型才能解决，在建模过程中需要体现出创造性 | **问题描述：** 肺部肿瘤检测需要将复杂的3D医学影像数据抽象为可训练的深度学习模型，需要创新性地处理数据不平衡、小样本学习等问题。<br><br>**解决方案：** 创造性设计3D卷积神经网络架构，结合数据增强技术、注意力机制等创新方法，建立适合医学影像特征的抽象模型。 |
| 不是仅靠常用方法就可以完全解决的 | **问题描述：** 肺部肿瘤检测面临数据稀缺、标注困难、假阳性高等挑战，传统方法难以达到临床要求。<br><br>**解决方案：** 综合运用深度学习、数据增强、模型集成、后处理优化等先进技术，结合医学专业知识，开发专门针对肺部肿瘤检测的解决方案。 |
| 问题中涉及的因素可能没有完全包含在专业工程实践的标准和规范中 | **问题描述：** 医学AI领域发展迅速，缺乏统一的标准规范，需要处理医学伦理、数据隐私、模型可解释性等新兴问题。<br><br>**解决方案：** 参考医学影像AI最新研究进展，结合临床需求，建立适合项目的技术规范和评估标准，确保模型的可信度和实用性。 |
| 问题相关各方利益不完全一致 | **问题描述：** 项目涉及开发团队、医学专家、患者等多方利益，对模型精度、速度、可解释性等要求存在差异。<br><br>**解决方案：** 通过需求分析、原型验证、专家反馈等环节，平衡各方需求，设计满足临床实际应用的综合解决方案。 |
| 具有较高的综合性，包含多个相互关联的子问题 | **问题描述：** 肺部肿瘤检测系统包含数据预处理、模型训练、结果可视化、性能评估等多个相互依赖的子模块，需要整体协调。<br><br>**解决方案：** 采用系统工程方法，建立完整的开发流程，通过Tensorboard可视化、本地缓存机制、模块化设计等确保各子系统的协调运行。 |

## 4.2 实训体会、收获与建议 