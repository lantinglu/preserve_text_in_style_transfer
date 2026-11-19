# Lanting Lu 的部分 / Team handoff

根据当前报告和 Notebook 的分工整理；原项目文件保留。

## 负责内容
- VGG-19 风格迁移：content loss、Gram style loss、L-BFGS 优化。
- Sobel edge-preservation loss 的实现与验证。
- baseline 与不同 edge weight 的公平对比、结果图及 edge/style loss 分析。

## 文件
- text_preserving_style_transfer.ipynb：独立 Notebook，主要负责 B（VGG/Sobel）及相关优化和 pilot 对比内容；其中 OCR 示例是参考代码，同学的正式工作区仍由对应成员完成。
- main.py：原始 VGG 风格迁移实现，也是 pilot.py 的依赖。
- edge_loss.py：Sobel 损失。
- pilot.py：共用实验流程；main_baseline.py、main_edge.py 是入口。
- imgs/、style.jpg：实验输入。
- pilot_output/：已有实验结果及日志，保留原目录结构，供报告引用。
- PILOT_README.md：运行方式和实验说明。
- test_edge_loss.py：已有验证。
- template-project-599——1.tex：完整共享报告的参考副本，包含其他同学的 TODO，不代表整篇都是个人工作。

## 报告中自己的内容
Methodology 下的 Image Preparation and Feature Extraction、Baseline Content and Style Losses、Sobel Edge-Preservation Loss、Implementation and Optimization；Experimental Setup 下的 Pilot Data and Edge-Weight Selection、Evaluation for Weight Tuning；Results 下的 Pilot Edge-Weight Sweep、Qualitative Comparison；Discussion 中的边缘与风格权衡及 pilot 局限；Statement of Individual Contribution 中 Lanting Lu 段。
Introduction、Abstract、Conclusion 等共同部分需要合并团队最终结果。

## 同学负责
- Ziyang Li：照片采集、角点标注、透视矫正、四条 pipeline 与完整测试集运行。
- Zijian Zhong：OCR、人工转录、CER 与拍摄角度影响分析。

## 运行脚本
在本文件夹打开终端：

    python -m pip install -r requirements.txt
    python pilot.py --edge-weights 0 1000 10000 100000 --output-dir new_pilot_output

使用新输出目录可保留附带的已有结果。首次运行需要下载 VGG-19 权重。
Notebook 自行安装依赖（包括 OCR 参考示例的 EasyOCR）；运行 Notebook 前，把第一段代码的 PROJECT_DIR 改为本文件夹的实际路径，避免仍指向原项目。

未复制虚拟环境、模型缓存及同学的独立 OCR 输出；这些并非个人代码交接所需。
