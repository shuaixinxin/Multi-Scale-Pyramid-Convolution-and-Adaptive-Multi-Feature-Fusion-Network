# 基于多尺度卷积神经网络的纺织品瑕疵检测研究

第四章 基于多比例空间金字塔卷积和自适应多尺度特征融合的纺织品瑕疵检测
该方法引入了可变形卷积(Deformable Convolution，DConv)模块、多比例空间金字塔卷积(Multi-proportion Spatial Pyramid Convolution，MSPC)结构和自适应多尺度特征融合(Adaptive Spatial Feature Fusion Network，ASFFN)策略来增强算法适应性。具体而言，在特征提取网络中用可变形卷积模块替代原卷积模块，使网络能动态调整卷积核大小，增强对不同尺寸瑕疵特征的捕捉能力；多比例空间金字塔卷积结构通过拼接多个1×3和3×1卷积核，提取更丰富的瑕疵特征信息，提高模型对不同比例尺度特征的敏感性；应用特征融合策略整合ASFF结构，高效捕捉多尺度的瑕疵信息。

第五章 基于深度可分离卷积和聚集与分配机制的纺织品瑕疵检测
本章提出了一种基于深度可分离卷积和聚集与分配发机制的纺织品瑕疵检测方法（Depthwise Separable Convolution and Gathering and Distribution Mechanism Network，DSC-GDNet）。方法流程如图5-1所示，通过采用聚集与分配（Gathering and Distribution，GD）机制的多尺度特征聚合对YOLOv8进行改进，以增强模型对不同尺度和形态瑕疵的特征提取能力；引入深度可分离卷积（Depthwise Separable Convolution，DSC）模块，有效降低计算复杂度和参数数量，提高模型的计算效率；同时，优化MDPIoU损失函数[88]，充分考虑目标的对角线距离，提高目标定位的准确性和精度。最后本章通过在天池纺织品瑕疵数据集和MS-COCO数据集上的实验验证，证明了该方法在提高纺织品瑕疵检测的准确性和速度方面具有显著优势，为纺织品质量检测提供了一种更加高效、可靠的解决方案


ultralytics\nn\AFPN.py与ultralytics\MS.py 为第四章的网络框架
ultralytics\nn\phd_benel_Glod_YOLO_DSC.py  为第五章的网络框架；
