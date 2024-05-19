import nibabel as nib
import numpy as np

# 使用 nibabel 读取 NIfTI 文件
nib_image = nib.load('/home/dxw/Desktop/testImg.nii.gz')
"""
方向矩阵: [[ 1.  0. -0. -0.]
 [ 0.  1. -0. -0.]
 [ 0.  0.  1.  0.]
 [ 0.  0.  0.  1.]]
方向矩阵的轴序: ('R', 'A', 'S')
图像原点: [-0. -0.  0.]
像素间距: (1.0, 1.0, 1.0)
图像尺寸: (512, 512, 331)
像素类型: float64
图像数据类型: float64
图像数据形状: (512, 512, 331)
"""
# nib_image = nib.load('/home/dxw/Desktop/train_1_a_1.nii.gz')
"""
方向矩阵: [[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
方向矩阵的轴序: ('R', 'A', 'S')
图像原点: [0. 0. 0.]
像素间距: (1.0, 1.0, 1.0)
图像尺寸: (512, 512, 303)
像素类型: float64
图像数据类型: float64
图像数据形状: (512, 512, 303)
"""
# 获取图像数据
img_data = np.array(nib_image.dataobj)
direction = nib_image.affine
orientation =nib.aff2axcodes(direction)
origin = nib_image.header.get_qform()[0:3, 3]
spacing = nib_image.header.get_zooms()
# 获取图像的尺寸
size = nib_image.shape
# 获取图像的像素类型
pixel_type = img_data.dtype
print("方向矩阵:", direction)
print("方向矩阵的轴序:", orientation)
print("图像原点:", origin)
print("像素间距:", spacing)
print("图像尺寸:", size)
print("像素类型:", pixel_type)
print("图像数据类型:", img_data.dtype)
print("图像数据形状:", img_data.shape)
