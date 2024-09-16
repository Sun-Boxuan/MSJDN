import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

# len_x = 512
# mask_mat = np.ones((len_x, len_x))
#
# for line in range(1, 400, 2):
#     mask_mat[line, :] = 0
#
# img = Image.fromarray(mask_mat)
# grayscale_image = ImageOps.grayscale(img)
# # 将灰度图像转化成二值图像
# threshold = 0.5
# binary_image = grayscale_image.point(lambda p: p > threshold and 255)
#
# # 保存二值化图像
# binary_image.save('result.png')
#
# np.save('result2.npy', mask_mat)

# 制作数据的npy
# from shutil import copyfile
# from tqdm import tqdm
# from os.path import join
#
# old_ = "result.npy"
# new_dir = "data/datasets/denoise-mask-128-test2"
#
# for i in tqdm(range(100)):
#     new_path = join(new_dir, f"foward{i}.npy")
#     copyfile(old_, new_path)
#
# arr_mask = np.load('data/datasets/denoise-mask-128-test2/foward0.npy')
# arr_mask = arr_mask*255
# img = Image.fromarray(arr_mask)
# # img.show()
# if img.mode == 'F':
#     img = img.convert('RGB')
# img.save('sample.jpg', 'JPEG')
#
# print(arr_mask.shape)

# 制作mask文件
a = np.ones((128, 128))
b = (80, 81, 82, 83, 84, 86, 85, 71, 69, 65, 67, 66, 52, 49, 46, 44, 41, 39, 38, 36, 34, 33, 14, 17, 18, 19, 20, 22, 23, 24)

for line in b:
    a[line, :] = 0
np.save(r'E:\Software\RePaint-main\data\datasets\denoise-mask-128-real\foward1.npy', a)

# 调整实际数据的格式
# def read_csv(path):
#     data = np.loadtxt(path, delimiter=',')
#     x = data[:, 0]
#     y = data[:, 1]
#     data_list = data[:, 2]
#     data_list = np.array(data_list)
#     data_list = data_list.reshape(128, 128)
#     return np.array(x), np.array(y), data_list
#
#
# x, y, z = read_csv(r"C:\Users\DELL\Desktop\128.csv")
# z = z.reshape(128, 128)
# np.savetxt(r'C:\Users\DELL\Desktop\foward1', z, delimiter=',', fmt='%.6e')


# import numpy as np
#
# # 加载.npy文件
# data = np.load(r'E:\Software\RePaint-main\log\denoise_128_real\final\foward1.npy')
#
# # 将数据写入.txt文件
#
# fname4 = r'E:\Software\RePaint-main\log\denoise_128_real\final\foward1.txt'
# np.savetxt(fname4, data, fmt='%.6e', delimiter=',')