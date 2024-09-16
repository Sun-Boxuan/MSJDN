from Taper2d import taper2d, antitaper2d
import numpy as np
from matplotlib import pyplot as plt
from Cosinto0 import cosinto0

def grid_plot(ax, z, separate_number=12):
    global x, y
    x_mesh, y_mesh = np.meshgrid(x, y)
    z = z.reshape(256*256)
    length_x = len(x)
    length_z = len(z)
    times = length_z // length_x
    ellipses = [z[0:length_x]]
    ax.set_xlabel("x/m")
    ax.set_ylabel("y/m")
    for num in range(1, times):
        ellipses += [z[length_x * num:length_x * (num + 1)]]

    return ax.contourf(x_mesh, y_mesh, ellipses, separate_number, cmap='jet')

def plot_mat(mat, title=''):
    fig = plt.figure()
    plt.suptitle(title)
    ax = fig.add_subplot(111)
    ax = grid_plot(ax, mat)
    cb = fig.colorbar(ax)
    cb.set_label('nT', labelpad=-26, y=1.05, rotation=0)

# data_path = "./data/datasets/Dataset-forward-model2.2/foward45"
# data45 = np.loadtxt(data_path, delimiter=',')
# data45 = cosinto0(data45, 100, 100, 78, 78)
# data_path = "./data/datasets/Dataset-double-model2.2/foward45"
# np.savetxt(data_path, data45, delimiter=',')
#
# data_path = "./data/datasets/Dataset-forward-model2.2/foward60"
# data60 = np.loadtxt(data_path, delimiter=',')
# data60 = cosinto0(data60, 100, 100, 78, 78)
# data_path = "./data/datasets/Dataset-double-model2.2/foward60"
# np.savetxt(data_path, data60, delimiter=',')
#
# data_path = "./data/datasets/Dataset-forward-model2.2/foward90"
# data90 = np.loadtxt(data_path, delimiter=',')
# data90 = cosinto0(data90, 100, 100, 78, 78)
# data_path = "./data/datasets/Dataset-double-model2.2/foward90"
# np.savetxt(data_path, data90, delimiter=',')
# #
# data_path = "./data/datasets/Dataset-double-model2.2/foward90"
# data90 = np.loadtxt(data_path, delimiter=',')
# x, y = np.linspace(0, 640, 256), np.linspace(0, 640, 256)
# plot_mat(data90, title="pure data")
# plt.show()

mask_path = r"./data/datasets/denoise-mask-128-model2/foward90.npy"
mask = np.load(mask_path)
print(mask.shape)
mask = np.pad(mask, pad_width=16, constant_values=1)
print(mask.shape)
mask_path = r"./data/datasets/denoise-mask-128-model2.2/foward90.npy"
np.save(mask_path, mask)

mask_path = r"./data/datasets/denoise-mask-128-model2/foward60.npy"
mask = np.load(mask_path)
print(mask.shape)
mask = np.pad(mask, pad_width=16, constant_values=1)
print(mask.shape)
mask_path = r"./data/datasets/denoise-mask-128-model2.2/foward60.npy"
np.save(mask_path, mask)

mask_path = r"./data/datasets/denoise-mask-128-model2/foward45.npy"
mask = np.load(mask_path)
print(mask.shape)
mask = np.pad(mask, pad_width=16, constant_values=1)
print(mask.shape)
mask_path = r"./data/datasets/denoise-mask-128-model2.2/foward45.npy"
np.save(mask_path, mask)