import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


def get_matrix_maximum_value(mat):
    max_index = np.unravel_index(np.argmax(mat, axis=None), mat.shape)
    min_index = np.unravel_index(np.argmin(mat, axis=None), mat.shape)
    return mat[min_index], mat[max_index]


def normalization(mat):
    m_min, m_max = get_matrix_maximum_value(mat)
    return (mat - m_min) / (m_max - m_min)


def read_txt(file_path, delimiter=','):  # delimiter是数据分隔符
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = [[float(i) for i in line.strip().split(delimiter)] for line in lines]
    return np.array(data)


def denor(mat, m_min, m_max):
    return mat * (m_max - m_min) + m_min


def mse(mat_a, mat_b):
    return np.sum((mat_a - mat_b) ** 2) / np.size(mat_a)


def grid_plot(ax, z, separate_number=12):
    global x, y
    x_mesh, y_mesh = np.meshgrid(x, y)
    z = z.reshape(128*128)
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

if __name__ == '__main__':
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 1239.154
    mpl.rcParams['figure.titlesize'] = 12
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['legend.title_fontsize'] = 12

    x, y = np.linspace(0, 640, 128), np.linspace(0, 640, 128)
    number = 1
    ori_data = read_txt(f'data/datasets/Dataset-double-real/foward{number}')
    mat_min, mat_max = get_matrix_maximum_value(ori_data)
    ori_data = (ori_data - mat_min) / (mat_max - mat_min)

    pure_data = (read_txt(f'data/datasets/Dataset-pure-real/foward{number}') - mat_min) / (mat_max - mat_min)
    masked_data = np.load(f'log/denoise_128_real/gt_masked/foward{number}.npy')
    denoise_data = np.load(f'log/denoise_128_real/inpainted/foward{number}.npy')
    result_data = np.load(f'log/denoise_128_real/final/foward{number}.npy')

    print("mse:", mse(pure_data, result_data))
    ori_data2 = np.sqrt(np.mean(np.square(ori_data - pure_data)))
    denoise2 = np.sqrt(np.mean(np.square(result_data - pure_data)))
    Nrr = (ori_data2 - denoise2) / ori_data2 * 100
    print('噪声消除率：' + str(Nrr))

    # plot_mat(denor(pure_data, mat_min, mat_max), title="pure data")
    # plot_mat(denor(ori_data, mat_min, mat_max), title="observe data")
    # plot_mat(denor(denoise_data, mat_min, mat_max), title="repaint data")
    # plot_mat(denor(result_data, mat_min, mat_max), title='denoise data')
    # plt.show()