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
    z = z.reshape(128 * 128)
    length_x = len(x)
    length_z = len(z)
    times = length_z // length_x
    ellipses = [z[0:length_x]]
    ax.set_xlabel("x(m)")
    ax.set_ylabel("y(m)")
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
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['figure.titlesize'] = 12
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['legend.title_fontsize'] = 12

    x, y = np.linspace(0, 640, 128), np.linspace(0, 640, 128)
    number = 11
    ori_data = read_txt(f'data/datasets/Dataset-double-model1/foward{number}')
    mat_min, mat_max = get_matrix_maximum_value(ori_data)
    ori_data = (ori_data - mat_min) / (mat_max - mat_min)

    pure_data = (read_txt(f'data/datasets/Dataset-pure-model1/foward{number}') - mat_min) / (mat_max - mat_min)
    masked_data = np.load(f'log/denoise_128_model1/gt_masked/foward{number}.npy')
    denoise_data = np.load(f'log/denoise_128_model1/inpainted/foward{number}.npy')
    result_data = np.load(f'log/denoise_128_model1/final/foward{number}.npy')

    print("mse:", mse(pure_data, result_data))

    plot_mat(denor(pure_data, mat_min, mat_max), title="pure data")
    plot_mat(denor(ori_data, mat_min, mat_max), title="observe data")
    plot_mat(denor(denoise_data, mat_min, mat_max), title="repaint data")
    plot_mat(denor(result_data, mat_min, mat_max), title='denoise data')
    plt.show()
    cbdnetresult = read_txt(f'log/denoise_128_model1/CBDresult')
    mfresult = read_txt(f'log/denoise_128_model1/mfresult.txt')

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 3, 1)
    im1 = plt.imshow(denor(pure_data, mat_min, mat_max), vmax=mat_max, vmin=mat_min, cmap='jet')
    plt.gca().invert_yaxis()
    plt.axis('off')

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = plt.imshow(denor(ori_data, mat_min, mat_max), vmax=mat_max, vmin=mat_min, cmap='jet')
    plt.gca().invert_yaxis()
    plt.axis('off')

    ax3 = fig.add_subplot(2, 3, 3)
    im3 = plt.imshow(denor(denoise_data, mat_min, mat_max), vmax=mat_max, vmin=mat_min, cmap='jet')
    plt.gca().invert_yaxis()
    plt.axis('off')

    ax4 = fig.add_subplot(2, 3, 4)
    im4 = plt.imshow(denor(result_data, mat_min, mat_max), vmax=mat_max, vmin=mat_min, cmap='jet')
    plt.gca().invert_yaxis()
    plt.axis('off')

    ax5 = fig.add_subplot(2, 3, 5)
    im5 = plt.imshow(cbdnetresult, vmax=mat_max, vmin=mat_min, cmap='jet')
    plt.gca().invert_yaxis()
    plt.axis('off')

    ax6 = fig.add_subplot(2, 3, 6)
    im6 = plt.imshow(mfresult, vmax=mat_max, vmin=mat_min, cmap='jet')
    plt.gca().invert_yaxis()
    # plt.xlabel('X(m)')
    # plt.ylabel('Y(m)')
    plt.axis('off')

    # 指定colorbar的位置和大小
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    fig.colorbar(im2, cax=cax)
    plt.show()

    mat_min2 = -40
    mat_max2 = 80

    fig2 = plt.figure()

    ax7 = fig2.add_subplot(1, 3, 1)
    im7 = plt.imshow(denor(pure_data, mat_min, mat_max) - denor(result_data, mat_min, mat_max), vmax=mat_max2,
                     vmin=mat_min2, cmap='jet')
    plt.gca().invert_yaxis()
    plt.axis('off')
    print(mse(denor(pure_data, mat_min, mat_max), denor(result_data, mat_min, mat_max)))
    ax8 = fig2.add_subplot(1, 3, 2)
    im8 = plt.imshow(denor(pure_data, mat_min, mat_max) - cbdnetresult, vmax=mat_max2, vmin=mat_min2, cmap='jet')
    plt.gca().invert_yaxis()
    plt.axis('off')

    ax9 = fig2.add_subplot(1, 3, 3)
    im9 = plt.imshow(denor(pure_data, mat_min, mat_max) - mfresult, vmax=mat_max2, vmin=mat_min2, cmap='jet')
    plt.gca().invert_yaxis()
    plt.axis('off')

    cax2 = fig2.add_axes([0.92, 0.1, 0.02, 0.8])
    fig2.colorbar(im7, cax=cax2)
    plt.show()

    # data1 = denor(pure_data, mat_min, mat_max) - denor(result_data, mat_min, mat_max)
    # data1 = np.array(data1)
    # data1 = data1.reshape(-1, 1)
    #
    # data2 = denor(pure_data, mat_min, mat_max) - cbdnetresult
    # data2 = np.array(data2)
    # data2 = data2.reshape(-1, 1)
    #
    # data3 = denor(pure_data, mat_min, mat_max) - mfresult
    # data3 = np.array(data3)
    # data3 = data3.reshape(-1, 1)

    # data = np.hstack((data1, data2, data3))
    # fig = plt.figure()
    # view = plt.boxplot(data, patch_artist=True, meanprops={'ls': '--'})
    # plt.show()
    # q11, q13 = np.percentile(data1, [25, 75])
    # iqr1 = q13 - q11
    # print("MSJDN的四分位数：", iqr1)
    #
    # q21, q23 = np.percentile(data2, [25, 75])
    # iqr2 = q23 - q21
    # print("CBD的四分位数：", iqr2)
    #
    # q31, q33 = np.percentile(data3, [25, 75])
    # iqr3 = q33 - q31
    # print("MF的四分位数：", iqr3)

    # noiseori = np.sqrt(np.mean(np.square(ori_data - denor(pure_data, mat_min, mat_max))))
    #
    # noisemsjdn = np.sqrt(
    #     np.mean(np.square((denor(result_data, mat_min, mat_max) - denor(pure_data, mat_min, mat_max)))))
    #
    # noisecbd = np.sqrt(np.mean(np.square((denor(pure_data, mat_min, mat_max) - cbdnetresult))))
    #
    # noisemf = np.sqrt(np.mean(np.square(denor(pure_data, mat_min, mat_max) - mfresult)))
    #
    # Nrr1 = (noiseori - noisemsjdn) / noiseori * 100
    # Nrr2 = (noiseori - noisecbd) / noiseori * 100
    # Nrr3 = (noiseori - noisemf) / noiseori * 100
    #
    # print('MSJDN噪声消除率：' + str(Nrr1))
    # print('CBD噪声消除率：' + str(Nrr2))
    # print('MF噪声消除率：' + str(Nrr3))
