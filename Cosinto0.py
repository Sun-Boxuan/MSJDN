import numpy as np
import math

def cosinto0(g, n, m, ndiff, mdiff):
#(data, 128, 128, 64, 64)
    # prepare
    # g = np.zeros((npts, npts), dtype=float)
    # for i in range(mdiff+1, mdiff+m):
    #     for j in range(ndiff+1, ndiff+n):
    #         g[i - 1][j - 1] = dataset[i - mdiff-1][j - ndiff - 1]

    # Initialize variables
    g1 = np.zeros((n, ndiff))
    g2 = np.zeros((n, ndiff))
    g3 = np.zeros((mdiff, 2*ndiff+n))
    g4 = np.zeros((mdiff, 2*ndiff+n))

    # domain 1
    for i in range(n):
        for j in range(ndiff):
            g1[i, j] = g[i, 0] * np.cos((np.pi/2) * (j/ndiff))

    # domain 2
    for i in range(n):
        for j in range(ndiff):
            g2[i, j] = g[i, m-1] * np.cos((np.pi/2) * (j/ndiff))

    # domain 3
    gp1 = np.concatenate((np.fliplr(g1), g, g2), axis=1)
    for i in range(mdiff):
        for j in range(2*ndiff+n):
            g3[i, j] = gp1[0, j] * np.cos((np.pi/2) * (i/mdiff))

    for i in range(mdiff):
        for j in range(2*ndiff+n):
            g4[i, j] = gp1[n-1, j] * np.cos((np.pi/2) * (i/mdiff))

    gf = np.concatenate((np.flipud(g3), gp1, g4), axis=0)


    # (data, 144, 128, 128, 8, 8)
    # # Sides
    # for J in range(mdiff + 1, mdiff + m + 1):
    #     for I in range(1, ndiff + 1):
    #         gf[I - 1, J - 1] = gf[I - 1, mdiff] * (np.sin(-math.pi / 2 + (I - 1) * math.pi / ndiff) * 0.5)  # 对应于图像中标号为1
    #         gf[npts - I, J - 1] = gf[npts - I, mdiff + m] * (
    #                 np.sin(-math.pi / 2 + (I - 1) * math.pi / ndiff) * 0.5)  # 对应于图像中标号为2
    #
    # for I in range(ndiff + 1, ndiff + n + 1):
    #     for J in range(1, mdiff + 1):
    #         gf[I - 1, J - 1] = gf[ndiff, J - 1] * (np.sin(-math.pi / 2 + (J - 1) * math.pi / mdiff) * 0.5)  # 对应于图像中标号为3
    #         gf[I - 1, npts - J] = gf[ndiff + n - 1, npts - J] * (
    #                 np.sin(-math.pi / 2 + (J - 1) * math.pi / mdiff) * 0.5)  # 对应于图像中标号为4
    #
    # for J in range(1, mdiff + 1):
    #     for I in range(1, ndiff + 1):
    #         gf[I - 1, J - 1] = gf[I - 1, J - 1] * (np.sin(-math.pi / 2 + (I - 1) * math.pi / ndiff) * 0.5)  # 对应于图像中标号为5
    #         gf[npts - I, J - 1] = gf[npts - I, J - 1] * (
    #                 np.sin(-math.pi / 2 + (I - 1) * math.pi / ndiff) * 0.5)  # 对应于图像中标号为6
    #
    # for J in range(mdiff + m, npts + 1):
    #     for I in range(1, ndiff + 1):
    #         gf[I - 1, J - 1] = gf[I - 1, J - 1] * (np.sin(-math.pi / 2 + (I - 1) * math.pi / ndiff) * 0.5)  # 对应于图像中标号为7
    #         gf[npts - I, J - 1] = gf[npts - I, J - 1] * (
    #                 np.sin(-math.pi / 2 + (I - 1) * math.pi / ndiff) * 0.5)  # 对应于图像中标号为8
    #
    # # Corners
    # for J in range(mdiff + m, npts + 1):
    #     for I in range(ndiff + n, npts + 1):
    #         gf[I - 1, J - 1] = gf[I - 1, J - 1] * np.cos((I - ndiff - n) * math.pi / (2 * ndiff)) * np.cos(
    #             (J - ndiff - m) * math.pi / (2 * mdiff))
    #
    # for J in range(1, mdiff + 1):
    #     for I in range(1, ndiff + 1):
    #         gf[I - 1, J - 1] = gf[I - 1, J - 1] * np.cos((I - ndiff) * math.pi / (2 * ndiff)) * np.cos(
    #             (J - ndiff) * math.pi / (2 * mdiff))
    #
    # for J in range(1, mdiff + 1):
    #     for I in range(ndiff + n, npts + 1):
    #         gf[I - 1, J - 1] = gf[I - 1, J - 1] * np.cos((I - ndiff - n) * math.pi / (2 * ndiff)) * np.cos(
    #             (J - ndiff) * math.pi / (2 * mdiff))
    #
    # for J in range(mdiff + m, npts + 1):
    #     for I in range(1, ndiff + 1):
    #         gf[I - 1, J - 1] = gf[I - 1, J - 1] * np.cos((I - ndiff) * math.pi / (2 * ndiff)) * np.cos(
    #             (J - ndiff - m) * math.pi / (2 * mdiff))
    return gf