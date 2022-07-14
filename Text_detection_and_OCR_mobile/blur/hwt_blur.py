import numpy as np
import pywt


def hwt_blur_detect(img, threshold):
    try:
        M, N = img.shape
    except:
        raise ValueError("Img must have shape of len 2, got shape len: %s" % len(
            img.shape))

    # Crop input image to be 3 divisible by 2
    img = img[0:int(M / 16) * 16, 0:int(N / 16) * 16]

    # Step 1, compute Haar wavelet of input image
    LL1, (LH1, HL1, HH1) = pywt.dwt2(img, 'haar')
    # Another application of 2D haar to LL1
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, 'haar')
    # Another application of 2D haar to LL2
    LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, 'haar')

    # Construct the edge map in each scale Step 2
    E1 = np.sqrt(np.power(LH1, 2) + np.power(HL1, 2) + np.power(HH1, 2))
    E2 = np.sqrt(np.power(LH2, 2) + np.power(HL2, 2) + np.power(HH2, 2))
    E3 = np.sqrt(np.power(LH3, 2) + np.power(LH3, 2) + np.power(HH3, 2))

    M1, N1 = E1.shape

    # Sliding window size level 1
    sizeM1 = 8
    sizeN1 = 8

    # Sliding windows size level 2
    sizeM2 = int(sizeM1 / 2)
    sizeN2 = int(sizeN1 / 2)

    # Sliding windows size level 3
    sizeM3 = int(sizeM2 / 2)
    sizeN3 = int(sizeN2 / 2)

    # Number of edge maps, related to sliding windows size
    n_iter = int((M1 / sizeM1) * (N1 / sizeN1))

    Emax1 = np.zeros(n_iter)
    Emax2 = np.zeros(n_iter)
    Emax3 = np.zeros(n_iter)

    # Sliding windows index of level 1
    x1 = 0
    y1 = 0
    # Sliding windows index of level 2
    x2 = 0
    y2 = 0
    # Sliding windows index of level 3
    x3 = 0
    y3 = 0

    # Sliding windows limit on horizontal dimension
    Y_limit = N1 - sizeN1
    for i in range(n_iter):
        # Get the maximum value of slicing windows over edge maps
        # in each level
        Emax1[i] = np.max(E1[x1:x1 + sizeM1, y1:y1 + sizeN1])
        Emax2[i] = np.max(E2[x2:x2 + sizeM2, y2:y2 + sizeN2])
        Emax3[i] = np.max(E3[x3:x3 + sizeM3, y3:y3 + sizeN3])
        # if sliding windows ends horizontal direction
        # move along vertical direction and resets horizontal
        # direction
        if y1 == Y_limit:
            x1 = x1 + sizeM1
            y1 = 0
            x2 = x2 + sizeM2
            y2 = 0
            x3 = x3 + sizeM3
            y3 = 0
        # windows moves along horizontal dimension
        else:
            y1 = y1 + sizeN1
            y2 = y2 + sizeN2
            y3 = y3 + sizeN3

    # Step 3
    # print(Emax1)
    # print(Emax2)
    # print(Emax3)
    EdgePoint1 = Emax1 > threshold
    EdgePoint2 = Emax2 > threshold
    EdgePoint3 = Emax3 > threshold

    # Rule 1 Edge Pojnts
    EdgePoint = EdgePoint1 + EdgePoint2 + EdgePoint3

    n_edges = EdgePoint.shape[0]

    # Rule 2 Dirak-Structure or Astep-Structure
    da_structure1 = Emax1[EdgePoint] > Emax2[EdgePoint]
    print(round(np.sum(da_structure1) / len(da_structure1) * 100, 1), len(da_structure1))
    print(np.array(da_structure1).astype('int'))
    da_structure2 = Emax2[EdgePoint] > Emax3[EdgePoint]
    print(round(np.sum(da_structure2) / len(da_structure2) * 100, 1), len(da_structure2))
    print(np.array(da_structure2).astype('int'))
    DAstructure = da_structure1 * da_structure2
    print(round(np.sum(DAstructure) / len(DAstructure) * 100, 1), len(DAstructure))
    print("sum(DAstructure):", np.sum(DAstructure))
    print("sum(EdgePoint):", np.sum(EdgePoint))
    print("End")

    # Rule 3 Roof-Structure or Gstep-Structure

    RGstructure = np.zeros(n_edges)
    for i in range(n_edges):
        if EdgePoint[i] == 1:
            if Emax1[i] < Emax2[i] and Emax2[i] < Emax3[i]:
                RGstructure[i] = 1

    # Rule 4 Roof-Structure

    RSstructure = np.zeros(n_edges)
    for i in range(n_edges):
        if EdgePoint[i] == 1:
            if Emax2[i] > Emax1[i] and Emax2[i] > Emax3[i]:
                RSstructure[i] = 1

    # Rule 5 Edge more likely to be in a blurred image
    BlurC = np.zeros(n_edges)

    for i in range(n_edges):
        if RGstructure[i] == 1 or RSstructure[i] == 1:
            if Emax1[i] < threshold:
                BlurC[i] = 1
                # Step 6
    Per = np.sum(DAstructure) / np.sum(EdgePoint)
    # Step 7
    if np.sum(RSstructure) == 0:
        BlurExtent = 100
    else:
        BlurExtent = np.sum(BlurC) / np.sum(RSstructure)
    return Per, BlurExtent
