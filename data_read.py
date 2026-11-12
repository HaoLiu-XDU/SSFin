import scipy.io as sio
import torch
import random
import numpy as np
from sklearn.decomposition import PCA
import rasterio
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def min_max(x):
    min = np.min(x)
    max = np.max(x)
    return (x - min) / (max - min)

def pca_whitening(image, number_of_pc):

    shape = image.shape

    image = np.reshape(image, [shape[0]*shape[1], shape[2]])
    number_of_rows = shape[0]
    number_of_columns = shape[1]
    pca = PCA(n_components = number_of_pc)
    image = pca.fit_transform(image)
    pc_images = np.zeros(shape=(number_of_rows, number_of_columns, number_of_pc),dtype=np.float32)
    for i in range(number_of_pc):
        pc_images[:, :, i] = np.reshape(image[:, i], (number_of_rows, number_of_columns))
    return pc_images

def load_data(dataset):
    if dataset == 'Augsburg':
        hsi_path = "/media/datapart/haoliu/Augsburg/augsburg_hsi.mat"
        sar_path = "/media/datapart/haoliu/Augsburg/augsburg_sar.mat"
        gt_path = "/media/datapart/haoliu/Augsburg/augsburg_gt.mat"
        index_path = "/media/datapart/haoliu/Augsburg/augsburg_index.mat"
        augsburg_keys = ['augsburg_hsi', 'augsburg_sar', 'augsburg_gt', 'augsburg_train', 'augsburg_test', 'augsburg_all']
        hsi = sio.loadmat(hsi_path)[augsburg_keys[0]]
        X = sio.loadmat(sar_path)[augsburg_keys[1]]
        gt = sio.loadmat(gt_path)[augsburg_keys[2]]
        image=hsi
        image_X=X
        label=gt
        train_index = None
        test_index = None
    elif dataset == 'HK2B':
        #cross-region dataset, make sure they all have 10 classes
        with rasterio.open("/media/datapart/haoliu/HK/HK_MSI.tif") as msi:
            msi_image = msi.read()  # Reads all bands into a numpy array (shape: [bands, height, width])
        with rasterio.open("/media/datapart/haoliu/HK/HK_SAR.tif") as sar:
            sar_image = sar.read()
        with rasterio.open("/media/datapart/haoliu/HK/HK_GT.tif") as gt:
            gt_image = gt.read()
        data = np.transpose(msi_image, (1, 2, 0))
        dataX = np.transpose(sar_image, (1, 2, 0))
        label = gt_image.squeeze(0)
        value_changes = {1: 0,2: 1,3: 0, 4: 2, 5: 3, 6: 4, 8: 5,10:0, 11: 6, 12: 7, 13: 8, 14: 9, 17: 10,
                         }  # Change value 8 to 7, and value 10 to 8

        # Apply changes
        for old_value, new_value in value_changes.items():
            label[label == old_value] = new_value
        unique_values, counts = np.unique(label, return_counts=True)

        for value, count in zip(unique_values, counts):
            print(f"Value {value}: {count} occurrences")
        height_scale = data.shape[0] / dataX.shape[0]  # Scale factor for height
        width_scale = data.shape[1] / dataX.shape[1]  # Scale factor for width
        current_bands = np.linspace(0, 1, 10)  # Assume normalized range [0, 1] for simplicity
        target_bands = np.linspace(0, 1, 12)  # 16 evenly spaced target bands
        merged_image = data
        # Interpolate along the spectral dimension
        interpolated_image = np.empty((merged_image.shape[0], merged_image.shape[1], len(target_bands)))
        for i in range(merged_image.shape[0]):  # Iterate over height
            for j in range(merged_image.shape[1]):  # Iterate over width
                spectral_values = merged_image[i, j, :]
                f = interp1d(current_bands, spectral_values, kind='linear', fill_value="extrapolate")
                interpolated_image[i, j, :] = f(target_bands)
        image = interpolated_image
        # Resize SAR image using zoom
        image_X = zoom(dataX, (height_scale, width_scale, 1), order=1)
        train_index = None
        test_index = None
    elif dataset == 'B2C':
        # cross-region dataset, make sure they all have 10 classes
        with rasterio.open("/media/datapart/haoliu/HK/Berlin_MSI.tif") as msi:
            msi_image = msi.read()  # Reads all bands into a numpy array (shape: [bands, height, width])
        with rasterio.open("/media/datapart/haoliu/HK/Berlin_SAR.tif") as sar:
            sar_image = sar.read()
        with rasterio.open("/media/datapart/haoliu/HK/Berlin_GT.tif") as gt:
            gt_image = gt.read()
        data = np.transpose(msi_image, (1, 2, 0))
        dataX = np.transpose(sar_image, (1, 2, 0))
        label = gt_image.squeeze(0)
        value_changes = {2: 1, 4: 2, 5: 3, 6: 4, 8: 5,9: 0, 11: 6, 12: 7, 13: 8, 14: 9,16: 0, 17: 10}
        for old_value, new_value in value_changes.items():
            label[label == old_value] = new_value
        unique_values, counts = np.unique(label, return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Value {value}: {count} occurrences")
        height_scale = data.shape[0] / dataX.shape[0]  # Scale factor for height
        width_scale = data.shape[1] / dataX.shape[1]  # Scale factor for width
        current_bands = np.linspace(0, 1, 10)  # Assume normalized range [0, 1] for simplicity
        target_bands = np.linspace(0, 1, 12)  # 16 evenly spaced target bands
        merged_image = data
        # Interpolate along the spectral dimension
        interpolated_image = np.empty((merged_image.shape[0], merged_image.shape[1], len(target_bands)))
        for i in range(merged_image.shape[0]):  # Iterate over height
            for j in range(merged_image.shape[1]):  # Iterate over width
                spectral_values = merged_image[i, j, :]
                f = interp1d(current_bands, spectral_values, kind='linear', fill_value="extrapolate")
                interpolated_image[i, j, :] = f(target_bands)
        image = interpolated_image
        # Resize SAR image using zoom
        image_X = zoom(dataX, (height_scale, width_scale, 1), order=1)
        train_index = None
        test_index = None
    elif dataset == 'HK':
        # Load TIFF image
        with rasterio.open("/media/datapart/haoliu/HK/HK_MSI.tif") as msi:
            msi_image = msi.read()  # Reads all bands into a numpy array (shape: [bands, height, width])
        with rasterio.open("/media/datapart/haoliu/HK/HK_SAR.tif") as sar:
            sar_image = sar.read()
        with rasterio.open("/media/datapart/haoliu/HK/HK_GT.tif") as gt:
            gt_image = gt.read()
        data = np.transpose(msi_image, (1, 2, 0))
        dataX = np.transpose(sar_image, (1, 2, 0))
        label = gt_image.squeeze(0)
        value_changes = {8: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12, 17: 13}  # Change value 8 to 7, and value 10 to 8

        # Apply changes
        for old_value, new_value in value_changes.items():
            label[label == old_value] = new_value
        unique_values, counts = np.unique(label, return_counts=True)

        for value, count in zip(unique_values, counts):
            print(f"Value {value}: {count} occurrences")
        height_scale = data.shape[0] / dataX.shape[0]  # Scale factor for height
        width_scale = data.shape[1] / dataX.shape[1]  # Scale factor for width
        current_bands = np.linspace(0, 1, 10)  # Assume normalized range [0, 1] for simplicity
        target_bands = np.linspace(0, 1, 12)  # 16 evenly spaced target bands
        merged_image = data
        # Interpolate along the spectral dimension
        interpolated_image = np.empty((merged_image.shape[0], merged_image.shape[1], len(target_bands)))
        for i in range(merged_image.shape[0]):  # Iterate over height
            for j in range(merged_image.shape[1]):  # Iterate over width
                spectral_values = merged_image[i, j, :]
                f = interp1d(current_bands, spectral_values, kind='linear', fill_value="extrapolate")
                interpolated_image[i, j, :] = f(target_bands)
        image = interpolated_image
        # Resize SAR image using zoom
        image_X = zoom(dataX, (height_scale, width_scale, 1), order=1)
        train_index = None
        test_index = None
        # trntst_index = np.concatenate((train_index, test_index), axis=0)
    elif dataset == 'Houston':
        hsi_path = "/media/datapart/haoliu/Houston2013/HoustonU_hsi.mat"
        lidar_path = "/media/datapart/haoliu/Houston2013/HoustonU_lidar.mat"
        gt_path = "/media/datapart/haoliu/Houston2013/HoustonU_gt.mat"
        hsi = sio.loadmat(hsi_path)['hsi']
        X = sio.loadmat(lidar_path)['lidar'][:,:,None]
        gt = sio.loadmat(gt_path)['img_gt']
        image = hsi
        image_X = X
        label = gt
        train_index = None
        test_index = None
    elif dataset == 'Yellow':

        hsi = sio.loadmat('/media/datapart/haoliu/HHK/data_hsi.mat')['data'] #(960, 1170, 166)
        X = sio.loadmat('/media/datapart/haoliu/HHK/data_sar.mat')['data'] #(960, 1170, 3)
        test_index = sio.loadmat("/media/datapart/haoliu/HHK/mask_test.mat")['mask_test'] #(960, 1170)
        gt = test_index
        image = hsi
        image_X = X
        label = gt
        train_index = None
        test_index = None
    else:
        raise Exception('dataset does not find')
    image = image.astype(np.float32)
    return image, label, image_X, train_index, test_index

def readdata(args, num=0):

    or_image, or_label, image_X, train_index, test_index = load_data(args.dataset)
    windowsize = args.windowsize
    train_num = args.train_num
    halfsize = int((windowsize-1)/2)
    number_class = np.max(or_label)
    image = np.pad(or_image, ((halfsize, halfsize), (halfsize, halfsize), (0, 0)), 'edge')
    if image_X.shape[0] == 1202 or image_X.shape[0]==166:
        imageX = np.pad(image_X, ((halfsize, halfsize), (halfsize, halfsize)), 'edge')
    else:
        imageX = np.pad(image_X, ((halfsize, halfsize), (halfsize, halfsize), (0, 0)), 'edge')
    label = np.pad(or_label, ((halfsize, halfsize), (halfsize, halfsize)), 'constant',constant_values=0)
    if args.type == 'PCA':
        image1 = pca_whitening(image, number_of_pc = 30)
    if args.type == 'none':
        image1 = image

    image = image1
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    imageX = (imageX - np.min(imageX)) / (np.max(imageX) - np.min(imageX))
    imageX = imageX.reshape(imageX.shape[0],imageX.shape[1],-1)

    n = np.zeros(number_class,dtype=np.int64)
    print(n)

    for i in range(number_class):
        temprow, tempcol = np.where(or_label == i + 1)
        n[i] = len(temprow)
    total = np.sum(n)

    nTrain_perClass = np.ones(number_class,dtype=np.int32) * train_num
    ind = np.where(n <= train_num)

    if train_num <=15:
        nTrain_perClass[ind[0]] = train_num//2
    else:
        nTrain_perClass[ind[0]] = 15

    # adjust for parameter tuning
    num_vali = 0
    nValidation_perClass = (n/total)*num_vali
    nvalid_perClass = nValidation_perClass.astype(np.int32)


    index = []
    s = 0
    flag = 0
    f = 0
    band_X = np.size(imageX, 2)
    bands = np.size(image,2)
    validation_image = np.zeros([np.sum(nvalid_perClass), windowsize, windowsize, bands], dtype=np.float32)
    validation_imageX = np.zeros([np.sum(nvalid_perClass), windowsize, windowsize, band_X], dtype=np.float32)
    validation_label = np.zeros(np.sum(nvalid_perClass), dtype=np.int32)

    train_image = np.zeros([np.sum(nTrain_perClass), windowsize, windowsize, bands], dtype=np.float32)
    train_imageX = np.zeros([np.sum(nTrain_perClass), windowsize, windowsize, band_X], dtype=np.float32)
    train_label = np.zeros(np.sum(nTrain_perClass), dtype=np.int32)

    for i in range(number_class):

        temprow, tempcol = np.where(label == i + 1)
        s = s + len(temprow)
        matrix = np.zeros([len(temprow),2], dtype=np.int64)
        matrix[:,0] = temprow
        matrix[:,1] = tempcol
        np.random.seed(num)
        np.random.shuffle(matrix)
        temprow = matrix[:,0]
        tempcol = matrix[:,1]
        index.append(matrix)
        for j in range(nTrain_perClass[i]):
            train_image[flag + j, :, :, :] = image[(temprow[j] - halfsize):(temprow[j] + halfsize + 1),
                                            (tempcol[j] - halfsize):(tempcol[j] + halfsize + 1)]
            train_imageX[flag + j, :, :, :] = imageX[(temprow[j] - halfsize):(temprow[j] + halfsize + 1),
                                             (tempcol[j] - halfsize):(tempcol[j] + halfsize + 1)]
            train_label[flag + j] = i
        flag = flag + nTrain_perClass[i]
        # validation set, change num_vali for different parameter test
        for j in range(nvalid_perClass[i]):
            r, c = matrix[nTrain_perClass[i] + j]
            validation_image[f + j, :, :, :] = image[(r - halfsize):(r + halfsize + 1),
                                            (c - halfsize):(c + halfsize + 1)]
            validation_imageX[f + j, :, :, :] = imageX[(r - halfsize):(r + halfsize + 1),
                                                                    (c - halfsize):(c+ halfsize + 1)]
            validation_label[f + j] = i
        f = f + nvalid_perClass[i]

    return train_image, train_label,validation_image, validation_label, nTrain_perClass, nvalid_perClass, index, image, or_label, s, train_imageX, validation_imageX, imageX, test_index


