
# coding: utf-8

# # Classification of paddy rice over Thai Binh in Vietnam
# 
# * Exploring time series data for classification

import argparse
import os
import sys
import itertools
from collections import namedtuple

import numpy as np
import pandas as pd
from osgeo import gdal, gdal_array, osr

from sklearn.ensemble import RandomForestClassifier
# from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import MeanShift
from sklearn.metrics import confusion_matrix

# import skimage as ski
from scipy import ndimage

import matplotlib.pyplot as plt

gdal.AllRegister()

# A class for image classification
class ImageClassifier:
    def __init__(self):
        # flat_ind: row-major flat indices to pixels given image
        # dimensions, dims.
        self.imgdata = dict(\
                            X=None, y=None, prob=None, \
                            dims=None, \
                            file_ind=None, \
                            valid_flag=None, Xsrc=None)
        self.traindata = dict(\
                              X=None, y=None, dims=None, \
                              label_meta=None, flat_ind=None, \
                              ysrc=None, Xsrc=None)

        self._img_nodata = -9999
        self._cls_nodata = 0
        self._clsprob_nodata = -9999

    def readImageData(self, img_files, bands=None):
        """Read image data from all the bands of one or mulitiple raster
        files as input to the classification.
        
        Parameters: **img_files**, *str or a sequence of str*
                        Raster file/s from which all bands are read
                        and used in the classification.
                    **bands**, *sequence or list of sequence*
                        Band indices to be read in each raster file
                        given in *img_files*. 
        """
        # Prep the input image data for classification. 
        # All the bands of all the images are used in the classification. 
        # Each band is a feature and each pixel is a sample.

        if type(img_files) is str:
            img_files = [img_files]
        if bands is None:
            bands = [None for i in img_files]

        if len(img_files) != len(bands):
            raise RuntimeError("Number of input raster files does not match the number of band index sequences")

        self.imgdata['Xsrc'] = {'files':img_files, 'bands':bands}

        read_raster_out = [self.readRaster(fname, band=bd, out_nodata=self._img_nodata) for fname, bd in zip(img_files, bands)]
        imgdata_list = [rro[0] for rro in read_raster_out]
        self.imgdata['Xsrc']['bands'] = [rro[1] for rro in read_raster_out]

        self.imgdata['file_ind'] = np.array([n for n, subl in enumerate(imgdata_list) for d in subl])
        imgdata_list = [d for subl in imgdata_list for d in subl]

        # check the dimension of all bands from all files, make sure they are the same
        for d in imgdata_list:
            if d.shape[0] != imgdata_list[0].shape[0] or d.shape[1] != imgdata_list[0].shape[1]:
                raise RuntimeError("Dimension of input image bands are not consistent")

        imgdata = np.stack(imgdata_list, axis=2)

        self.imgdata['dims'] = imgdata.shape

        imgdata = imgdata.reshape(imgdata.shape[0]*imgdata.shape[1], imgdata.shape[2])

        # Set a 1D array to record the row-major flat indices to pixel
        # locations.
        self.imgdata['valid_flag'] = np.all(imgdata!=self._img_nodata, axis=1)
        self.imgdata['X'] = imgdata
        

    def readTrainingData(self, train_file, fmt="ENVI ROI ASCII", train_X_file=None):
        """Read training data from a user-generated file in the designated
        file format.

        Parameters: **train_file**, *str*
                        File name of training data in the data format
                        given by *ftype*. Some formats only provides
                        pixel locations of training samples and labels
                        of these pixels. In this case, either an extra
                        parameter *train_X_file* is required to
                        provide training X data or the training X data
                        will be extracted from the input image data if
                        the training samples are from the input
                        image/s.
                    **fmt**, *str, optional*
                        Format of the training data. Default: "ENVI
                        ROI ASCII". Supported format includes: "ENVI
                        ROI ASCII".
                    **train_X_file**, *str, optional*
                        File name of training X data. 
        """
        self.traindata['Xsrc'] = train_X_file
        self.traindata['ysrc'] = train_file
        if fmt == "ENVI ROI ASCII":
            roi_meta, roi_data = self._readEnviRoiAscii(train_file)
            # Now prep the ROI data for training and testing
            if train_X_file is None:
                self.traindata['X'] = None if self.imgdata['X'] is None else self.imgdata['X'][roi_data['Flat Index'].astype(np.int64), :]
                self.traindata['y'] = roi_data['Label'].as_matrix().astype(np.int)
                self.traindata['dims'] = roi_meta['dims']
                self.traindata['flat_ind'] = roi_data['Flat Index'].astype(np.int64).as_matrix()
                self.traindata['label_meta'] = {'cls_code':roi_meta['cls_code'], \
                                                'cls_name':roi_meta['cls_name'], \
                                                'cls_npts':roi_meta['cls_npts']}
        elif fmt == "GTiff ROI":
            roi_meta, roi_data = self._readGTiffRoi(train_file)
            if train_X_file is None:
                self.traindata['X'] = None if self.imgdata['X'] is None else self.imgdata['X'][roi_data['Flat Index'].astype(np.int64), :]
                self.traindata['y'] = roi_data['Label'].as_matrix().astype(np.int)
                self.traindata['dims'] = roi_meta['dims']
                self.traindata['flat_ind'] = roi_data['Flat Index'].astype(np.int64).as_matrix()
                self.traindata['label_meta'] = {'cls_code':roi_meta['cls_code'], \
                                                'cls_name':roi_meta['cls_name'], \
                                                'cls_npts':roi_meta['cls_npts']}
            

    def runRandomForest(self, **params):
        """Train a randome forest classifier from scikit-learn with loaded
        training data and apply it to loaded image data for
        classification.

        Parameters: **\*\*params**, **
                        keyword parameters for scikit-learn random
                        forest classifier.
        """
        if self.traindata['Xsrc'] is None:
            # Train data's X is from the input image data. Now remove
            # the invalid pixels according to the input images.
            train_valid_flag = self.imgdata['valid_flag'][self.traindata['flat_ind']]
        else:
            train_valid_flag = np.ones(len(self.traindata['y']), dtype=np.bool_)


        rf = RandomForestClassifier(**params)
        print "Training Random Forest Classifier with settings:"
        print rf
        rf.fit(self.traindata['X'][train_valid_flag, :],  self.traindata['y'][train_valid_flag])

        print "Predicting labels for input images"
        self.imgdata['y'] = np.zeros(len(self.imgdata['X']), dtype=self.traindata['y'].dtype)
        self.imgdata['y'][self.imgdata['valid_flag']] = rf.predict(self.imgdata['X'][self.imgdata['valid_flag'], :])
        self.imgdata['prob'] = self._clsprob_nodata+np.zeros((len(self.imgdata['X']),
                                                              len(self.traindata['label_meta']['cls_code'])),
                                                             dtype=np.float32)
        self.imgdata['prob'][self.imgdata['valid_flag'], :] = rf.predict_proba(self.imgdata['X'][self.imgdata['valid_flag'], :])
        self.estimator = "Random Forest"
        return rf

    def runCrossValidation(self, estimator, test_size=0.25, n_reps=5):
        """Generate training and test datasets from given training pixels and
        do cross validation for a guess of our classification
        accuracy. All the trainig pixels are first clustered spatially
        to determine pataches of training samples. Training and test
        datasets are then drawn as patches rather individual pixels to
        avoid spatial correlation in our cross validation.

        This cross validation is NOT meant to replace the actual
        classification accuracy assessment via random sampling of
        pixels in the classification image.

        Parameters: **estimator**, *scikit-learn estimtor object with
                        implementation of "fit"*
                        The object to use to fit the data
                    **test_size**, *float* (default is 0.25)
                        0.0-1.0, represent the proportion of the
                        pixel to include in the test split.
        Returns:    **C**, *2D numpy array, shape=[n_classes, n_classes]*
                        Confusion matrix, each row is true label and
                        each column is predicted label.
        """
        if self.traindata['Xsrc'] is None:
            # Train data's X is from the input image data. Now remove
            # the invalid pixels according to the input images.
            train_valid_flag = self.imgdata['valid_flag'][self.traindata['flat_ind']]
        else:
            train_valid_flag = np.ones(len(self.traindata['y']), dtype=np.bool_)

        trainX = self.traindata['X'][train_valid_flag, :]
        trainy = self.traindata['y'][train_valid_flag]
        train_pidx = self.traindata['flat_ind'][train_valid_flag]
        px, py = np.unravel_index(train_pidx, self.traindata['dims'], order='C')

        label_image = np.zeros(self.traindata['dims'], dtype=trainy.dtype)
        label_image[px, py] = trainy

        con_mat_list = [None for n in range(n_reps)]
        ntrain_list = np.zeros(n_reps)
        ntest_list = np.zeros(n_reps)
        for n in range(n_reps):
            print "Cross validation trial #{0:d}".format(n+1)
            test_image_flag = self.trainTestSplit(label_image, test_size=test_size, no_data=0)
            test_flag = test_image_flag[px, py]
            test_idx = np.where(test_flag)[0]
            train_idx = np.where(np.logical_not(test_flag))[0]

            estimator.fit(trainX[train_idx, :], trainy[train_idx])
            testy_hat = estimator.predict(trainX[test_idx, :])
            con_mat = confusion_matrix(trainy[test_idx], testy_hat)
            con_mat_list[n] = con_mat # / float(np.sum(con_mat))
            ntrain_list[n] = len(train_idx)
            ntest_list[n] = len(test_idx)
        con_mat = np.stack(con_mat_list, axis=2)
        return np.mean(con_mat, axis=2), int(np.mean(ntrain_list)), int(np.mean(ntest_list))
        

    def trainTestSplit(self, label_image, no_data=0, test_size=0.25):
        """Given pixel locations and labels, determine patches and split the
        patches of samples into training and test sets.

        Parameters: **px, py, plabel**, *numpy 1D array*
                        The *px and py* are X (column/sample) and Y
                        (row/line) of pixels, *plabel* is the label of
                        the pixels. All the three have the same
                        length.
                    **test_size**, *float* (default is 0.25)
                        0.0-1.0, represent the proportion of the pixel
                        to include in the test split. Because the
                        actual splitting is based on patches of pixels
                        rather than individual pixels, the actual
                        proportion is determined when the number of
                        pixels in the patches drawn as test samples is
                        equal to or just less/greater than the given
                        proportion of pixels for test samples.
        Returns:    **train_idx, test_idx**, *numpy 1D array*
                        Indices to the training and test pixels. 
        """
        # X = np.stack((px, py, plabel), axis=1)
        # # cluster the pixels using pixel locations and pixel labels
        # # 1. normalized the data
        # sc = StandardScaler()
        # X_new = sc.fit_transform(X)
        # # 2. clustering
        # ms = MeanShift(n_jobs=-1)
        # patch = ms.fit_predict(X_new)
        tmp_flag = label_image != no_data
        px, py = np.where(tmp_flag)
        plabel = label_image[tmp_flag]

        markers = np.zeros_like(label_image)
        markers[tmp_flag] = 1
        patch, npat = ndimage.label(markers, structure=np.ones((3, 3)))
        patch = patch[tmp_flag]

        # for each class (a unique value in plabel), find how many
        # patches it has.
        cls_list, cls_pcnt = np.unique(plabel, return_counts=True)
        pat_list = np.unique(patch)
        # 2D numpy array to store the number of pixels for each patch
        # (column) in each class (row). The last additional two
        # columns are number of patches and number of pixels for each
        # class.
        pcnt_stats = np.zeros((len(cls_list), len(pat_list)+2), dtype=np.int32)
        for i, cls in enumerate(cls_list):
            for j, pat in enumerate(pat_list):
                pcnt_stats[i,j] = np.sum(np.logical_and(plabel == cls, patch == pat))
        pcnt_stats[:, -2] = np.sum(pcnt_stats[:, 0:-2]>0, axis=1)
        pcnt_stats[:, -1] = np.sum(pcnt_stats[:, 0:-2], axis=1)
        # randomize sequence of patches for each class
        cls_pat_list = [np.random.permutation(np.where(pcnt_stats[k, 0:-2]>0)[0]) for k in range(len(cls_list))]
        # cumulative number of pixels in the sequence of patches for
        # each class
        cls_pcscnt_list = [np.cumsum(pcnt_stats[k, p]) for k, p in enumerate(cls_pat_list)]
        # find up to which patch in the sequence will give us the
        # proper sample size for each class.
        cls_test_pcnt_thresh = (pcnt_stats[:, -1]*test_size).astype(np.int32)
        cls_pat_idx = np.array([np.where(pcscnt>pthresh)[0][0] for pcscnt, pthresh in zip(cls_pcscnt_list, cls_test_pcnt_thresh)])
        tmp_flag = cls_pat_idx == 0
        cls_pat_idx[tmp_flag] = 1
        # get the patches for test dataset for each class
        cls_test_pat_list = [pat[0:pidx] for pidx, pat in zip(cls_pat_idx, cls_pat_list)]
        # generate the indices to test pixels for each class
        test_flag = np.zeros_like(px, dtype=np.bool_)
        for i, (cls, tpat) in enumerate(zip(cls_list, cls_test_pat_list)):
            # the patch indices in cls_test_pat_list is patch label
            # number - 1
            tmp_flag = np.logical_and(plabel==cls, np.in1d(patch, tpat+1))
            test_flag[tmp_flag] = True

        test_image_flag = np.zeros_like(label_image, dtype=np.bool_)
        test_image_flag[px[test_flag], py[test_flag]] = True
        return test_image_flag


    # def segPixels(self, image, thresh4marker=0):
    #     """Segment the image
    #     """
    #     elevation_map = ski.filters.sobel(image)
    #     markers = np.zeros_like(image)
    #     markers[image <= thresh4marker] = 1
    #     markers[image > thresh4marker] = 2
    #     seg = ski.morphology.watershed(elevation_map, markers)

    def writeClassData(self, out_cls_file, fmt="ENVI"):
        """Write classified image to a raster file given file format.

        Parameters: **out_cls_file**, *str*
                        File name of the output classification image. 
                    **fmt**, *str*
                        Format of the output raster. Default:
                        "ENVI". Supported format: all GDAL-supported
                        format with writing ability.
        """
        cls_map = np.zeros((self.imgdata['dims'][0], self.imgdata['dims'][1]), dtype=np.uint8)
        cls_map.flat[:] = self.imgdata['y'].astype(np.uint8)

        # Write the classification to an ENVI file

        raster_profile = self.getRasterProfile(self.imgdata['Xsrc']['files'][0])

        cm_used = plt.get_cmap("rainbow", len(np.unique(cls_map)))
        color_table = cm_used(range(len(np.unique(cls_map))))

        driver = gdal.GetDriverByName(fmt)
        if fmt == "ENVI":
            out_ds = driver.Create(out_cls_file, \
                                   cls_map.shape[1], cls_map.shape[0], 1, \
                                   gdal_array.NumericTypeCodeToGDALTypeCode(cls_map.dtype.type))
        else:
            cls_map = cls_map.astype(np.float32)
            ncls = self.imgdata['prob'].shape[1]
            out_ds = driver.Create(out_cls_file, \
                                   cls_map.shape[1], cls_map.shape[0], 1+ncls, \
                                   gdal_array.NumericTypeCodeToGDALTypeCode(cls_map.dtype.type))
            
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(cls_map)
        out_band.SetCategoryNames(["no_data"]+[cn for cn in self.traindata['label_meta']['cls_name']]) # set the class names
        out_band.SetDescription("Classification by {0:s}".format(self.estimator)) # set band name
        out_band.SetNoDataValue(self._cls_nodata)
        if fmt == "ENVI":
            out_ct = gdal.ColorTable()
            for i in range(len(color_table)):
                out_ct.SetColorEntry(i, tuple((color_table[i, :]*255).astype(np.int)))
            out_band.SetColorTable(out_ct)
        out_band.FlushCache()

        if fmt == "ENVI":
            bands_str = [str(b).replace('[', '"').replace(']', '"') for b in self.imgdata['Xsrc']['bands']]
            envi_meta_dict = dict(data_ignore_value=str(self._cls_nodata), \
                                  images_for_classification="{{{0:s}}}".format(", ".join(self.imgdata['Xsrc']['files'])), \
                                  bands_for_classification="{{{0:s}}}".format(", ".join(bands_str)))
            print self.imgdata['Xsrc']['files']
            print self.imgdata['Xsrc']['bands']
            print envi_meta_dict
            for kw in envi_meta_dict.keys():
                out_ds.SetMetadataItem(kw, envi_meta_dict[kw], "ENVI")
        else:
            for icls in range(ncls):
                cls_map_prob = np.zeros((self.imgdata['dims'][0], self.imgdata['dims'][1]), dtype=np.float32)
                cls_map_prob.flat[:] = self.imgdata['prob'][:, icls].astype(np.float32)
                out_band = out_ds.GetRasterBand(icls+1+1)
                out_band.WriteArray(cls_map_prob)
                out_band.SetDescription("Probability of {0:s} by {1:s}".format(self.traindata['label_meta']['cls_name'][icls], self.estimator))
                out_band.SetNoDataValue(self._clsprob_nodata)
                out_band.FlushCache()

        out_ds.SetGeoTransform(raster_profile['GeoTransform'])
        out_ds.SetProjection(raster_profile['ProjectionRef'])
        
        out_ds = None

    def writeRandomForestDiagnosis(self, rf_estimator, diagnosis_file):
        bands_str = [str(b).replace('[', '"').replace(']', '"') for b in self.imgdata['Xsrc']['bands']]
        images_for_classification="{{{0:s}}}".format(", ".join(self.imgdata['Xsrc']['files']))
        bands_for_classification="{{{0:s}}}".format(", ".join(bands_str))
        
        with open(diagnosis_file, "w") as outfobj:
            outfobj.write("# Classification Diagnosis Information for the Classifier {0:s}\n".format(self.estimator))
            outfobj.write("## Out-of-bag score = {0:.6f}\n".format(rf_estimator.oob_score_))
            outfobj.write("## Number of input features = {0:d} \n".format(rf_estimator.n_features_))
            outfobj.write("feature_id,feature_importance,raster_file,raster_band\n")
            i_feature = 0
            for fname, bdlist in zip(self.imgdata['Xsrc']['files'], self.imgdata['Xsrc']['bands']):
                for bd in bdlist:
                    outfobj.write("{0:d},{1:.6f},{2:s},{3:d}\n".format(i_feature+1, rf_estimator.feature_importances_[i_feature], fname, bd))
                    i_feature = i_feature + 1
            

    def _readEnviRoiAscii(self, envi_roi_file):
        """Read ENVI ROI ASCII file. 

        Parameters: **envi_roi_file**, *str*
                        File name of ENVI ROI ASCII data.
        Returns:    **roi_meta**, *dict*
                        Metadata of the ROI data file.
                    **roi_data**, *pandas DataFrame*
                        ROI data
        """
        # Read metadata from the beginning of the ASCII file.
        header_ind = 0
        roi_meta = dict()
        with open(envi_roi_file, 'r') as roifobj:
            roifobj.readline()
            line = roifobj.readline()
            roi_meta['ncls'] = int(line.split(':')[-1].strip())
            line = roifobj.readline()
            ns, nl = (int(s.strip()) for s in line.split(':')[-1].strip().split('x'))
            roi_meta['dims'] = (nl, ns)
            cls_npts = np.zeros(roi_meta['ncls'], dtype=np.int)
            cls_code = [None for n in cls_npts]
            cls_name = [None for n in cls_npts]
            header_ind = header_ind + 3
            for n in range(roi_meta['ncls']):
                roifobj.readline()
                line = roifobj.readline()
                cls_code[n] = n+1
                cls_name[n] = line.split(':')[-1].strip()
                roifobj.readline()
                line = roifobj.readline()
                cls_npts[n] = int(line.split(':')[-1].strip())
            header_ind = header_ind + 4*roi_meta['ncls']
            roi_meta['cls_code'] = cls_code
            roi_meta['cls_npts'] = cls_npts
            roi_meta['cls_name'] = cls_name


        roi_data = pd.read_table(envi_roi_file, header=None, delim_whitespace=True, comment=';', usecols=[0, 1, 2, 3, 4, 5, 6])
        roi_data.columns = ["ID", "X", "Y", "Map X", "Map Y", "Lat", "Lon"]
        # ENVI ROI pixel location with first being 1, here we convert to numpy convention with first being 0
        roi_data["X"] = roi_data["X"] - 1 
        roi_data["Y"] = roi_data["Y"] - 1 

        roi_data['Label'] = pd.Series(np.zeros(roi_data.shape[0]), roi_data.index)
        roi_data['Flat Index'] = pd.Series(np.zeros(roi_data.shape[0]), roi_data.index) # as X and Y, with first being 0.
        # roi_data['Flat Index'] = (roi_data['Y'] - 0) * roi_meta['dims'][1] + roi_data['X']
        roi_data['Flat Index'] = np.ravel_multi_index((roi_data["Y"], roi_data["X"]), roi_meta["dims"], order="C")

        end_ind = np.cumsum(roi_meta['cls_npts'])
        beg_ind = np.concatenate((np.array([0]), end_ind[0:-1]))

        col_ind = {cname:n for n, cname in enumerate(roi_data.columns)}
        for n, (bi, ei) in enumerate(zip(beg_ind, end_ind)):
            roi_data.iloc[bi:ei, col_ind['Label']] = np.zeros((ei-bi))+n+1

        return roi_meta, roi_data.loc[:, ["X", "Y", "Flat Index", "Label"]]

    def _readGTiffRoi(self, gtiff_file):
        """Read Geotiff ROI training file. 

        Parameters: **gtiff_file**, *str*
                        File name of Geotiff ROI training data.
        Returns:    **roi_meta**, *dict*
                        Metadata of the ROI data file.
                    **roi_data**, *pandas DataFrame*
                        ROI data
        """
        # Read metadata from the beginning of the ASCII file.
        header_ind = 0
        roi_meta = dict()

        # Open the geotiff file and read the class label image
        raster_profile = self.getRasterProfile(gtiff_file)
        raster_data, _ = self.readRaster(gtiff_file, [1],
                                         out_nodata=raster_profile["NoDataValue"][0])
        raster_data = raster_data[0]
        
        valid_ind = np.where(raster_data != raster_profile["NoDataValue"][0])
        roi_meta['dims'] = (raster_profile["RasterYSize"], raster_profile["RasterXSize"])
        roi_meta['cls_code'], roi_meta['cls_npts'] = np.unique(raster_data[valid_ind], return_counts=True)
        roi_meta['cls_name'] = [str(cc) if raster_profile["CategoryNames"][0] is None else raster_profile["CategoryNames"][0][cc] for cc in roi_meta['cls_code']]
        roi_meta['ncls'] = len(roi_meta['cls_code'])
        
        npts_train = np.sum(roi_meta['cls_npts'])
        roi_data = pd.DataFrame(np.zeros((npts_train, 4)), columns=["X", "Y", "Flat Index", "Label"])
        roi_data.loc[:, "Y"] = valid_ind[0]
        roi_data.loc[:, "X"] = valid_ind[1]
        # roi_data.loc[:, "Flat Index"] = (roi_data['Y'] - 0) * roi_meta['dims'][1] + roi_data['X']
        roi_data['Flat Index'] = np.ravel_multi_index((roi_data["Y"], roi_data["X"]), roi_meta["dims"], order="C")
        roi_data.loc[:, "Label"] = raster_data[valid_ind]

        return roi_meta, roi_data
        
    def readRaster(self, filename, band=None, out_nodata=-9999):
        """ Reads in a band from an images using GDAL
        Args:
          filename (str): filename to read from
          band (np.ndarray): 1D array of indices to the bands to read, 
              with first being 1, if None, read all bands
        Returns:
          dat (sequence of numpy 2D array): each list element is a band from the input image file.
          band (numpy 1d array): list of bands read to the *dat*
        """
        raster_profile = self.getRasterProfile(filename)
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        if ds is None:
            sys.stderr.write("Failed to open the file {0:s}\n".format(filename))
            return None

        if band is None:
            band = np.arange(ds.RasterCount, dtype=np.int)+1

        dat = [None for b in band]
        for n, b in enumerate(band):
            dat[n] = ds.GetRasterBand(b).ReadAsArray()
            tmp_ind = np.where(dat[n] == raster_profile["NoDataValue"][b-1])
            dat[n][tmp_ind] = out_nodata

        ds = None

        return dat, band


    def getRasterProfile(self, filename):
        """Retrieve the meta information of an image using GDAL
        Args:
          filename: str
              file name to read from
        Returns:
          meta: dict
              returned meta data records about the image
        """
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        if ds is None:
            sys.stderr.write("Failed to open the file {0:s}\n".format(filename))
            return None

        ncol = ds.RasterXSize
        nrow = ds.RasterYSize
        nbands = ds.RasterCount

        geo_trans = ds.GetGeoTransform()

        proj_ref = ds.GetProjectionRef()

        metadata = {dm:ds.GetMetadata(dm) for dm in ds.GetMetadataDomainList()}

        bands = [ds.GetRasterBand(idx+1) for idx in range(nbands)]
        nodata = [self._img_nodata if b.GetNoDataValue() is None else b.GetNoDataValue() for b in bands]
        category_names = [b.GetCategoryNames() for b in bands]
        dtype = [gdal_array.GDALTypeCodeToNumericTypeCode(b.DataType) for b in bands]

        return {"RasterXSize":ncol, "RasterYSize":nrow,
                "RasterCount":nbands, "DataType":dtype,
                "NoDataValue":nodata, "CategoryNames":category_names,
                "GeoTransform":geo_trans, "ProjectionRef":proj_ref,
                "Metadata":metadata}

def getCmdArgs():
    p = argparse.ArgumentParser(description="Image classification")

    p.add_argument("-i", '--images', dest="images", required=True, nargs='+', default=None, help="One or multiple image raster files to classify.")
    p.add_argument("-b", '--bands', dest="bands", required=False, nargs='*', default=None, help="One or multiple sequences of band indices to use for each image, with band indices within a sequence separated by ',' and different sequences separated by space. Default: None which uses all the bands of each image. Example for three images: --bands 1,2 2 2,3,4,5")

    p.add_argument('--trainF', dest="trainF", required=True, choices=['1', '2'], help="Format of training data, with the following choices, \n\t1: ENVI ROI ASCII\n\t2: GTiff ROI")
    p.add_argument('--train', dest="train", required=True, default=None, help="Training data file of known labels of training pixels.")
    p.add_argument('--trainX', dest="trainX", required=False, default=None, help="Multi-band raster file to provide the same features as the input images and bands for training samples.")

    p.add_argument("-c", '--classification', dest="cls", required=True, default=None, help="Classification file to output.")
    p.add_argument("--clsF", dest="clsF", required=False, default="ENVI", help="Raster format of the output file to write classification to. Default: ENVI.")

    p.add_argument("--diagnosis", dest="diagnosis", required=False, default=None, help="Ascii file name to output classification diagnosis information.")

    cmdargs = p.parse_args()

    return cmdargs
    
def main(cmdargs):
    img_files = cmdargs.images
    if cmdargs.bands is not None:
        bands = [map(int, bstr.strip().split(',')) for bstr in cmdargs.bands]
    else:
        bands = None

    train_fmt = {'1':"ENVI ROI ASCII", \
                 '2':"GTiff ROI"}

    ic = ImageClassifier()

    print "Reading image data ..."
    ic.readImageData(img_files, bands=bands)

    print "Reading training data ..."
    ic.readTrainingData(cmdargs.train, fmt=train_fmt[cmdargs.trainF])

    print "Running cross validation for preliminary accuracy guess ..."
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True, class_weight=None)
    cv_reps = 10
    con_mat, ntrain, ntest = ic.runCrossValidation(rf, test_size=0.5, n_reps=cv_reps)
    print "Cross validation using {0:d} training pixels and {1:d} test pixels".format(ntrain, ntest)
    print "Confusion matrix for the test pixels"
    print con_mat

    print "Running classification ..."
    rf = ic.runRandomForest(n_estimators=100, n_jobs=-1, oob_score=True, class_weight=None)

    print "Writing classification results ..."
    ic.writeClassData(cmdargs.cls, fmt=cmdargs.clsF)

    if cmdargs.diagnosis is not None:
        print "Writing classification diagnosis information ..."
        ic.writeRandomForestDiagnosis(rf, cmdargs.diagnosis)
        with open(cmdargs.diagnosis, "a") as outfobj:
            outfobj.write("\n## Training Random Forest Classifier with settings:\n")
            outfobj.write("{0:s}\n".format(str(rf)))
            outfobj.write("## Number of cross validation = {0:d}\n".format(cv_reps))
            outfobj.write("## Mean number of training samples = {0:d}\n".format(ntrain))
            outfobj.write("## Mean number of testing samples = {0:d}\n".format(ntest))
            outfobj.write("## Mean confusion matrix = \n".format())
            for r in range(con_mat.shape[0]):
                fmt_str = ",".join(["{{0[{0:d}]:.3f}}".format(j) for j in range(con_mat.shape[1])])+"\n"
                outfobj.write(fmt_str.format(con_mat[r, :]))

if __name__ == "__main__":
    cmdargs = getCmdArgs()
    main(cmdargs)
