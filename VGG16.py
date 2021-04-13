from numpy.random import seed
seed(1)
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import h5py
import scipy.io as sio
import cv2
import glob
import gc

from keras.models import load_model, Model, Sequential
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
		 	  Activation, Dense, Dropout, ZeroPadding2D)
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from keras.layers.advanced_activations import ELU
from tensorflow.keras.applications import EfficientNetB0

from PIL import Image
import numpy
from numpy import asarray
import os
import random
import math

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# CHANGE THESE VARIABLES ---
data_folder = 'URFD_opticalflow/'
mean_file = 'flow_mean.mat'
vgg_16_weights = 'weights.h5'
best_model_path = 'VGG16_best_model/' #directory to store best model
fold_best_model_path = '' #to be determined by k-fold cross validation
save_features = False
save_plots = True

#Can modify!
learning_rate = 0.0001
mini_batch_size = 32
weight_0 = 1
epochs = 10
learning_rate = 0.0001
training_ratio = 0.8 #portion of data used for training and valuation
test_ratio = 0.2 #portion of data used for testing
threshold = 0.5 # Threshold to classify between positive and negative
total_num_folds = 4

#for keeping track of fold performance
sensitivities = []
specificities = []
fars = []
mdrs = []
accuracies = []


#================Preprocessing================
label_assignment = {} #maps folder name to correct label

fall_path = data_folder + 'Falls/'
not_fall_path = data_folder + 'NotFalls/'

#==============Func to filter out short videos=================
num_frames = 20 #20 is set to function with provided weights
def is_sufficient_len(vid_path):
    img_paths = os.listdir(vid_path)
    return len(img_paths) > num_frames

#================Assign <x, c(x)> to dictionary================
data_paths = [] #data paths used from training and valuation
test_data_paths = [] #data paths used for testing
fall_vids = [fall_path + vid for vid in os.listdir(fall_path)]

for vids in fall_vids[:int(training_ratio * (len(fall_vids)))]:
    if is_sufficient_len(vids):
        case = {vids: 1} #1 is fall
        label_assignment.update(case)
        data_paths.append(vids)

for vids in fall_vids[int(training_ratio * (len(fall_vids))) : int((training_ratio + test_ratio) * (len(fall_vids)))]:
    if is_sufficient_len(vids):
        case = {vids: 1} #1 is fall
        label_assignment.update(case)
        test_data_paths.append(vids)

not_fall_vids = [not_fall_path + vid for vid in os.listdir(not_fall_path)]


for vids in not_fall_vids[:int(training_ratio * (len(fall_vids)))]:
    if is_sufficient_len(vids):
        case = {vids: 0} #0 is no fall
        label_assignment.update(case)
        data_paths.append(vids)

for vids in not_fall_vids[int(training_ratio * (len(fall_vids))) : int((training_ratio + test_ratio) * (len(fall_vids)))]:
    if is_sufficient_len(vids):
        case = {vids: 0} #0 is no fall
        label_assignment.update(case)
        test_data_paths.append(vids)

#==============Func to break list of frames into mutiple batches of 20 frames=================
def sample(lst):
    output = []
    if len(lst) < num_frames:
        return ouput
    interval_size = len(lst)// num_frames
    starting_frame = 0
    while starting_frame <= len(lst) - interval_size * 20:
        output.append(lst[starting_frame::interval_size][0:20])
        starting_frame += 1
    return output
#===============build input instances===============================
def build_data(paths):
    x = [] #list of video data represented by np.array matrices.
    y = [] #list of labels.
    for vid_path in paths:
        label = label_assignment[vid_path] # get c(x)
        #print(label)
        img_paths = os.listdir(vid_path)
        for batch in sample(img_paths):
            img_set = []
            for img_path in batch:
                img = Image.open(vid_path + '/' + img_path)
                img = asarray(img)
                #print(img.shape)
                img_set.append(img)
            img_set = np.array(img_set)
            #print(img_set.shape)
            img_set = img_set.reshape(224, 224, 20) #convert list of img matrices to nparray
            x.append(img_set)
            y.append(np.array([label]))
    return np.array(x), np.array(y)

x_val, y_val = build_data(test_data_paths) #build data sets for evaluation
print(x_val.shape)
print(y_val.shape)
#======================function to plot graph===============
def plot_training_info(case, metrics, save, history):
    '''
    Function to create plots for train and validation loss and accuracy
    Input:
    * case: name for the plot, an 'accuracy.png' or 'loss.png'
	will be concatenated after the name.
    * metrics: list of metrics to store: 'loss' and/or 'accuracy'
    * save: boolean to store the plots or only show them.
    * history: History object returned by the Keras fit function.
    '''
    val = False
    if 'val_accuracy' in history and 'val_loss' in history:
        val = True

    print('val' + str(val))
    print(history)
    plt.ioff()
    if 'accuracy' in metrics:
        fig = plt.figure()
        plt.plot(history['accuracy'])
        if val: plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        if val:
            plt.legend(['train', 'val'], loc='upper left')
        else:
            plt.legend(['train'], loc='upper left')
        if save == True:
            plt.savefig(case + 'accuracy.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)

    # summarize history for loss
    if 'loss' in metrics:
        fig = plt.figure()
        plt.plot(history['loss'])
        if val: plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        if val:
            plt.legend(['train', 'val'], loc='upper left')
        else:
            plt.legend(['train'], loc='upper left')
        if save == True:
            plt.savefig(case + 'loss.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)
#===============runs the cnn on a given fold=====================================
def run(train, test, fold_number):
    global fold_best_model_path
    x_train, y_train = build_data(train)
    x_test, y_test = build_data(test)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    #print(x_train.shape)
    #rint(y_train.shape)
    #print(x_test.shape)
    #print(y_test.shape)
    # ========================================================================
    # VGG-16 ARCHITECTURE
    # ========================================================================
    num_features = 4096
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 20)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(num_features, name='fc6',
            kernel_initializer='glorot_uniform'))
    #print(model.layers[-1].output_shape)

    # ========================================================================
    # WEIGHT INITIALIZATION
    # ========================================================================
    layerscaffe = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
            'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
            'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
    h5 = h5py.File(vgg_16_weights, 'r')

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # Copy the weights stored in the 'vgg_16_weights' file to the
    # feature extractor part of the VGG16
    for layer in layerscaffe[:-3]:
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (2,3,1,0))
        w2 = w2[::-1, ::-1, :, :]
        b2 = np.asarray(b2)
        layer_dict[layer].set_weights((w2, b2))
        layer_dict[layer].trainable = False

    # Copy the weights of the first fully-connected layer (fc6)
    layer = layerscaffe[-3]
    w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
    w2 = np.transpose(np.asarray(w2), (1,0))
    b2 = np.asarray(b2)
    layer_dict[layer].set_weights((w2, b2))
    layer_dict[layer].trainable = False

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
        epsilon=1e-08)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
            metrics=['accuracy'])
    # ==================== CLASSIFIER ========================
    extracted_features = model.output
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99,
                epsilon=0.001)(extracted_features)
        x = Activation('relu')(x)
    else:
        x = ELU(alpha=1.0)(extracted_features)

    x = Dropout(0.9)(x)
    x = Dense(4096, name='fc2_{}'.format(fold_number), kernel_initializer='glorot_uniform')(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
        x = Activation('relu')(x)
    else:
        x = ELU(alpha=1.0)(x)
    x = Dropout(0.8)(x)
    #x = Dense(4096, name='fc3_{}'.format(fold_number), kernel_initializer='glorot_uniform')(x)
    x = Dense(1, name='predictions{}'.format(fold_number),
                kernel_initializer='glorot_uniform')(x)
    x = Activation('sigmoid')(x)
    classifier = Model(inputs=model.inputs, outputs=x, name='classifier_{}'.format(fold_number))
    fold_best_model_path = best_model_path + 'urfd_fold_{}.h5'.format(
                            fold_number)
    classifier.compile(optimizer=adam, loss='binary_crossentropy',
            metrics=['accuracy'])

    metric = 'val_loss'
    
    e = EarlyStopping(monitor=metric, min_delta=0, patience=100,
        mode='auto')
    c = ModelCheckpoint(fold_best_model_path, monitor=metric,
            save_best_only=True,
            save_weights_only=False, mode='auto')
    callbacks = [e, c]
    class_weight = {0: weight_0, 1: 1}
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_number} ...')
    history = classifier.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=mini_batch_size,
        epochs=epochs,
        shuffle=True,
        class_weight=class_weight,
        callbacks=callbacks
    )
    #================testing=========================================================
    classifier = load_model(fold_best_model_path)
    predicted = classifier.predict(x_val)
    for i in range(len(predicted)):
        if predicted[i] < threshold:
            predicted[i] = 0
        else:
            predicted[i] = 1
    # Array of predictions 0/1
    predicted = np.asarray(predicted).astype(int)
    # Compute metrics and print them
    cm = confusion_matrix(y_val, predicted,labels=[0,1])
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]
    tpr = tp/float(tp+fn)
    fpr = fp/float(fp+tn)
    fnr = fn/float(fn+tp)
    tnr = tn/float(tn+fp)
    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    specificity = tn/float(tn+fp)
    f1 = 2*float(precision*recall)/float(precision+recall)
    accuracy = accuracy_score(y_val, predicted)

    print('FOLD {} results:'.format(fold_number))
    print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
    print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(
                    tpr,tnr,fpr,fnr))
    print('Sensitivity/Recall: {}'.format(recall))
    print('Specificity: {}'.format(specificity))
    print('Precision: {}'.format(precision))
    print('F1-measure: {}'.format(f1))
    print('Accuracy: {}'.format(accuracy))

    # Store the metrics for this epoch
    sensitivities.append(tp/float(tp+fn))
    specificities.append(tn/float(tn+fp))
    fars.append(fpr)
    mdrs.append(fnr)
    accuracies.append(accuracy)



#===============run k_fold cross validation=====================================
fold_number = 0
batch_norm = True
kfold = KFold(n_splits = total_num_folds, shuffle = True)
for train, test in kfold.split(data_paths):
    fold_number += 1
    run([data_paths[i] for i in train], [data_paths[i] for i in test], fold_number)



print('5-FOLD CROSS-VALIDATION RESULTS ===================')
print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities)*100.,
                np.std(sensitivities)*100.))
print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(specificities)*100.,
                np.std(specificities)*100.))
print("FAR: %.2f%% (+/- %.2f%%)" % (np.mean(fars)*100.,
            np.std(fars)*100.))
print("MDR: %.2f%% (+/- %.2f%%)" % (np.mean(mdrs)*100.,
            np.std(mdrs)*100.))
print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies)*100.,
                np.std(accuracies)*100.))