import cv2
import os
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from keras.models import Model
from keras.utils import multi_gpu_model

MAX_NB_CLASSES = 2

#extract on the fly and dont save
def extract_vgg16_features_live(model, video_input_file_path):
    if os.path.exists(feature_output_file_path):
        return np.load(feature_output_file_path)
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    # seconds = 1
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    #frame_to_extract = 30
    #interval = int(frame_count/frame_to_extract)
    #if interval == 0:
    #    interval = 1
    fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))        
    multiplier = fps * seconds
    print('fps + multiplyer',seconds, fps, multiplier)
    while success:
        frameId = int(round(vidcap.get(1)))
        success, image = vidcap.read()
        #if frameId % interval == 0 and success: 
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        feature = model.predict(input).ravel()
        features.append(feature)
    unscaled_features = np.array(features)
    return unscaled_features



def extract_vgg16_features(model, video_input_file_path, feature_output_file_path):
    if os.path.exists(feature_output_file_path):
        return np.load(feature_output_file_path)
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    # seconds = 1
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_to_extract = 30
    interval = int(frame_count/frame_to_extract)
    if interval == 0:
        interval = 1
    #interval = 1
    fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))        
    # multiplier = fps * seconds
    # print('fps + multiplyer',seconds, fps, multiplier)
    while success:
        frameId = int(round(vidcap.get(1)))
        success, image = vidcap.read()
        if frameId % interval == 0 and success: 
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            #cv2.imsave(,image)
            input = img_to_array(img)
            input = np.expand_dims(input, axis=0)
            input = preprocess_input(input)
            feature = model.predict(input).ravel()
            #print(feature.shape)
            features.append(feature)

    unscaled_features = np.array(features)
    np.save(feature_output_file_path, unscaled_features)
    return unscaled_features


def scan_and_extract_vgg16_features(data_dir_path, output_dir_path, model=None, data_set_name=None):
    if data_set_name is None:
        data_set_name = 'UniCrime'

    input_data_dir_path = data_dir_path + '/' + data_set_name
    output_feature_data_dir_path = data_dir_path + '/' + output_dir_path

    if model is None:
        base_model = VGG16(weights='imagenet')
        model = Model(inputs=base_model.input,outputs=base_model.get_layer('block4_pool').output)
        model = multi_gpu_model(model)

        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
    
    if not os.path.exists(output_feature_data_dir_path):
        os.makedirs(output_feature_data_dir_path)

    y_samples = []
    x_samples = []

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + os.path.sep + f
        if not os.path.isfile(file_path):
            output_dir_name = f
            output_dir_path = output_feature_data_dir_path + os.path.sep + output_dir_name
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                output_feature_file_path = output_dir_path + os.path.sep + ff.split('.')[0] + '.npy'
                x = extract_vgg16_features(model, video_file_path, output_feature_file_path) # x.shape=(number_of_frames,7x7x512)
                print("X.shape ========: ", np.shape(x))
                #print("F ======== : ",f)
                y = f # f = label of folder
                # print("X====: ",x) # features extraction of VGG16 as np array.
                y_samples.append(y)
                x_samples.append(x)

        if dir_count == MAX_NB_CLASSES:
            break

    return x_samples, y_samples

