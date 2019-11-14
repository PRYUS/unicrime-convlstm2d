import numpy as np
from keras import backend as K
import os
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


def main():
    #K.set_image_dim_ordering('tf')
    K.set_image_data_format('channels_last')
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from keras_video_classifier.library.utility.plot_utils import plot_and_save_history
    from keras_video_classifier.library.recurrent_networks import VGG16LSTMVideoClassifier
    from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf
    
    data_set_name = 'UniCrime'
    input_dir_path = os.path.join(os.path.dirname(__file__), '/home/prathyush/Video-classifier-keras-master/data')
    output_dir_path = os.path.join(os.path.dirname(__file__), '/home/prathyush/Video-classifier-keras-master/results/models/', data_set_name)
    report_dir_path = os.path.join(os.path.dirname(__file__), '/home/prathyush/Video-classifier-keras-master/results/reports/', data_set_name)

    np.random.seed(42)

    # this line downloads the video files of UCF-101 dataset if they are not available in the very_large_data folder
    load_ucf(input_dir_path)

    classifier = VGG16LSTMVideoClassifier()

    history = classifier.fit(data_dir_path=input_dir_path, model_dir_path=output_dir_path, vgg16_include_top=False, data_set_name=data_set_name, test_size=0.1)

    plot_and_save_history(history, VGG16LSTMVideoClassifier.model_name,
                          report_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-hi-dim-history.png')


if __name__ == '__main__':
    main()
