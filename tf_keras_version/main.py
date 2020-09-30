import os
import argparse
import sys
import time
import numpy as np
from matplotlib.image import imread
import tensorflow as tf # Tensorflow 2
import arch
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM 
import math
import tensorflow.keras as keras
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical


######################## DONOTCHANGE ###########################
def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_weights(os.path.join(dir_name, 'model'))
        print('model loaded!')

    def infer(image_path):
        result = []
        X = PathDataset(image_path, labels=None, batch_size = batch_size)
        y_hat = model.predict(X)
        result.extend(np.argmax(y_hat, axis=1))

        print('predicted')
        return np.array(result)

    nsml.bind(save=save, load=load, infer=infer)


def path_loader (root_path):
    image_path = []
    image_keys = []
    for _,_,files in os.walk(os.path.join(root_path,'train_data')):
        for f in files:
            path = os.path.join(root_path,'train_data',f)
            if path.endswith('.png'):
                image_keys.append(int(f[:-4]))
                image_path.append(path)

    return np.array(image_keys), np.array(image_path)


def label_loader (root_path, keys):
    labels_dict = {}
    labels = []
    with open (os.path.join(root_path,'train_label'), 'rt') as f :
        for row in f:
            row = row.split()
            labels_dict[int(row[0])] = (int(row[1]))
    for key in keys:
        labels = [labels_dict[x] for x in keys]
    return labels
############################################################


class PathDataset(tf.keras.utils.Sequence): 
    def __init__(self,image_path, labels=None, batch_size=128, test_mode= True): 
        self.image_path = image_path
        self.labels = labels
        self.mode = test_mode
        self.batch_size = batch_size
        
    def load_image(self, image_path):
        img = map_func(image_path, None)
        return img

    def __getitem__(self, idx): 
        image_paths = self.image_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([self.load_image(x) for x in image_paths])
        batch_x = prep_func(batch_x)
                ### REQUIRED: PREPROCESSING ###

        if self.mode:
            return batch_x
        else: 
            batch_y = np.array(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size])
            return batch_x, batch_y

    def __len__(self):
        return math.ceil(len(self.image_path) / self.batch_size)

def map_func(image_path, cap):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.dtypes.cast(img, tf.float32)
    img = tf.image.resize(img, (300, 300))
    
#     cap_onehot = keras.utils.to_categorical(
#         cap, num_classes=2, dtype='float32'
#     )

    return img, cap 

def prep_func(image, cap):
#     result_image = tf.keras.applications.resnet_v2.preprocess_input(image)
    result_image = image
    result_image -= tf.keras.backend.mean(
        result_image, axis=0, keepdims=True
    )
    result_image /= (tf.keras.backend.std(
        result_image, axis=0, keepdims=True
    )+1e-9)
    
    return result_image, cap    
    
if __name__ == '__main__':

    ########## ENVIRONMENT SETUP ############
    args = argparse.ArgumentParser()

    ########### DONOTCHANGE: They are reserved for nsml ###################
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    ######################################################################

    # hyperparameters
    args.add_argument('--epoch', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--learning_rate', type=int, default=0.00005)

    config = args.parse_args()

    # training parameters
    num_epochs = config.epoch
    batch_size = config.batch_size
    num_classes = 2
    learning_rate = config.learning_rate  

    # model setting ## 반드시 이 위치에서 로드해야함
    model = arch.ResNet50() # ResNet50, MobileNetV2, 

    # Loss and optimizer
    model.compile(
        tf.keras.optimizers.Adam(),
#         loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    ############ DONOTCHANGE ###############
    bind_model(model)
    if config.pause: ## test mode 일때는 여기만 접근
        print('Inferring Start...')
        nsml.paused(scope=locals())
    #######################################

    if config.mode == 'train': ### training mode 일때는 여기만 접근
        print('Training Start...')

        ############ DONOTCHANGE: Path loader ###############
        root_path = os.path.join(DATASET_PATH,'train')        
        image_keys, image_path = path_loader(root_path)
        labels = label_loader(root_path, image_keys)
        ##############################################
        
#         X = PathDataset(image_path, labels, batch_size = batch_size, test_mode=False)

        labels_onehot = np.array([to_categorical(label, 2) for label in labels])
        img_name_train, img_name_val, cap_train, cap_val = train_test_split(image_path, labels_onehot, test_size=0.2, random_state=2718)

        dataset_train = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
        dataset_train = dataset_train.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_train = dataset_train.batch(batch_size)
        dataset_train = dataset_train.map(prep_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
        dataset_val = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
        dataset_val = dataset_val.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_val = dataset_val.batch(batch_size)
        dataset_val = dataset_val.map(prep_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
 
        for epoch in range(num_epochs):
            print('=== Epoch: {:06d} ==='.format(epoch))
            hist = model.fit(
                dataset_train,
                validation_data=dataset_val, 
                shuffle=True,
#                 batch_size=config['batch_size']                
            )

            nsml.report(
                summary=True, step=epoch, epoch_total=num_epochs, 
                loss=hist.history['loss'], accuracy=hist.history['accuracy'],
                val_loss=hist.history['val_loss'], val_accuracy=hist.history['val_accuracy'],
            )#, acc=train_acc)
            nsml.save(epoch)
