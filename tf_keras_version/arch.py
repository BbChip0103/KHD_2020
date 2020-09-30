import tensorflow as tf # Tensorflow 2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def cnn():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5),
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    input_shape=(512,512, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(rate = 0.2))

    model.add(tf.keras.layers.Conv2D(16, (3, 3),
                                    kernel_initializer='he_normal',
                                    activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64,
                                    kernel_initializer='he_normal',
                                    activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'relu'))
    return model


def VGG16():
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(300,300,3),
        pooling=None,
        classes=1,
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
#     predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name='CustomVGG16')    
    return model


def ResNet50():
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(300,300,3),
        pooling=None,
        classes=1,
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
#     predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name='CustomResnet50')    
    return model


def DenseNet121():
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(300,300,3),
        pooling=None,
        classes=1,
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
#     predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name='CustomDensenet101')    
    return model


def EfficientNetB0():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(300,300,3),
        pooling=None,
        classes=1,
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
#     predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name='CustomEfficientNetB0')    
    return model


def MobileNetV2():
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(300,300,3),
        pooling=None,
        classes=1,
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
#     predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name='CustomMobileNetV2')    
    return model


def Xception():
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(300,300,3),
        pooling=None,
        classes=1,
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
#     predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name='CustomXception')    
    return model