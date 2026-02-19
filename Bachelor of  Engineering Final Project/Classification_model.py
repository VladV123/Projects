from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalMaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import models
from tensorflow.keras.regularizers import l2

def Create_Vgg_Cnn():
    # vgg_model = VGG16(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    # feature_layer = vgg_model.get_layer('block5_pool').output
    # model = Model(inputs=vgg_model.input, outputs=feature_layer)
    # return model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224 ,224, 3))
    model = models.Sequential()
    for layer in vgg_model.layers:
        model.add(layer)
    for layer in model.layers:
        layer.trainable = False
    return model

def Custom_My_Model(num_classes,num_max_pred):
    feature_layer = Create_Vgg_Cnn()
    global_max_pooling = GlobalMaxPooling2D()(feature_layer.output)
    flatten = Flatten()(global_max_pooling)
    fc_layer = Dense(256, activation='relu', kernel_initializer='he_normal',
    kernel_regularizer=l2(0.01))(flatten)
    dropout_layer_class = Dropout(0.5)(fc_layer)
    fc_layer1 = Dense(128, activation='relu',
    kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(dropout_layer_class)
    output_layer = Dense(num_classes, activation='softmax', name='output_layer',
    kernel_initializer='glorot_uniform')(fc_layer1)
    model = Model(inputs=feature_layer.input, outputs=output_layer)
    return model