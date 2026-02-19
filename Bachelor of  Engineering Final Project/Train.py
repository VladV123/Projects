import tensorflow as tf
import os
import Classification_model
import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, MeanSquaredLogarithmicError, MeanAbsoluteError


class Custom_methods_for_my_model(Model):
    def build(self, input_shape):
        self.model.build(input_shape)
        super().build(input_shape)

    def __init__(self, drone, **kwargs):
        super().__init__(**kwargs)
        self.model = drone

    def compile(self, opt, classloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.opt = opt

    def train_step(self, batch):
        X, y = batch

        with tf.GradientTape() as tape:
            classes = self.model(X, training=True)
            batch_classloss = self.closs(tf.squeeze(y[0]), classes)

            grad = tape.gradient(batch_classloss, self.model.trainable_variables)

        opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"class_loss": batch_classloss}

    def test_step(self, batch):
        X, y = batch

        classes = self.model(X, training=False)

        batch_classloss = self.closs(tf.squeeze(y[0]), classes)

        return {"class_loss": batch_classloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)

    def show_architecture(self):
        self.model.summary()

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_rate):
        self.decay_rate = decay_rate
        self.initial_learning_rate = initial_learning_rate * 1 / (pow(self.decay_rate, 4))

    def __call__(self, step):

        self.initial_learning_rate = self.initial_learning_rate * self.decay_rate
        print(self.initial_learning_rate)
        return self.initial_learning_rate




imagini_antrenare = tf.data.Dataset.list_files("C:/Users/Aorus/Desktop/Slinding_windows/Train_full/Sectiuni/*.jpg", shuffle=False)
imagini_antrenare = imagini_antrenare.map(utils.load_image)
imagini_antrenare = imagini_antrenare.map(lambda x: tf.image.resize(x, (224, 224)))
imagini_antrenare = imagini_antrenare.map(lambda x: x / 255)

imagini_validare = tf.data.Dataset.list_files("C:/Users/Aorus/Desktop/Slinding_windows/validare/Sectiuni/*.jpg", shuffle=False)
imagini_validare = imagini_validare.map(utils.load_image)
imagini_validare = imagini_validare.map(lambda x: tf.image.resize(x, (224, 224)))
imagini_validare = imagini_validare.map(lambda x: x / 255)

etichete_antrenare = tf.data.Dataset.list_files("C:/Users/Aorus/Desktop/Slinding_windows/Train_full/Etichete_sectiuni/*.json", shuffle=False)
etichete_antrenare = etichete_antrenare.map(lambda x: tf.py_function(utils.load_labels, [x], [tf.uint8, tf.float32]))

etichete_validare = tf.data.Dataset.list_files("C:/Users/Aorus/Desktop/Slinding_windows/validare/Etichete_sectiuni/*.json", shuffle=False)
etichete_validare = etichete_validare.map(lambda x: tf.py_function(utils.load_labels, [x], [tf.uint8, tf.float32]))

antrenare = tf.data.Dataset.zip((imagini_antrenare, etichete_antrenare))
antrenare = antrenare.shuffle(5000)
antrenare = antrenare.batch(8)
antrenare = antrenare.prefetch(4)

validare = tf.data.Dataset.zip((imagini_validare, etichete_validare))
validare = validare.shuffle(1000)
validare = validare.batch(8)
validare = validare.prefetch(4)

model = Classification_model.Custom_My_Model(6,1)
final_model = Custom_methods_for_my_model(model)
classloss = tf.keras.losses.SparseCategoricalCrossentropy()
final_model.build(input_shape=(8,224,224, 3))
lr_schedule = MyLRSchedule(initial_learning_rate=0.001,decay_rate=0.6) 
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='class_loss', patience=3, restore_best_weights=True)
opt = tf.keras.optimizers.Adam(lr_schedule) 
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule) 
final_model.compile(opt, classloss)
final_model.fit(antrenare, epochs=20, validation_data=validare, callbacks=[early_stopping, lr_callback])
final_model.model.save('model_versiunea_finala_avion_cu_pasare_dr_0.6_initial_l_r_0.001_early_stopping_patience_3_adam.h5')