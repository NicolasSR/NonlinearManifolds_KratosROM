import numpy as np
import tensorflow as tf

from keras.models import Model
from keras import layers


if __name__=="__main__":

    input=layers.Input((6,))
    layer_out=layers.Dense(100, "relu", kernel_initializer="he_normal")(input)
    layer_out=layers.Dense(100, "relu", kernel_initializer="he_normal")(layer_out)
    output_class=layers.Dense(3, "softmax", kernel_initializer="glorot_normal", name="out_class")(layer_out)
    output_regr=layers.Dense(3, "linear", kernel_initializer="he_normal", name="out_regr")(layer_out)


    my_nn = Model(input, [output_class,output_regr])

    my_nn.compile(optimizer="adam", loss={"out_class": "categorical_crossentropy", "out_regr": "mse"}, metrics={"out_class": "categorica_accuracy"})

    my_nn.summary()