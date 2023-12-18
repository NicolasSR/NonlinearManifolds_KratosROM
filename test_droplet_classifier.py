import numpy as np
import tensorflow as tf

from keras.models import Model, load_model
from keras import layers
from keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


if __name__=="__main__":

    X=np.random.rand(200,6)
    y_possibilities = np.eye(2)
    Y=y_possibilities[np.random.choice(y_possibilities.shape[0], size=200)]

    input=layers.Input((6,))
    layer_out=layers.Dense(100, "relu", kernel_initializer="he_normal")(input)
    layer_out=layers.Dense(100, "relu", kernel_initializer="he_normal")(layer_out)
    output_class=layers.Dense(2, "softmax", kernel_initializer="glorot_normal", name="out_class")(layer_out)
    output_regr=layers.Dense(2, "linear", kernel_initializer="he_normal", name="out_regr")(layer_out)


    my_nn = Model(input, [output_class,output_regr])

    my_nn.compile(optimizer=Adam(0.001), loss={"out_class": "categorical_crossentropy", "out_regr": "mse"}, metrics={"out_class": "categorical_accuracy"})

    my_nn.summary()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=Y)

    my_nn.fit(X_train, Y_train, batch_size=16, validation_split=0.2, epochs=50, shuffle=True)

    my_nn.save('my_model.keras')

    copy_nn = load_model('my_model.keras')
    
    copy_nn.summary()

    Y_test_pred = my_nn.predict(X_test, batch_size=16)
    roc_curve(Y_test, Y_test_pred)
    roc_auc_score(Y_test, Y_test_pred)
    confusion_matrix(Y_test, Y_test_pred)