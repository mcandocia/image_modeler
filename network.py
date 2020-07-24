import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import sys
import time
import constants as c
from functools import partial
from utility import Clogger
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects

LOG_FILENAME='network.log'
logger=Clogger(LOG_FILENAME)



# quick way to pass in a layer
def keras_layer(layer_type, *args, **kwargs):
    func = {
        'conv': Conv2D,
        'relu': partial(Activation, 'relu'),
        'pool': MaxPooling2D,
        'dropout': Dropout,
        'dense': Dense,
        'flatten': Flatten,
        'softmax': partial(Activation, 'softmax'),
        'sigmoid': partial(Activation, 'sigmoid'),
        'leaky_relu': keras.layers.LeakyReLU,
    }[layer_type]

    logger.debug('Evaluating %s layer' % layer_type)
    return func(*args, **kwargs)


def custom_loss(t, p):
    # mean absolute sesteric error(?)
    return K.mean(K.pow(K.abs(t-p),2.5 ))


# 3x loss penalty for predictions over true value
def asymmetric_loss(t, p):
    return K.mean(
        K.pow(K.abs(t-p), 2) * (
            K.maximum(K.sign(t-p), 0.0) +
            3.0 *  K.maximum(K.sign(p-t), 0.0)
        )
    )

def asymmetric_loss2(t, p):
    return K.mean(
        K.pow(K.abs(t-p), 2) * (
            K.maximum(K.sign(t-p), 0.0) +
            2.0 *  K.maximum(K.sign(p-t), 0.0)
        )
    )


get_custom_objects().update(
        {
            "custom_loss": custom_loss,
            'asymmetric_loss': asymmetric_loss,
            'asymmetric_loss2': asymmetric_loss2,
        }
)

        

class Network:
    def __init__(
            self,
            loss='mse',
            opt=None,
            layer_plan = [
                # 256 => 244
                keras_layer('conv', 32, (7,7),dilation_rate=1),
                keras_layer('relu', ),

                # RECALCULATE
                keras_layer('conv', 36, (3,3)),
                keras_layer('relu',),
                
                # 244 => 122
                keras_layer('pool', pool_size=(3,3), strides=2),
                keras_layer('dropout', 0.05),

                # 122 => 118
                keras_layer('conv', 48, (5, 5)),
                keras_layer('relu',),

                # 118 => 40
                keras_layer('pool', pool_size=(3,3)),

                # 40 => 38
                keras_layer('conv', 60, (3,3)),
                keras_layer('relu',),

                # 40 => 20
                keras_layer('pool', pool_size=(2,2)),

                # 20 => 18
                keras_layer('conv', 72, (3,3)),
                keras_layer('relu'),
                
                keras_layer('flatten', ),
                keras_layer('dense', 48, ),
                keras_layer('relu',),
                
                keras_layer('dense', 48),
                # make it easier for network to estimate
                keras_layer('relu',),

                # make it easier for intermediate values
                keras_layer('dense', 12),
                keras_layer('relu',),

                keras_layer('dense', 4),
                keras_layer('relu',),

                keras_layer('dense', 2),
                keras_layer('relu',),

                # output
                keras_layer('dense', 1),
                #keras_layer('sigmoid'),

            ]
    ):
        logger.info('Initializing network')

        input_layer = keras.layers.Input(shape=c.INPUT_DIM_RGB)
        model=input_layer

        
        for layer in layer_plan:
            if isinstance(layer, list):
                logger.debug('Adding parallel layer of length %d' % len(layer))
                layers = [
                    x(model)
                    for x in layer
                ]
                model = keras.layers.concatenate(layers)
            else:
                logger.debug('Adding single layer')
                model = layer(model)


        model=keras.models.Model(
            inputs=input_layer,
            outputs=model
        )

        if opt is None:
            #opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
            opt = keras.optimizers.Adadelta()

        model.compile(
            loss=loss,
            optimizer=opt,
            metrics=['mse','mae',]
        )



        self.model = model
        self.logger=logger

    def __getattr__(self, *args, **kwargs):
        return getattr(self.model, *args, **kwargs)


