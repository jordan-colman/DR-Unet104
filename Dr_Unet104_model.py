import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Cropping2D, Dropout, Conv2D, Conv2DTranspose, Activation, BatchNormalization, ZeroPadding2D

def input_bottleneck_block(X, filters, kernel_size=(3, 3), padding="same", strides=1):
    F1, F2, F3 = filters

    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1),strides=(1,1),padding='valid',kernel_initializer='he_normal')(X)

    #conv 1
    X = Conv2D(filters=F1, kernel_size=(1,1),strides=(1,1),padding='valid',kernel_initializer='he_normal')(X)

    #conv 2
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F2, kernel_size=kernel_size, strides=(1, 1), padding='same',kernel_initializer='he_normal')(X)

    #conv 3
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F3, kernel_size=(1,1),strides=(1,1),padding='valid',kernel_initializer='he_normal')(X)

    ## concatenate
    X = Add()([X,X_shortcut])
    return X

def bottleneck_block(X,filters, kernel_size=(3,3),first=False):
    F1, F2, F3 = filters

    X_shortcut = X

    #conv 1
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    if first:
        X_shortcut = X
    X = Conv2D(filters=F1, kernel_size=(1,1),strides=(1,1),padding='valid',kernel_initializer='he_normal')(X)

    #conv 2
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F2, kernel_size=kernel_size, strides=(1, 1), padding='same',kernel_initializer='he_normal')(X)

    #conv 3

    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F3, kernel_size=(1,1),strides=(1,1),padding='valid',kernel_initializer='he_normal')(X)

    ## concatenate
    X = Add()([X,X_shortcut])


    return X

def bottleneck_downsample_block(X, filters, kernel_size=(3,3), s=2):
    F1, F2, F3 = filters

    X_shortcut = X

    # conv 1
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding='valid',kernel_initializer='he_normal')(X)


    # conv 2
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F2, kernel_size=kernel_size, strides=(1,1), padding='same',kernel_initializer='he_normal')(X)


    # conv 3
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid',kernel_initializer='he_normal')(X)


    #short cut path
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding='valid',kernel_initializer='he_normal')(X_shortcut)


    #concatenate
    X = Add()([X,X_shortcut])


    return X


def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    #shortcut using convolution to reduce number of features as have been concatenated with encoder laevel output
    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides,kernel_initializer='he_normal')(x)

    #convolutional layer 1
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)

    #convolutional layer 2
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)

    output = tf.keras.layers.Add()([shortcut, x])
    return output

def upsample_and_concatenation(x, xskip, layers):
    u = Conv2DTranspose(layers, (3, 3), strides=(2, 2), padding='same')(x)
    c = tf.keras.layers.Concatenate()([u, xskip])
    return c

def DR_Unet104(image_width, image_height, num_classes):
    f = [16, 32, 64, 128, 256, 512]
    inputs = tf.keras.layers.Input((image_width, image_height, 4))

    dropout=0.5
    ###Encoder

    ## Input and Level 1
    e0 = inputs
    e0 = input_bottleneck_block(e0,filters=[16,16,64])
    e0 = bottleneck_block(e0,filters=[16,16,64])
    ## level 2
    e1 = bottleneck_downsample_block(e0, filters=[32,32,128])
    e1 = bottleneck_block(e1, filters=[32,32,128])
    e1 = bottleneck_block(e1, filters=[32,32,128])
    e1 = Dropout(dropout)(e1)

    ## level 3
    e2 = bottleneck_downsample_block(e1, filters=[64,64,256])
    e2 = bottleneck_block(e2, filters=[64,64,256])
    e2 = bottleneck_block(e2, filters=[64, 64, 256])
    e2 = Dropout(dropout)(e2)

    ## level 4
    e3 = bottleneck_downsample_block(e2, filters=[128,128,512])
    e3 = bottleneck_block(e3, filters=[128,128,512])
    e3 = bottleneck_block(e3, filters=[128,128,512])
    e3 = bottleneck_block(e3, filters=[128,128,512])
    e3 = bottleneck_block(e3, filters=[128,128,512])
    e3 = Dropout(dropout)(e3)
    e3 = ZeroPadding2D((1,1))(e3)

    ## level 5
    e4 = bottleneck_downsample_block(e3, filters=[256,256,1024])
    e4 = bottleneck_block(e4, filters=[256,256,1024])
    e4 = bottleneck_block(e4, filters=[256,256,1024])
    e4 = bottleneck_block(e4, filters=[256, 256, 1024])
    e4 = bottleneck_block(e4, filters=[256, 256, 1024])
    e4 = bottleneck_block(e4, filters=[256,256,1024])
    e4 = bottleneck_block(e4, filters=[256,256,1024])
    e4 = bottleneck_block(e4, filters=[256, 256, 1024])
    e4 = bottleneck_block(e4, filters=[256, 256, 1024])
    e4 = bottleneck_block(e4, filters=[256,256,1024])
    e4 = bottleneck_block(e4, filters=[256,256,1024])
    e4 = bottleneck_block(e4, filters=[256, 256, 1024])
    e4 = bottleneck_block(e4, filters=[256, 256, 1024])
    e4 = bottleneck_block(e4, filters=[256,256,1024])
    e4 = Dropout(dropout)(e4)


    ### Bridge
    e5 = bottleneck_downsample_block(e4, filters=[512, 512, 2048])
    e5 = bottleneck_block(e5, filters=[512, 512, 2048])
    e5 = bottleneck_block(e5, filters=[512, 512, 2048])
    e5 = bottleneck_block(e5, filters=[512, 512, 2048])
    e5 = Dropout(dropout)(e5)


    ### Decoder

    #Level 5
    u0 = upsample_and_concatenation(e5, e4, 1024)
    d0 = residual_block(u0, f[5])
    d0 = Dropout(dropout)(d0)

    #Level 4
    u1 = upsample_and_concatenation(d0, e3, 512)
    u1 = Cropping2D((1,1))(u1)
    d1 = residual_block(u1, f[4])
    d1 = Dropout(dropout)(d1)

    #Level 3
    u2 = upsample_and_concatenation(d1, e2, 256)
    d2 = residual_block(u2, f[3])
    d2 = Dropout(dropout)(d2)

    #Level 2
    u3 = upsample_and_concatenation(d2, e1, 128)
    d3 = residual_block(u3,f[2])
    d3 = Dropout(dropout)(d3)

    #Level 1
    u4 = upsample_and_concatenation(d3, e0, 64)
    d4 = residual_block(u4,f[1])
    X = BatchNormalization(axis=3)(d4)
    X = Activation('relu')(X)

    #Output layer
    outputs = Conv2D(num_classes, (1, 1), name='output_layer')(X)
    model = Model(inputs=[inputs], outputs=[outputs], name='DR_Unet104')

    return model
