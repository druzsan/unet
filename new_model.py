from keras import layers, models


def double_conv2d(x, filters, kernel_size, padding='same', batch_normalization=False):
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer='he_normal')(x)
    if batch_normalization:
        x = layers.BatchNormalization(axis=3, scale=False)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer='he_normal')(x)
    if batch_normalization:
        x = layers.BatchNormalization(axis=3, scale=False)(x)
    x = layers.Activation('relu')(x)
    return x


def unet(weights=None, input_shape=(256, 256, 1), classes=1, background_as_class=False, batch_normalization=False):
    """
    Instantiates the U-Net architecture for Keras

    Note that convolutions in this model have `padding='same'` instead of `padding='valid'` in reference paper
    in order to have the same input and output shapes
    :param weights: optional path to the weights file to be loaded (random initialization if `None`)
    :param input_shape: optional input shape tuple
    :param classes: optional number of classes to predict
    :param background_as_class: whether to create additional channel for background class
    :param batch_normalization: whether to apply batch normalization after each convolution
    :return: Keras model instance
    """
    if background_as_class is True:
        # Add one more class for background
        classes += 1
        # Classes (and background) probabilities in each pixel are conditional dependent
        top_activation = 'softmax'
    else:
        # Classes (and background) probabilities in each pixel are independent
        # Some pixel is background if all classes activations in this pixel are nearly zeros
        top_activation = 'sigmoid'

    inputs = layers.Input(input_shape)

    # conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = double_conv2d(inputs, 64, 3, padding='same', batch_normalization=batch_normalization)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = double_conv2d(pool1, 128, 3, padding='same', batch_normalization=batch_normalization)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    # conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = double_conv2d(pool2, 256, 3, padding='same', batch_normalization=batch_normalization)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    # conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = double_conv2d(pool3, 512, 3, padding='same', batch_normalization=batch_normalization)
    conv4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    # conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = double_conv2d(pool4, 1024, 3, padding='same', batch_normalization=batch_normalization)
    conv5 = layers.Dropout(0.5)(conv5)

    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), activation='relu', kernel_initializer='he_normal')(conv5)
    merge6 = layers.concatenate([conv4, up6], axis=3)
    # conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    # conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = double_conv2d(merge6, 512, 3, padding='same', batch_normalization=batch_normalization)

    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), activation='relu', kernel_initializer='he_normal')(conv6)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    # conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    # conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = double_conv2d(merge7, 256, 3, padding='same', batch_normalization=batch_normalization)

    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), activation='relu', kernel_initializer='he_normal')(conv7)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    # conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    # conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = double_conv2d(merge8, 128, 3, padding='same', batch_normalization=batch_normalization)

    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), activation='relu', kernel_initializer='he_normal')(conv8)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    # conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    # conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = double_conv2d(merge9, 64, 3, padding='same', batch_normalization=batch_normalization)
    conv9 = layers.Conv2D(classes, 1, activation=top_activation, padding='same', kernel_initializer='he_normal')(conv9)

    model = models.Model(inputs, conv9)

    if weights is not None:
        model.load_weights(weights)

    return model
