from keras import backend, layers, models


def double_conv2d(x, filters, kernel_size, padding='same', batch_normalization=False, dropout_rate=0.):
    """
    Building block of U-Net architecture: two successive identical convolutions + batch normalization and dropout
    afterwards. Parameters `x`, `filters`, `kernel_size` and `padding` are the same as in the regular Keras Conv2D
    layer and valid for both convolutions.

    :param x: [4D tensor] input
    :param filters: [integer] number of filters or number of channels in the output
    :param kernel_size: [integer/tuple] size of convolution kernels
    :param padding: ['valid'/'same'] convolution padding
    :param batch_normalization: [bool] whether to apply optional batch normalization between convolution and ReLU
    :param dropout_rate: [float] optional dropout rate (dropout is not applied if `dropout_rate=0.0`)
    :return: [4D tensor] output
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer='he_normal')(x)
    if batch_normalization is True:
        x = layers.BatchNormalization(axis=bn_axis, scale=False)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer='he_normal')(x)
    if batch_normalization is True:
        x = layers.BatchNormalization(axis=bn_axis, scale=False)(x)
    x = layers.Activation('relu')(x)
    if dropout_rate > 0.:
        x = layers.Dropout(dropout_rate)(x)
    return x


def unet(weights=None,
         input_shape=(256, 256, 1),
         classes=1,
         background_as_class=False,
         batch_normalization=False,
         dropout_rate=0.):
    """
    Instantiates the U-Net architecture for Keras

    Note that convolutions in this model have `padding='same'` instead of `padding='valid'` in reference paper
    in order to have the same input and output shapes

    :param weights: optional path to the weights file to be loaded (random initialization if `None`)
    :param input_shape: [integer/tuple] optional number of input channels or input shape
    :param classes: [int] optional number of classes to predict
    :param background_as_class: [bool] whether to create additional channel for background class
    :param batch_normalization: [bool] whether to apply batch normalization after each convolution
    :param dropout_rate: [integer/tuple/list] dropout rate to apply to all building blocks or
                         tuple/list of size 9 with block-wise dropout rates
    :return: Keras model instance
    """

    if isinstance(input_shape, int):
        # As U-Net is fully-convolutional network, in fact it requires no input height and width
        if backend.image_data_format() == 'channels_last':
            input_shape = (None, None, input_shape)
        else:
            input_shape = (input_shape, None, None)
    elif isinstance(input_shape, tuple) and len(input_shape) == 3:
        if backend.image_data_format() == 'channels_last':
            input_height, input_width = input_shape[0], input_shape[1]
        else:
            input_height, input_width = input_shape[1], input_shape[2]
        if input_height % 16 != 0 or input_width % 16 != 0:
            raise ValueError("Input height and width should be a multiply of 16 in order to do 4 down-samplings and "
                             "then 4 up-samplings correctly")
    else:
        raise ValueError("The `input_shape` argument should be either integer (number of input channels)"
                         "or tuple of size 3 with input shape")

    if background_as_class is True:
        # Add one more class for background
        classes += 1
        # Classes (and background) probabilities in each pixel are conditional dependent
        top_activation = 'softmax'
    else:
        # Classes (and background) probabilities in each pixel are independent
        # Some pixel is background if all classes activations in this pixel are nearly zeros
        top_activation = 'sigmoid'

    if isinstance(dropout_rate, float):
        dropout_rate = [dropout_rate] * 9
    elif not isinstance(dropout_rate, tuple) and not isinstance(dropout_rate, list) or len(dropout_rate) != 9:
        raise ValueError("The `dropout_rate` argument should be either float (the same dropout rate"
                         "for all building blocks) or list/tuple of size 9 with block-wise dropout rates")

    channel_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    inputs = layers.Input(input_shape)

    conv1 = double_conv2d(inputs, 64, 3, padding='same',
                          batch_normalization=batch_normalization, dropout_rate=dropout_rate[0])
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = double_conv2d(pool1, 128, 3, padding='same',
                          batch_normalization=batch_normalization, dropout_rate=dropout_rate[1])
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = double_conv2d(pool2, 256, 3, padding='same',
                          batch_normalization=batch_normalization, dropout_rate=dropout_rate[2])
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = double_conv2d(pool3, 512, 3, padding='same',
                          batch_normalization=batch_normalization, dropout_rate=dropout_rate[3])
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = double_conv2d(pool4, 1024, 3, padding='same',
                          batch_normalization=batch_normalization, dropout_rate=dropout_rate[4])

    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), activation='relu', kernel_initializer='he_normal')(conv5)
    merge6 = layers.concatenate([conv4, up6], axis=channel_axis)
    conv6 = double_conv2d(merge6, 512, 3, padding='same',
                          batch_normalization=batch_normalization, dropout_rate=dropout_rate[5])

    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), activation='relu', kernel_initializer='he_normal')(conv6)
    merge7 = layers.concatenate([conv3, up7], axis=channel_axis)
    conv7 = double_conv2d(merge7, 256, 3, padding='same',
                          batch_normalization=batch_normalization, dropout_rate=dropout_rate[6])

    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), activation='relu', kernel_initializer='he_normal')(conv7)
    merge8 = layers.concatenate([conv2, up8], axis=channel_axis)
    conv8 = double_conv2d(merge8, 128, 3, padding='same',
                          batch_normalization=batch_normalization, dropout_rate=dropout_rate[7])

    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), activation='relu', kernel_initializer='he_normal')(conv8)
    merge9 = layers.concatenate([conv1, up9], axis=channel_axis)
    conv9 = double_conv2d(merge9, 64, 3, padding='same',
                          batch_normalization=batch_normalization, dropout_rate=dropout_rate[8])

    conv9 = layers.Conv2D(classes, 1, activation=top_activation, padding='same', kernel_initializer='he_normal')(conv9)

    model = models.Model(inputs, conv9)

    if weights is not None:
        model.load_weights(weights)

    return model
