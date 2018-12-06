from keras import backend


def image_binary_accuracy(y_true, y_pred):
    square_wise_binary_accuracy = backend.equal(y_true, backend.round(y_pred))
    # backend.mean instead of backend.sum gives the same effect
    # as just using standard Keras binary_accuracy
    return backend.sum(square_wise_binary_accuracy, axis=(-1, -2, -3))