import os
import cv2
import numpy as np
from keras import callbacks


class TestPredictor(callbacks.Callback):
    """
    Save test predictions after every epoch.
    """
    def __init__(self, test_generator, path, background_as_class=False):
        super(TestPredictor, self).__init__()
        self.test_generator = test_generator
        self.dst_path = path
        self.background_as_class = background_as_class

    def on_epoch_end(self, epoch, logs=None):
        results = self.model.predict_generator(
            self.test_generator,
            steps=self.test_generator.samples / self.test_generator.batch_size,
            verbose=0
        )
        if self.background_as_class is True:
            segmentations = np.array(results[..., 1:] * 255., dtype=np.uint8)
        else:
            segmentations = np.array(results * 255., dtype=np.uint8)
        dst_dir = os.path.join(self.dst_path, "epoch_" + str(epoch + 1))
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        for filename, segmentation in zip(self.test_generator.filenames, segmentations):
            image_name = "out-" + os.path.split(filename)[-1]
            cv2.imwrite(os.path.join(dst_dir, image_name), segmentation)
        print('\nEpoch %05d: saving test results to %s' % (epoch + 1, dst_dir))
