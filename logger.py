from io import BytesIO

import scipy.misc
import tensorflow as tf

try:
    FileWriter = tf.compat.v1.summary.FileWriter
    Summary = tf.compat.v1.Summary
except:
    FileWriter = tf.summary.FileWriter
    Summary = tf.Summary


class Logger(object):

    def __init__(self, log_dir):
        self.writer = FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def image_summary(self, tag, image, step):
        s = BytesIO()
        scipy.misc.toimage(image).save(s, format="png")

        # Create an Image object
        img_sum = Summary.Image(
            encoded_image_string=s.getvalue(),
            height=image.shape[0],
            width=image.shape[1],
        )

        # Create and write Summary
        summary = Summary(value=[Summary.Value(tag=tag, image=img_sum)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def image_list_summary(self, tag, images, step):
        if len(images) == 0:
            return
        img_summaries = []
        for i, img in enumerate(images):
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = Summary.Image(
                encoded_image_string=s.getvalue(),
                height=img.shape[0],
                width=img.shape[1],
            )

            # Create a Summary value
            img_summaries.append(
                Summary.Value(tag="{}/{}".format(tag, i), image=img_sum)
            )

        # Create and write Summary
        summary = Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()
