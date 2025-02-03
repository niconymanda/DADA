import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

try:
    from StringIO import StringIO
except:
    from io import BytesIO as StringIO

import scipy.misc


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()
        try:
            self.writer.add_scalar(tag, value, global_step=step)
        except:
            print(tag, value)
            value = np.array(value)
            self.writer.add_scalar(tag, value, global_step=step)

    def image_summary(self, tag, images, step):
        img_summaries = []
        for i, img in enumerate(images):
            s = StringIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = {
                "height": img.shape[0],
                "width": img.shape[1],
                "data": s.getvalue(),
            }
            # Create a Summary value
            img_summaries.append(img_sum)

        self.writer.add_images(tag, img_summaries, global_step=step)

    def histo_summary(self, tag, values, step, bins=1000):
        counts, bin_edges = np.histogram(values.cpu().numpy(), bins=bins)

        hist = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "num": int(np.prod(values.shape)),
            "sum": float(np.sum(values)),
            "sum_squares": float(np.sum(values**2)),
            "bucket_limit": bin_edges[1:].tolist(),
            "bucket": counts.tolist(),
        }

        self.writer.add_histogram(tag, hist, global_step=step)
