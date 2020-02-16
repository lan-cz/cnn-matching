import torch
from lib.model import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale
import scipy
import scipy.io
import scipy.misc
import numpy as np

use_cuda = torch.cuda.is_available()

# Creating CNN model
model = D2Net(
    model_file="models/d2_tf.pth",
    use_relu=True,
    use_cuda=use_cuda
)
device = torch.device("cuda:0" if use_cuda else "cpu")

multiscale = True
max_edge = 2500
max_sum_edges = 5000
# de-net feature extract function
def cnn_feature_extract(image,scales=[.25, 0.50, 1.0], nfeatures = 1000):
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    if max(resized_image.shape) > max_edge:
        resized_image = scipy.misc.imresize(
            resized_image,
            max_edge / max(resized_image.shape)
        ).astype('float')
    if sum(resized_image.shape[: 2]) > max_sum_edges:
        resized_image = scipy.misc.imresize(
            resized_image,
            max_sum_edges / sum(resized_image.shape[: 2])
        ).astype('float')

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(
        resized_image,
        preprocessing="torch"
    )
    with torch.no_grad():
        if multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]

    if nfeatures != -1:
        #根据scores排序
        scores2 = np.array([scores]).T
        res = np.hstack((scores2, keypoints))
        res = res[np.lexsort(-res[:, ::-1].T)]

        res = np.hstack((res, descriptors))
        #取前几个
        scores = res[0:nfeatures, 0].copy()
        keypoints = res[0:nfeatures, 1:4].copy()
        descriptors = res[0:nfeatures, 4:].copy()
        del res
    return keypoints, scores, descriptors
