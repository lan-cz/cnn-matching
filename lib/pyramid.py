import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.exceptions import EmptyTensorError
from lib.utils import interpolate_dense_features, upscale_positions
import numpy as np
import matplotlib.pyplot as plt

def process_multiscale(image, model, scales=[.25, 0.50, 1.0]):
    b, _, h_init, w_init = image.size()
    device = image.device
    assert(b == 1)

    all_keypoints = torch.zeros([3, 0])
    all_descriptors = torch.zeros([
        model.dense_feature_extraction.num_channels, 0
    ])
    all_scores = torch.zeros(0)

    previous_dense_features = None
    banned = None
    for idx, scale in enumerate(scales):
        current_image = F.interpolate(
            image, scale_factor=scale,
            mode='bilinear', align_corners=True
        )
        _, _, h_level, w_level = current_image.size()

        dense_features = model.dense_feature_extraction(current_image)
        del current_image

        _, _, h, w = dense_features.size()

        # Sum the feature maps.
        if previous_dense_features is not None:
            dense_features += F.interpolate(
                previous_dense_features, size=[h, w],
                mode='bilinear', align_corners=True
            )
            del previous_dense_features

        # Recover detections.
        detections = model.detection(dense_features)
        if banned is not None:
            banned = F.interpolate(banned.float(), size=[h, w]).bool()
            detections = torch.min(detections, ~banned)
            banned = torch.max(
                torch.max(detections, dim=1)[0].unsqueeze(1), banned
            )
        else:
            banned = torch.max(detections, dim=1)[0].unsqueeze(1)
        fmap_pos = torch.nonzero(detections[0].cpu()).t()
        del detections
        # vis

        """
        fig = plt.figure()

        #plt.subplot(2, 1, 2)
        #plt.imshow(img_out)
        for i in range(25):
            vismap = dense_features[0,i,::,::]
            #
            vismap = vismap.cpu()

            #use sigmod to [0,1]
            vismap= 1.0/(1+np.exp(-1*vismap))

            # to [0,255]
            vismap=np.round(vismap*255)
            vismap=vismap.data.numpy()
            plt.subplot(5, 5, i+1)
            plt.axis('off')
            plt.imshow(vismap)
            filename = '/home/asky/featuremap/CH%d.jpg'% (i)

            #cv2.imwrite(filename,vismap)

        plt.tight_layout()
        fig.show()
        """
        # Recover displacements.
        displacements = model.localization(dense_features)[0].cpu()
        displacements_i = displacements[
            0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]
        displacements_j = displacements[
            1, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]
        del displacements

        mask = torch.min(
            torch.abs(displacements_i) < 0.5,
            torch.abs(displacements_j) < 0.5
        )
        fmap_pos = fmap_pos[:, mask]
        valid_displacements = torch.stack([
            displacements_i[mask],
            displacements_j[mask]
        ], dim=0)
        del mask, displacements_i, displacements_j

        fmap_keypoints = fmap_pos[1 :, :].float() + valid_displacements
        del valid_displacements

        try:
            raw_descriptors, _, ids = interpolate_dense_features(
                fmap_keypoints.to(device),
                dense_features[0]
            )
        except EmptyTensorError:
            continue
        fmap_pos = fmap_pos[:, ids]
        fmap_keypoints = fmap_keypoints[:, ids]
        del ids

        keypoints = upscale_positions(fmap_keypoints, scaling_steps=2)
        del fmap_keypoints

        descriptors = F.normalize(raw_descriptors, dim=0).cpu()
        del raw_descriptors

        keypoints[0, :] *= h_init / h_level
        keypoints[1, :] *= w_init / w_level

        fmap_pos = fmap_pos.cpu()
        keypoints = keypoints.cpu()

        keypoints = torch.cat([
            keypoints,
            torch.ones([1, keypoints.size(1)]) * 1 / scale,
        ], dim=0)

        scores = dense_features[
            0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ].cpu() / (idx + 1)
        del fmap_pos

        all_keypoints = torch.cat([all_keypoints, keypoints], dim=1)
        all_descriptors = torch.cat([all_descriptors, descriptors], dim=1)
        all_scores = torch.cat([all_scores, scores], dim=0)
        del keypoints, descriptors

        previous_dense_features = dense_features
        del dense_features
    del previous_dense_features, banned

    keypoints = all_keypoints.t().numpy()
    del all_keypoints
    scores = all_scores.numpy()
    del all_scores
    descriptors = all_descriptors.t().numpy()
    del all_descriptors
    return keypoints, scores, descriptors
