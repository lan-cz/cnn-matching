import numpy as np


def plot_matches(ax, image1, image2, keypoints1, keypoints2, matches,
                 keypoints_color='r', matches_color=None, plot_matche_points=True, matchline = True, matchlinewidth = 0.5,
                 alignment='horizontal'):
    """Plot matched features.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matches and image are drawn in this ax.
    image1 : (N, M [, 3]) array
        First grayscale or color image.
    image2 : (N, M [, 3]) array
        Second grayscale or color image.
    keypoints1 : (K1, 2) array
        First keypoint coordinates as ``(row, col)``.
    keypoints2 : (K2, 2) array
        Second keypoint coordinates as ``(row, col)``.
    matches : (Q, 2) array
        Indices of corresponding matches in first and second set of
        descriptors, where ``matches[:, 0]`` denote the indices in the first
        and ``matches[:, 1]`` the indices in the second set of descriptors.
    keypoints_color : matplotlib color, optional
        Color for keypoint locations.
    matches_color : matplotlib color, optional
        Color for lines which connect keypoint matches. By default the
        color is chosen randomly.
    only_matches : bool, optional
        Whether to only plot matches and not plot the keypoint locations.
    alignment : {'horizontal', 'vertical'}, optional
        Whether to show images side by side, ``'horizontal'``, or one above
        the other, ``'vertical'``.

    """

    #image1 = img_as_float(image1)
    #image2 = img_as_float(image2)

    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        #new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1 = np.full(new_shape1, 255)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        #new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2 = np.full(new_shape2, 255)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    offset = np.array(image1.shape)
    if alignment == 'horizontal':
        if image2.ndim == 3:
            blank = np.full((new_shape2[0], 10, 3), 255)
        if image1.ndim == 2 or image2.ndim == 2:
            blank = np.full((new_shape2[0], 10), 255)
        image = np.concatenate([image1, blank, image2], axis=1)
        offset[0] = 0
        offset[1] += 10
    elif alignment == 'vertical':
        if image2.ndim == 3 :
            blank = np.full(10,(new_shape2[1], 3), 255)
        if image1.ndim == 2:
            blank = np.full(10,(new_shape2[1]), 255)
        image = np.concatenate([image1, blank, image2], axis=0)
        offset[1] = 0
        offset[0] += 10
    else:
        mesg = ("plot_matches accepts either 'horizontal' or 'vertical' for "
                "alignment, but '{}' was given. See "
                "https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.plot_matches "  # noqa
                "for details.").format(alignment)
        raise ValueError(mesg)


    if plot_matche_points:
        ax.scatter(keypoints1[:, 0], keypoints1[:, 1],
                   facecolors='none', edgecolors=keypoints_color, marker = '.')
        ax.scatter(keypoints2[:, 0] + offset[1], keypoints2[:, 1] + offset[0],
                   facecolors='none', edgecolors=keypoints_color, marker = '.')

    ax.imshow(image, interpolation='nearest', cmap='gray')
    ax.axis((0, image1.shape[1] + offset[1], image1.shape[0] + offset[0], 0))

    if matchline == True:
        for i in range(matches.shape[0]):
            idx1 = matches[i, 0]
            idx2 = matches[i, 1]

            if matches_color is None:
                color = np.random.rand(3)
            else:
                color = matches_color

            ax.plot((keypoints1[idx1, 0], keypoints2[idx2, 0] + offset[1]),
                    (keypoints1[idx1, 1], keypoints2[idx2, 1] + offset[0]),
                    '-', color=color, linewidth=matchlinewidth, marker='+', markersize=8)


def plot_matches2(ax, image1, image2, keypoints1, keypoints2,
                 keypoints_color='r', matches_color=None, plot_matche_points=True, matchline = True, matchlinewidth = 0.5,
                 alignment='horizontal'):


    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        #new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1 = np.full(new_shape1, 255)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        #new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2 = np.full(new_shape2, 255)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    offset = np.array(image1.shape)
    if alignment == 'horizontal':
        if image2.ndim == 3:
            blank = np.full((new_shape2[0], 10, 3), 255)
        if image1.ndim == 2 or image2.ndim == 2:
            blank = np.full((new_shape2[0], 10), 255)
        image = np.concatenate([image1, blank, image2], axis=1)
        offset[0] = 0
        offset[1] += 10
    elif alignment == 'vertical':
        if image2.ndim == 3 :
            blank = np.full(10,(new_shape2[1], 3), 255)
        if image1.ndim == 2:
            blank = np.full(10,(new_shape2[1]), 255)
        image = np.concatenate([image1, blank, image2], axis=0)
        offset[1] = 0
        offset[0] += 10
    else:
        mesg = ("plot_matches accepts either 'horizontal' or 'vertical' for "
                "alignment, but '{}' was given. See "
                "https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.plot_matches "  # noqa
                "for details.").format(alignment)
        raise ValueError(mesg)


    if plot_matche_points:
        ax.scatter(keypoints1[:, 0], keypoints1[:, 1],
                   facecolors='none', edgecolors=keypoints_color, marker = '.')
        ax.scatter(keypoints2[:, 0] + offset[1], keypoints2[:, 1] + offset[0],
                   facecolors='none', edgecolors=keypoints_color, marker = '.')

    ax.imshow(image, interpolation='nearest', cmap='gray')
    ax.axis((0, image1.shape[1] + offset[1], image1.shape[0] + offset[0], 0))

    if matchline == True:
        for i in range(keypoints1.shape[0]):

            if matches_color is None:
                color = np.random.rand(3)
            else:
                color = matches_color

            ax.plot((keypoints1[i, 0], keypoints2[i, 0] + offset[1]),
                    (keypoints1[i, 1], keypoints2[i, 1] + offset[0]),
                    '-', color=color, linewidth=matchlinewidth, marker='+', markersize=5)



