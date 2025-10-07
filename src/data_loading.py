import numpy as np
import os as os

def load_data(root, reduce=2, normalization="per_image", eps=1e-12):
    """
    Load ORL or Extended YaleB dataset to numpy array.

    Args:
        root: path to dataset.
        reduce: scale factor for zooming out images.
        normalisation: context for which to apply normalisation
        epis:
    """
    images, labels = [], []
    height, width = 0, 0

    for i, person in enumerate(sorted(os.listdir(root))):

        if not os.path.isdir(os.path.join(root, person)):
            continue

        for fname in os.listdir(os.path.join(root, person)):

            # Remove background images in Extended YaleB dataset.
            if fname.endswith('Ambient.pgm'):
                continue

            if not fname.endswith('.pgm'):
                continue

            # load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L') # grey image.


            # reduce computation complexity.
            img = img.resize([s//reduce for s in img.size])

            # save the height/width after resizing
            if (height == 0 or width == 0):
              height = img.size[1]
              width = img.size[0]

            img = np.asarray(img, dtype=np.float32)

            # normalzation per image
            if normalization == "per_image":
              img = (img - img.min()) / (img.max() - img.min() + eps)

            img = img.reshape((-1,1))

            # collect data and label.
            images.append(img)
            labels.append(i)

    # concate all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)

    # normalize globally
    if normalization == "global":
      images = images.astype(np.float32)
      images = images / 255.0 # normalizing them


    return images, labels, height, width