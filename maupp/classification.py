"""Classification of built-up areas based on Random Forest."""

import rasterio
import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter, median_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_curve, accuracy_score, auc)
from sklearn.model_selection import GroupKFold


from maupp.reference import LEGEND


def _count_features(filenames):
    """Count features in a list of GeoTIFFs."""
    n_features = 0
    for fname in filenames:
        with rasterio.open(fname) as src:
            n_features += src.count
    return n_features


def _count_samples(filenames):
    """Count samples based on a list of GeoTIFFs. Return an error if the
    number of samples is not equal across the files.
    """
    n_samples_list = []
    for fname in filenames:
        with rasterio.open(fname) as src:
            n_samples_list.append(src.width * src.height)
    if len(set(n_samples_list)) == 1:
        return n_samples_list[0]
    else:
        raise ValueError('Number of samples not consistent.')


def transform(filenames, mask=None):
    """Transform input raster data to make it usable by sklearn.

    Parameters
    ----------
    filenames : list of str
        Single-band or multi-band rasters.
    mask : 2d array, optional
        Mask array.

    Returns
    -------
    X : array-like
        Output array of shape (n_samples, n_features).
    """
    use_mask = isinstance(mask, np.ndarray)
    if use_mask:
        n_samples = np.count_nonzero(mask)
    else:
        n_samples = _count_samples(filenames)
    n_features = _count_features(filenames)
    X = np.empty(shape=(n_samples, n_features), dtype='float32')
    i = 0
    for fname in filenames:
        with rasterio.open(fname) as src:
            for j in range(src.count):
                if use_mask:
                    X[:, i] = src.read(j+1)[mask].ravel()
                else:
                    X[:, i] = src.read(j+1).ravel()
                i += 1
    return X


def clustering(src_labels, class_value, **kwargs):
    """KMeans-based spatial clustering. Positive pixels are clustered
    into `k` groups based on their `x` and `y` coordinates.

    Parameters
    ----------
    src_labels : 2d array
        Input training labels.
    class_value : int
        Pixel value corresponding to the class that will be clustered.
    **kwargs
        Passed to MiniBatchKMeans().

    Returns
    -------
    dst_labels : array
        Output raster with cluster labels as pixel values.
    """
    class_mask = src_labels == class_value
    xy = np.where(class_mask)
    nb_samples = xy[0].shape[0]
    X = np.empty(shape=(nb_samples, 2))
    X[:, 0] = xy[0]
    X[:, 1] = xy[1]
    km = MiniBatchKMeans(**kwargs)
    clusters = km.fit_predict(X)
    clusters = clusters + 1
    dst_labels = np.zeros_like(src_labels)
    dst_labels[src_labels == class_value] = clusters
    return dst_labels


def random_choice(src_array, size, random_seed=None):
    """Randomly choose a given amount of pixels from a binary raster.
    Unchosen pixels are assigned the value of 0.

    Parameters
    ----------
    src_array : array-like
        Input binary raster as a 2D NumPy array.
    size : int
        Number of pixels wanted.
    random_seed : int, optional
        Numpy random seed for reproducibility.

    Returns
    -------
    dst_array : array-like
        Output raster.
    """
    if random_seed:
        np.random.seed(random_seed)
    if np.count_nonzero(src_array == 1) == size:
        return src_array
    pixels = np.where(src_array.ravel() == 1)[0]
    selected = np.random.choice(pixels, size=size, replace=False)
    dst_array = np.zeros_like(src_array.ravel())
    dst_array[selected] = 1
    return dst_array.reshape(src_array.shape)


def under_sampling(samples, max_samples=50000, random_seed=None):
    """Perform under-sampling of the majority class (usually, the non-built-up
    class).

    Parameters
    ----------
    samples : 2d array
        Input training labels.
    max_samples : int, optional
        Max. number of samples per class.
    random_seed : int, optional
        Numpy random seed for reproducibility.

    Returns
    -------
    out_samples : 2d array
        Output training labels.
    """
    out_samples = samples.copy()
    n_positive = np.count_nonzero(out_samples == 1)
    n_negative = np.count_nonzero(out_samples == 2)
    if n_positive > n_negative:
        n_samples = min(n_negative, max_samples)
        positive = random_choice(out_samples == 1, n_samples, random_seed)
        out_samples[(out_samples == 1) & (positive != 1)] = 0
    elif n_negative > n_positive:
        n_samples = min(n_positive, max_samples)
        negative = random_choice(out_samples == 2, n_samples, random_seed)
        out_samples[(out_samples == 2) & (negative != 1)] = 0
    if n_positive > max_samples:
        positive = random_choice(out_samples == 1, max_samples, random_seed)
        negative = random_choice(out_samples == 2, max_samples, random_seed)
        out_samples[:, :] = 0
        out_samples[positive] = 1
        out_samples[negative] = 2
    return out_samples


def cross_validation(filenames, training_samples, k=10, max_samples=50000,
                     random_seed=None):
    """Perform K-Fold cross-validation.

    Parameters
    ----------
    filenames : list of str
        Filenames of raster datasets.
    training_samples : 2d array
        Training labels.
    k : int, optional
        Number of folds (default=10).
    max_samples : int, optional
        Max. samples per class.
    random_seed : int, optional
        Numpy and Sklearn random seed for reproducibility.

    Returns
    -------
    scores : ndarray
        F1-scores as an array of length k.
    """
    # Perform under-sampling of the majority class.
    samples = under_sampling(training_samples, max_samples, random_seed)

    # Spatial clustering of both positive and negative training samples
    # into k*3 clusters.
    # This is to avoid spatial autocorrelation, i.e. avoiding neighbooring
    # samples of the same class being in both the training and the testing
    # datasets.
    pos_groups = clustering(samples, 1, n_clusters=k*3,
                            random_state=random_seed)
    neg_groups = clustering(samples, 2, n_clusters=k*3,
                            random_state=random_seed)
    groups = np.maximum(pos_groups, neg_groups)
    groups = groups[groups > 0].ravel()

    # Prediction mask is larger in order to allow post-processing of the
    # classification result with a moving window filter.
    train_mask = samples > 0
    pred_mask = binary_dilation(train_mask, iterations=3)

    # Transform input data
    X = transform(filenames, mask=pred_mask)
    X_train = transform(filenames, mask=train_mask)
    y_train = samples[train_mask].ravel()

    # Perform K-Fold cross-validation using groups to avoid
    # spatial autocorrelation and including a post-processing
    # step based on a 3x3 uniform filter.
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                random_state=random_seed)
    folds = GroupKFold(n_splits=k)
    scores = []
    for train, test in folds.split(X_train, y_train, groups=groups):
        rf.fit(X_train[train], y_train[train])
        proba = rf.predict_proba(X)[:, 0]
        img = np.zeros(shape=samples.shape, dtype=np.float32)
        img[pred_mask] = proba
        img = uniform_filter(img, size=3)
        y_pred = img[train_mask][test] >= 0.5
        y_true = y_train[test] == 1
        scores.append(f1_score(y_true, y_pred))

    return np.array(scores)


def train(filenames, training_labels, max_samples=50000, random_seed=None,
          **kwargs):
    """Train a Random Forest model.

    Parameters
    ----------
    filenames : list of str
        Filenames of raster datasets.
    training_labels : 2d array
        Training labels (1=BuiltUp, 2=NonBuiltUp).
    max_samples : int, optional
        Max. number of training samples per class.
    random_seed : int, optional
        Numpy and Sklearn random seed for reproducibility.
    **kwagrs
        Passed to RandomForestClassifier().

    Returns
    -------
    model : classifier
        Sklearn classifier object (fitted).
    """
    samples = under_sampling(training_labels, max_samples, random_seed)
    X_train = transform(filenames, mask=samples > 0)
    y_train = samples[samples > 0].ravel()
    rf = RandomForestClassifier(random_state=random_seed, **kwargs)
    rf.fit(X_train, y_train)
    return rf


def predict(model, X, width, height):
    """Prediction with a fitted model.

    Parameters
    ----------
    model : classifier
        Sklearn classifier (fitted).
    X : array
        Input data.
    width : int
        Raster width.
    height : int
        Raster height.

    Returns
    -------
    proba : 2d array
        RF probabilistic output of shape (height, width).
    """
    proba = model.predict_proba(X)[:, 0]
    proba = proba.reshape((height, width))
    return proba


def validate(proba, validation_samples, threshold=0.5):
    """Compute validation metrics from an independent validation dataset.

    Parameters
    ----------
    proba : 2d array
        RF probabilities for built-up class.
    validation_samples : 2d array
        Validation dataset. Legend: 0=NA, 1=BuiltUp, 2=BareSoil,
        3=LowVegetation, 4=HighVegetation, 5=OtherNonBuiltUp.
    threshold : float, optional
        Threshold applied on the probabilities for binarization.

    Returns
    -------
    metrics
    """
    metrics = {}
    mask = validation_samples > 0

    # Binary metrics
    y_true = validation_samples[mask].ravel() == 1
    y_pred = proba[mask].ravel() >= threshold
    metrics['f1_score'] = f1_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, proba[mask], pos_label=1)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    metrics['auc'] = auc(fpr, tpr)

    # Per landcover accuracy
    for land, value in LEGEND.items():
        # ignore "other non-built-up" land cover
        if value == 5 or not value in np.unique(validation_samples):
            continue
        mask = validation_samples == value
        y_true = validation_samples[mask].ravel() == 1
        y_pred = proba[mask].ravel() >= threshold
        metrics['{}_accuracy'.format(land)] = accuracy_score(y_true, y_pred)

    return metrics


def post_processing(probabilities, slope, water, proba_threshold=0.9,
                    max_slope=40, filter_size=5):
    """Post-process RF probabilistic output to a binary built-up
    areas raster.

    Parameters
    ----------
    probabilities : 2d array
        RF probabilistic output.
    slope : 2d array
        Slope in percents.
    water : 2d array
        Water mask.
    proba_threshold : float, optional
        Threshold on RF probabilistic output (default=0.9).
    max_slope : int, optional
        Max. slope in percents (default=40).
    filter_size : int, optional
        Size of the majority filter (default=5).

    Returns
    -------
    classes : 2d array
        Binary raster (built-up vs. non-built-up areas).
    """
    classes = (probabilities >= proba_threshold).astype(np.uint8)
    classes[water] = 0
    classes[slope >= max_slope] = 0
    classes = median_filter(classes, size=filter_size)
    return classes
