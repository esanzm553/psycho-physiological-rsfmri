from nilearn.image import threshold_img
from nilearn.masking import apply_mask, unmask
from scipy.stats import linregress, pearsonr
import numpy as np

def dual_regression(func_path, ica, mask_path):
    # Get the projection of the components in the subject space
    func_projected = ica.transform([func_path])[0]

    # Calculate the regression of each component over each voxel to get the spatial dimension
    masked_data = apply_mask(func_path, mask_path, smoothing_fwhm=ica.smoothing_fwhm)    
    weights_maps = np.zeros_like(ica.components_)
    correlations = []
    for c in range(ica.n_components):
        regressor = func_projected[:,c]
        for i in range(masked_data.shape[1]):
            reg = linregress(regressor,masked_data[:,i])
            weights_maps[c,i] = reg.slope
        # Calculate the correlation between the spatial map of the subject and the population components (GoF metric)
        correlations.append(pearsonr(ica.components_[c], weights_maps[c]).statistic)
    # Return the unmasked volume of the components for QA purposes
    unmasked = unmask(weights_maps, mask_path)
    unmasked = threshold_img(unmasked, "80%")
    return unmasked, correlations


