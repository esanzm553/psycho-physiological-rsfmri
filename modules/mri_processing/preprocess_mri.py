import os
import logging
from nilearn.image import mean_img
import nibabel as nib
from nilearn.masking import compute_background_mask, compute_multi_epi_mask, compute_multi_brain_mask
from scipy.ndimage import binary_dilation

def compute_mean_images(data_imgs, output_folder, overwrite=False):
    for im in data_imgs:
        fname = os.path.basename(im.get_filename())
        output_fname = os.path.join(output_folder, "mean_img_"+fname)
        if (not os.path.exists(output_fname)) or overwrite:
            mean_im = mean_img(im)
            nib.save(mean_im, output_fname)
            logging.info(f"Created mean image {output_fname}")
            im.uncache()
        else:
            logging.info(f"Loaded mean image {output_fname}")

def compute_mask(data_imgs, output_path, cache=None):
    mask_file = os.path.join(output_path, "brain_mask.nii.gz")
    try:
        # Mask Creation
        if not mask_file:
            mean_images_folder = os.path.join(output_path, "mean_images")
            logging.info("Computing mean images...")
            logging.info(f"Mean images stored in {mean_images_folder}")

            logging.info("Creating mask..")
            mask_file = os.path.join(output_path, "canica_mask.nii.gz")
        if not os.path.exists(mask_file):
            mask = compute_multi_brain_mask(data_imgs, memory=cache)
            mask_data = binary_dilation(mask.get_fdata(), iterations=5)
            mask = nib.Nifti1Image(mask_data, mask.affine, mask.header)
            nib.save(mask, mask_file)
            logging.info(f"Mask created in {mask_file}")
        else:
            mask = nib.load(mask_file)
            logging.info(f"Mask loaded from {mask_file}")
    except Exception as e:
        logging.error(f"Error creating the mask: {str(e)}")
        raise e
    return mask
