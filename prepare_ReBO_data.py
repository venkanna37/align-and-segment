"""
Download ReBO data zip file, unzip it
Give ReBO folder as input to this file
"""
import os
import skimage
import argparse
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


def prepare_ReBO_data(indir, set_name, outdir):
    image_dir = os.path.join(indir, set_name, f'isra_{set_name}/')
    annotation_path = os.path.join(indir, set_name, f'ReBO_{set_name}.json')

    # create label folders
    if not os.path.exists(os.path.join(outdir, f'label_{set_name}')):
        os.makedirs(os.path.join(outdir, f'label_{set_name}'))
    if not os.path.exists(os.path.join(outdir, f'osm_label_{set_name}')):
        os.makedirs(os.path.join(outdir, f'osm_label_{set_name}'))

    # data
    coco_ = COCO(annotation_path)
    images = coco_.imgs
    anns = coco_.anns
    image_ids = coco_.getImgIds()

    # iterate through images and create labels
    for image_id in tqdm(image_ids):
        # get image and respective anns
        image_info = coco_.loadImgs(image_id)[0]
        height, width = image_info['height'], image_info['width']
        img_file = image_info['file_name']
        annotation_ids = coco_.getAnnIds(imgIds=image_id)
        coco_annotations = coco_.loadAnns(annotation_ids)

        # create a roof and labels mask
        roof_mask = np.zeros((3, height, width), dtype=np.uint8)
        osm_mask = np.zeros((3, height, width), dtype=np.uint8)

        for ann in coco_annotations:
            # roof mask
            ann_copy = ann.copy()
            ann_copy['segmentation'] = [ann['roof_mask']]
            rle = coco_.annToRLE(ann_copy)
            mask = maskUtils.decode(rle)
            roof_mask = np.logical_or(roof_mask, mask).astype(np.uint8)

            # osm mask
            ann_copy = ann.copy()
            ann_copy['segmentation'] = [ann['osm_mask']]
            rle = coco_.annToRLE(ann_copy)
            mask = maskUtils.decode(rle)
            osm_mask = np.logical_or(osm_mask, mask).astype(np.uint8)

        # save masks using skimage
        filename = os.path.join(outdir, f'label_{set_name}', img_file)
        skimage.io.imsave(filename, roof_mask.transpose(1,2,0))
        filename = os.path.join(outdir, f'osm_label_{set_name}', img_file)
        skimage.io.imsave(filename, osm_mask.transpose(1,2,0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parameters
    parser.add_argument("--indir", type=str,
                        help='Input directory to ReBO data', default='./sample_data/ReBO/')
    parser.add_argument("--setname", type=str,
                        help='Set name from ReBO data (train or test)', default='test')
    parser.add_argument("--outdir", type=str,
                        help='Output directory to save labels. Save where you keep images)',
                        default='./sample_data/ReBO/')

    args = parser.parse_args()
    prepare_ReBO_data(args.indir, args.setname, args.outdir)


