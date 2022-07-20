import argparse
import json
import os
from tqdm import tqdm
import numpy as np
import cv2
import pickle

def loadPickleData(fName):
    with open(fName, 'rb') as f:
        try:
            pickData = pickle.load(f, encoding='latin1')
        except:
            pickData = pickle.load(f)

    return pickData

def main(base_path, pred_out_path, pred_func, set_name=None):
    """
        Main eval loop: Iterates over all evaluation samples and saves the corresponding predictions.
    """
    # default value
    if set_name is None:
        set_name = 'evaluation'

    # init output containers
    pose_pred_list = list()

    # read list of evaluation files
    img_dir = os.path.join(base_path, set_name, 'images')
    with open(os.path.join(base_path, 'evaluation.txt'), 'r') as f:
        file_list = f.readlines()
    file_list = [f.strip() for f in file_list]


    anno_dir = os.path.join(base_path, set_name, 'annotations')


    # iterate over the dataset once
    for idx in tqdm(range(len(file_list))):
        file_id = file_list[idx]

        # load input image
        img = cv2.imread(os.path.join(img_dir, file_id + '.png'))
        aux = loadPickleData(os.path.join(anno_dir, file_id + '.pickle'))

        # use some algorithm for prediction
        obj_rot_pred, obj_trans_pred = pred_func(
            img, {'mano_pose': aux['mano_pose'],
                  'mano_shape': aux['mano_shape']}
        )

        assert len(obj_rot_pred.shape) == 1 and obj_rot_pred.shape[
            0] == 3, 'Obj rotation should be 3-dimensional vector'
        assert len(obj_trans_pred.shape) == 1 and obj_trans_pred.shape[
            0] == 3, 'Obj translation should be 3-dimensional vector'

        pose_pred_list.append(np.concatenate([obj_rot_pred, obj_trans_pred], axis=0))

    # dump results
    dump(pred_out_path, pose_pred_list)


def dump(pred_out_path, pose_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    pose_pred_list = [x.tolist() for x in pose_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                pose_pred_list
            ], fo)
    print('Dumped %d poses to %s' % (len(pose_pred_list), pred_out_path))


def pred_template(img, aux_info):
    """ Predict object rotation and hand-relative translation.
        img: (256, 256, 3) RGB image.
        aux_info: dictionary containing hand pose
    """
    # TODO: Put your algorithm here, object rotation and translation wrt hand root joint
    obj_rot_pred = np.zeros((3,))
    obj_trans_pred = np.zeros((3,))
    return obj_rot_pred, obj_trans_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('--dataset_dir', type=str,
                        help='Path to where the HO3D dataset is located.')
    parser.add_argument('--out', type=str, default='pred.json',
                        help='File to save the predictions.')
    args = parser.parse_args()

    # call with a predictor function
    main(
        args.dataset_dir,
        args.out,
        pred_func=pred_template,
        set_name='evaluation'
    )

