from __future__ import print_function, unicode_literals

import argparse
import json
import os
import pip
import sys
import math

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        from pip._internal.main import main as pipmain
        pipmain(['install', package])


try:
    import numpy as np
except:
    install('numpy')
    import numpy as np

def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def rodrigues_vec_to_rotation_mat(rodrigues_vec):
    theta = np.linalg.norm(rodrigues_vec)
    if theta < sys.float_info.epsilon:
        rotation_mat = np.eye(3, dtype=float)
    else:
        r = rodrigues_vec / theta
        I = np.eye(3, dtype=float)
        r_rT = np.array([
            [r[0]*r[0], r[0]*r[1], r[0]*r[2]],
            [r[1]*r[0], r[1]*r[1], r[1]*r[2]],
            [r[2]*r[0], r[2]*r[1], r[2]*r[2]]
        ])
        r_cross = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        rotation_mat = math.cos(theta) * I + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross
    return rotation_mat

def _search_pred_file(pred_path, pred_file_name):
    """ Tries to select the prediction file. Useful, in case people deviate from the canonical prediction file name. """
    pred_file = os.path.join(pred_path, pred_file_name)
    if os.path.exists(pred_file):
        # if the given prediction file exists we are happy
        return pred_file

    print('Predition file "%s" was NOT found' % pred_file_name)

    # search for a file to use
    print('Trying to locate the prediction file automatically ...')
    files = [os.path.join(pred_path, x) for x in os.listdir(pred_path) if x.endswith('.json')]
    if len(files) == 1:
        pred_file_name = files[0]
        print('Found file "%s"' % pred_file_name)
        return pred_file_name
    else:
        print('Found %d candidate files for evaluation' % len(files))
        raise Exception('Giving up, because its not clear which file to evaluate.')


def main(gt_path, pred_path, output_dir, pred_file_name=None, set_name=None):
    if pred_file_name is None:
        pred_file_name = 'pred.json'
    if set_name is None:
        set_name = 'evaluation'

    # load eval annotations
    pose_list = json_load(os.path.join(gt_path, '%s_pose.json' % set_name))
    obj_names_list = json_load(os.path.join(gt_path, '%s_obj_names.json' % set_name))
    obj_verts_dict = json_load(os.path.join(gt_path, '%s_obj_verts.json' % set_name))

    # load predicted values
    pred_file = _search_pred_file(pred_path, pred_file_name)
    print('Loading predictions from %s' % pred_file)
    with open(pred_file, 'r') as fi:
        pred = json.load(fi)[0]

    assert len(pred) == len(pose_list), 'Expected format mismatch.'

    num_obj_samples = len(pred)
    all_obj_mssd_dict = {}

    pose_list = np.array(pose_list)
    pred = np.array(pred)

    for ii in range(num_obj_samples):

        # Get the rotation matrices for z and y axis symmetric objects
        z_rot_dir = rodrigues_vec_to_rotation_mat(pose_list[ii][:3])[:3, 2] * np.pi  # N x 3
        y_rot_dir = rodrigues_vec_to_rotation_mat(pose_list[ii][:3])[:3, 1] * np.pi  # N x 3
        flipped_obj_rot_z = np.matmul(rodrigues_vec_to_rotation_mat(z_rot_dir),
                                      rodrigues_vec_to_rotation_mat(pose_list[ii][:3]))  # 3 x 3 # flipped rot
        flipped_obj_rot_y = np.matmul(rodrigues_vec_to_rotation_mat(y_rot_dir),
                                      rodrigues_vec_to_rotation_mat(pose_list[ii][:3]))  # 3 x 3 # flipped rot
        flipped_obj_rot_zy = np.matmul(rodrigues_vec_to_rotation_mat(z_rot_dir),
                                       flipped_obj_rot_y)  # 3 x 3 # flipped rot

        obj_vert_rest = np.array(obj_verts_dict[obj_names_list[ii]])
        # Transform vertices using the gt pose
        obj_vert_gt = np.matmul(obj_vert_rest, rodrigues_vec_to_rotation_mat(pose_list[ii][:3]).T) + pose_list[ii][3:]  # N x 8 x 3
        # Transform vertices using the predicted pose
        obj_vert_pred = np.matmul(obj_vert_rest, rodrigues_vec_to_rotation_mat(pred[ii][:3]).T) + pred[ii][3:]  # N x 8 x 3

        if obj_names_list[ii] in ['025_mug', '019_pitcher_base', '011_banana']:
            # Cylindrical objects (angle of symmetry = inf)

            # Take rotations in steps of 5degrees along z-axis
            z_rots = np.arange(-np.pi, np.pi, 5 * np.pi / 180)
            obj_err = np.inf
            for z in z_rots:
                rot_dir_curr = rodrigues_vec_to_rotation_mat(pose_list[ii][:3])[:3, 2] * z  # N x 3
                # Rotation matrix for the current rotation
                z_obj_rot = np.matmul(rodrigues_vec_to_rotation_mat(rot_dir_curr),
                                      rodrigues_vec_to_rotation_mat(pose_list[ii][:3]))  # 3 x 3 # flipped rot
                # Transform the vertices
                obj_vert_z_rot_gt = np.matmul(obj_vert_rest, z_obj_rot.T) + pose_list[ii][3:]
                # MSSD error
                obj_err = min(obj_err, np.max(np.linalg.norm(obj_vert_z_rot_gt - obj_vert_pred, axis=1)))

        elif obj_names_list[ii] in ['006_mustard_bottle', '021_bleach_cleanser']:
            # Z axis symmetrical, Angle of symmetry = 180 degrees

            # Transform the vertices
            obj_vert_flipped_gt = np.matmul(obj_vert_rest, flipped_obj_rot_z.T) + pose_list[ii][3:]

            # MSSD error
            obj_err = min(np.max(np.linalg.norm(obj_vert_gt - obj_vert_pred, axis=1)),
                          np.max(np.linalg.norm(obj_vert_flipped_gt - obj_vert_pred, axis=1)))

        elif obj_names_list[ii] in ['003_cracker_box', '004_sugar_box', '010_potted_meat_can']:
            # Cuboid objects, Angle of symmetry = 180 degrees

            # Transform the vertices (z-axis symmetry)
            obj_vert_flipped_gt = np.matmul(obj_vert_rest, flipped_obj_rot_z.T) + pose_list[ii][3:]
            # MSSD error
            obj_err = min(np.max(np.linalg.norm(obj_vert_gt - obj_vert_pred, axis=1)),
                          np.max(np.linalg.norm(obj_vert_flipped_gt - obj_vert_pred, axis=1)))

            # Transform the vertices (y-axis symmetry)
            obj_vert_flipped_gt = np.matmul(obj_vert_rest, flipped_obj_rot_y.T) + pose_list[ii][3:]
            # MSSD error
            obj_err = min(obj_err, np.max(np.linalg.norm(obj_vert_flipped_gt - obj_vert_pred, axis=1)))

            # Transform the vertices (z-axis then y-axis symmetry)
            obj_vert_flipped_gt = np.matmul(obj_vert_rest, flipped_obj_rot_zy.T) + pose_list[ii][3:]
            # MSSD error
            obj_err = min(obj_err, np.max(np.linalg.norm(obj_vert_flipped_gt - obj_vert_pred, axis=1)))

        elif obj_names_list[ii] in ['037_scissors']:
            # Custom axis symmetrical, Angle of symmetry = 180 degrees

            # Get the rotation matrix for symmetricity along the axis of the scissor
            obj_rot_mat_gt = rodrigues_vec_to_rotation_mat(pose_list[ii][:3])
            rot_axis = 0.0366*obj_rot_mat_gt[:3,0] + 0.1271*obj_rot_mat_gt[:3,1]
            rot_axis = rot_axis/np.linalg.norm(rot_axis)
            rot_axis = rot_axis*np.pi
            obj_rot_mat_flipped_gt = np.matmul(rodrigues_vec_to_rotation_mat(rot_axis),
                                    obj_rot_mat_gt)  # 3 x 3 # flipped rot

            # Transform the vertices
            obj_vert_flipped_gt = np.matmul(obj_vert_rest, obj_rot_mat_flipped_gt.T) + pose_list[ii][3:]
            # MSSD error
            obj_err = min(np.max(np.linalg.norm(obj_vert_gt - obj_vert_pred, axis=1)),
                          np.max(np.linalg.norm(obj_vert_flipped_gt - obj_vert_pred, axis=1)))

        elif obj_names_list[ii] in ['035_power_drill']:
            # Not symmetrical

            # MSSD error
            obj_err = np.max(np.linalg.norm(obj_vert_gt - obj_vert_pred, axis=1))

        else:
            raise NotImplementedError

        metric_mssd = obj_err

        # Save them
        if obj_names_list[ii] not in all_obj_mssd_dict.keys():
            all_obj_mssd_dict[obj_names_list[ii]] = []

        all_obj_mssd_dict[obj_names_list[ii]].append(metric_mssd)

    # Print them
    all_obj_mssd_list = []
    ind_obj_mssd_dict = {}
    for id in all_obj_mssd_dict.keys():
        ind_obj_mssd_dict[id] = np.mean(np.array(all_obj_mssd_dict[id]))*100
        print('%s MSSD (cnt %d) = %f cms' % (
            id, len(all_obj_mssd_dict[id]), ind_obj_mssd_dict[id]))
        all_obj_mssd_list = all_obj_mssd_list + all_obj_mssd_dict[id]

    all_obj_mean_mssd = np.mean(np.array(all_obj_mssd_list))*100
    print('All objects mean MSSD = %f cms' % (all_obj_mean_mssd))


    # Dump results
    score_path = os.path.join(output_dir, 'scores.txt')
    with open(score_path, 'w') as fo:
        fo.write('025_mug_MSSD: %f\n' % ind_obj_mssd_dict['025_mug'])
        fo.write('019_pitcher_base_MSSD: %f\n' % ind_obj_mssd_dict['019_pitcher_base'])
        fo.write('011_banana_MSSD: %f\n' % ind_obj_mssd_dict['011_banana'])
        fo.write('006_mustard_bottle_MSSD: %f\n' % ind_obj_mssd_dict['006_mustard_bottle'])
        fo.write('021_bleach_cleanser_MSSD: %f\n' % ind_obj_mssd_dict['021_bleach_cleanser'])
        fo.write('003_cracker_box_MSSD: %f\n' % ind_obj_mssd_dict['003_cracker_box'])
        fo.write('004_sugar_box_MSSD: %f\n' % ind_obj_mssd_dict['004_sugar_box'])
        fo.write('010_potted_meat_can_MSSD: %f\n' % ind_obj_mssd_dict['010_potted_meat_can'])
        fo.write('037_scissors_MSSD: %f\n' % ind_obj_mssd_dict['037_scissors'])
        fo.write('035_power_drill_MSSD: %f\n' % ind_obj_mssd_dict['035_power_drill'])

        fo.write('Mean_MSSD: %f\n' % all_obj_mean_mssd)


    print('Scores written to: %s' % score_path)


    print('Evaluation complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('input_dir', type=str,
                        help='Path to where prediction the submited result and the ground truth is.')
    parser.add_argument('output_dir', type=str,
                        help='Path to where the eval result should be.')
    parser.add_argument('--pred_file_name', type=str, default='pred.json',
                        help='Name of the eval file.')
    args = parser.parse_args()

    # call eval
    main(
        gt_path=os.path.join(args.input_dir, 'ref'),
        pred_path=os.path.join(args.input_dir, 'res'),
        output_dir=args.output_dir,
        pred_file_name=args.pred_file_name,
        set_name='evaluation'
    )
