import open3d as o3d
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
import torch
import smplx
from PIL import Image, ImageDraw
plt.ion()


def loadPickleData(fName):
    with open(fName, 'rb') as f:
        try:
            pickData = pickle.load(f, encoding='latin1')
        except:
            pickData = pickle.load(f)

    return pickData

def get_mano_vertices(mano_layer, pose, shape, transl):
    mesh = mano_layer(global_orient=torch.from_numpy(pose[:3]).float().unsqueeze(0),
                                      hand_pose=torch.from_numpy(pose[3:]).float().unsqueeze(0),
                                      betas=torch.from_numpy(shape).float().unsqueeze(0),
                                      transl=torch.from_numpy(transl).float().unsqueeze(0))

    return mesh.vertices[0].numpy(), mano_layer.faces, mesh.joints[0].numpy()

def vis_2d_obj_corners(imgIn, gtIn, estIn=None, filename=None, upscale=1, lineThickness=3):
    jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    jointColsGt = (255,0,0)
    newCol = (jointColsGt[0] + jointColsGt[1] + jointColsGt[2]) / 3
    jointColsEst  = (newCol, newCol, newCol)

    # draws lines connected using jointConns
    img = np.zeros((imgIn.shape[0], imgIn.shape[1], imgIn.shape[2]), dtype=np.uint8)
    img[:, :, :] = (imgIn).astype(np.uint8)

    img = cv2.resize(img, (upscale * imgIn.shape[1], upscale * imgIn.shape[0]), interpolation=cv2.INTER_CUBIC)
    if gtIn is not None:
        gt = gtIn.copy() * upscale
    if estIn is not None:
        est = estIn.copy() * upscale

    for i in range(len(jointConns)):
        for j in range(len(jointConns[i]) - 1):
            jntC = jointConns[i][j]
            jntN = jointConns[i][j+1]
            if gtIn is not None:
                cv2.line(img, (int(gt[jntC,0]), int(gt[jntC,1])), (int(gt[jntN,0]), int(gt[jntN,1])), jointColsGt, lineThickness)
            if estIn is not None:
                cv2.line(img, (int(est[jntC,0]), int(est[jntC,1])), (int(est[jntN,0]), int(est[jntN,1])), jointColsEst, lineThickness)

    if filename is not None:
        cv2.imwrite(filename, img)

    return img


def get_keypoint_rgb(skeleton):
    rgb_dict = {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb_null'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith('index_null'):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith('middle_null'):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith('ring_null'):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith('pinky_null'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)

    return rgb_dict

def vis_2d_hand(img, kps, score, skeleton, filename=None, score_thr=0.4, line_width=3, circle_rad=3, save_path=None, hand_type='right'):
    rgb_dict = get_keypoint_rgb(skeleton)
    _img = Image.fromarray(img.astype('uint8'))
    draw = ImageDraw.Draw(_img)
    for i in range(21):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        colr_p_in = list(rgb_dict[parent_joint_name])
        colr_in = list(rgb_dict[joint_name])

        colr_p = colr_p_in
        colr = colr_in

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=tuple(colr_p), width=line_width)
        if score[i] > score_thr:
            draw.ellipse(
                (kps[i][0] - circle_rad, kps[i][1] - circle_rad, kps[i][0] + circle_rad, kps[i][1] + circle_rad),
                fill=tuple(colr))
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0] - circle_rad, kps[pid][1] - circle_rad, kps[pid][0] + circle_rad,
                          kps[pid][1] + circle_rad), fill=tuple(colr_p))

    return np.array(_img)

def load_skeleton(path, joint_num):

    # load joint info (name, parent_id)
    skeleton = [{} for _ in range(joint_num)]

    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id
    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        skeleton[i]['child_id'] = joint_child_id

    return skeleton

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--smplx_path', type=str, required=True)
    parser.add_argument('--object_models_dir', type=str, required=True)
    args = parser.parse_args()

    mano_layer = smplx.create(args.smplx_path, 'mano', use_pca=False, is_rhand=True, flat_hand_mean=True)

    img_dir = os.path.join(args.dataset_dir, 'train', 'images')
    anno_dir = os.path.join(args.dataset_dir, 'train', 'annotations')
    seg_dir = os.path.join(args.dataset_dir, 'train', 'seg')

    with open(os.path.join(args.dataset_dir, 'train.txt'), 'r') as f:
        files = f.readlines()
        files = [f.strip() for f in files]

    skeleton = load_skeleton(os.path.join(args.dataset_dir,'skeleton.txt'), 42)

    fig, ax = plt.subplots(1, 2)

    for f in files:
        # load image and annotation
        img = cv2.imread(os.path.join(img_dir, f+'.png'))
        seg = cv2.imread(os.path.join(seg_dir, f+'.png'))
        anno = loadPickleData(os.path.join(anno_dir, f+'.pickle'))

        # display the 2d projections of object corners and hand joints
        img = vis_2d_obj_corners(img, anno['obj_corners_2d'], lineThickness=1)
        img = vis_2d_hand(img, anno['joint_2d'], np.ones((21)), skeleton, line_width=1,
                                   circle_rad=1.5, hand_type='right')

        # display image and segmentation
        ax[0].imshow(img[:, :, ::-1])
        ax[1].imshow(seg[:, :, ::-1])
        plt.waitforbuttonpress(1)

        joint_locs = anno['joint_3d']
        mano_pose = anno['mano_pose']
        mano_shape = anno['mano_shape']

        obj_name = anno['obj_name']
        obj_rot = anno['obj_rot']
        obj_trans_root = anno['obj_trans']  # this obj translation is wrt root joint of hand
        # Get obj translation wrt to global frame of reference
        obj_trans_global = obj_trans_root + joint_locs[20]


        # Get MANO vertices and faces
        verts_right, faces, _ = get_mano_vertices(mano_layer, mano_pose, mano_shape,
                                                  np.zeros((3,)))



        # visualize hand and object meshes in open3d
        mesh_list = []
        opencv_to_opengl_conv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) * 1.0

        mesh_right_o3d = o3d.geometry.TriangleMesh()
        mesh_right_o3d.vertices = o3d.utility.Vector3dVector(verts_right)
        mesh_right_o3d.triangles = o3d.utility.Vector3iVector(faces)
        mesh_right_o3d.paint_uniform_color([0,0,1.])
        mesh_list.append(mesh_right_o3d.transform(opencv_to_opengl_conv))

        obj_mesh_path = os.path.join(args.object_models_dir, obj_name, 'textured.obj')
        obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
        obj_mesh.rotate(cv2.Rodrigues(obj_rot)[0].squeeze())
        obj_mesh.translate(obj_trans_global)
        obj_mesh.transform(opencv_to_opengl_conv)
        obj_mesh.paint_uniform_color([0, 1., 0.])
        mesh_list.append(obj_mesh)

        coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        mesh_list.append(coord_mesh)

        o3d.visualization.draw_geometries(mesh_list, mesh_show_back_face=True)
        print('Press \'Q\' on the open3d window to go to the next image...')

if __name__ == '__main__':
    main()
