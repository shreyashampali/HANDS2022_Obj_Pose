import open3d as o3d
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
import torch
import smplx
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

    fig, ax = plt.subplots(1, 2)

    for f in files:
        # load image and annotation
        img = cv2.imread(os.path.join(img_dir, f+'.png'))
        seg = cv2.imread(os.path.join(seg_dir, f+'.png'))
        anno = loadPickleData(os.path.join(anno_dir, f+'.pickle'))

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
