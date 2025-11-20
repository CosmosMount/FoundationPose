# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
import trimesh
import rospy
import csv


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))

  mesh_dir = "meshes/chair3/"
  mesh_file = os.path.join(mesh_dir, "chair.obj")
  tex_file = os.path.join(mesh_dir, "chair_tex0.png")
  test_scene_dir = "demo_data/data10"


  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(mesh_file)
  tex_img = PIL.Image.open(tex_file).convert("RGB")
  mesh.visual.material.image = tex_img

  debug = 1
  debug_dir = "debug"
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)

  # Open a CSV file to store Euler angles
  csv_file = os.path.join(debug_dir, 'euler_angles.csv')
  with open(csv_file, mode='w', newline='') as file:
      csv_writer = csv.writer(file)
      csv_writer.writerow(['Frame', 'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)'])  # Write header

      for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if i == 0:
          mask = reader.get_mask(0).astype(bool)
          pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5)

          if debug >= 3:
            m = mesh.copy()
            m.apply_transform(pose)
            m.export(f'{debug_dir}/model_tf.obj')
            xyz_map = depth2xyzmap(depth, reader.K)
            valid = depth >= 0.001
            pcd = toOpen3dCloud(xyz_map[valid], color[valid])
            o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
          pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=2)

        if debug >= 1:
          center_pose = pose @ np.linalg.inv(to_origin)
          R = center_pose[:3, :3]
          from scipy.spatial.transform import Rotation
          rotation = Rotation.from_matrix(R)
          quat = rotation.as_quat()  # [qx, qy, qz, qw]
          euler = rotation.as_euler('xyz', degrees=True)  # [roll, pitch, yaw]
          # yaw = np.arctan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]), 1.0 - 2.0 * (quat[1]**2 + quat[2]**2)) * 180 / np.pi

          # Write Euler angles to CSV
          csv_writer.writerow([i, euler[0], euler[1], euler[2]])

          print(f"🔍 Euler angles (degrees): Roll={euler[0]:.2f}, Pitch={euler[1]:.2f}, Yaw={euler[2]:.2f}")

          vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
          vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
          # cv2.imshow('1', vis[...,::-1])
          # cv2.waitKey(1)

          os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
          imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

          # # Ensure the directory exists
          # pose_dir = os.path.join(test_scene_dir, 'pose')
          # os.makedirs(pose_dir, exist_ok=True)

          # # Save the visualization to the specified directory
          # vis_path = os.path.join(pose_dir, f'{reader.id_strs[i]}.png')
          # imageio.imwrite(vis_path, vis)