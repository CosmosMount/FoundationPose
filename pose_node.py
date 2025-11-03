import os
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
import subprocess
import threading
import signal
import trimesh
import numpy as np
from collections import deque
from cv_bridge import CvBridge

# FoundationPose 的依赖
from estimater import *
from datareader import *

import PIL


class FoundationPoseNode(Node):
    def __init__(self):
        super().__init__('foundationpose_node')

        self.base_dir = "demo_data/data2"
        self.color_dir = os.path.join(self.base_dir, "rgb")
        self.depth_dir = os.path.join(self.base_dir, "depth")
        self.masks_dir = os.path.join(self.base_dir, "masks")
        self.mesh_file = os.path.join(self.base_dir, "mesh/magic.obj")
        self.tex_file = os.path.join(self.base_dir, "mesh/magic_tex0.png")
        self.debug_dir = "debug"
        self.est_refine_iter = 5
        self.track_refine_iter = 2
        self.debug = 1

        # 初始化日志和随机种子
        set_logging_format()
        set_seed(0)

        # 加载网格和纹理
        self.mesh = trimesh.load(self.mesh_file)
        self.tex_img = PIL.Image.open(self.tex_file).convert("RGB")
        self.mesh.visual.material.image = self.tex_img

        # 接收 mask_ready 信号
        self.create_subscription(Bool, '/mask_ready', self.mask_ready_callback, 10)

        self.bridge = CvBridge()

        # 分别订阅 color / depth
        self.color_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')

        ats = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.05)
        ats.registerCallback(self.synced_callback)

        self.get_logger().info("🟢 FoundationPoseNode initialized. Waiting for /mask_ready=True ...")
        self.pose_thread = None
        self.mask_finished  = False

    def synced_callback(self, color_msg: Image, depth_msg: Image):
        if not self.mask_finished:
            return
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        timestamp = f"{color_msg.header.stamp.sec % 100000}_{color_msg.header.stamp.nanosec // 1000000:03d}"
        color_path = os.path.join(self.color_dir, f"{timestamp}.png")
        depth_path = os.path.join(self.depth_dir, f"{timestamp}.png")

        cv2.imwrite(color_path, color)
        cv2.imwrite(depth_path, depth)

        self.get_logger().info(f"✅ 保存帧 {color_path}, {depth_path}")

    def mask_ready_callback(self, msg):
        """收到 mask_ready=True 后启动位姿估计"""
        if msg.data:
            self.get_logger().info("✅ Received /mask_ready=True, starting FoundationPose estimation...")
            if self.pose_thread is None or not self.pose_thread.is_alive():
                # self.pose_thread = threading.Thread(target=self.run_foundationpose)
                # self.pose_thread.start()
                self.mask_finished = True

    def run_foundationpose(self):
        """完全复刻原始 FoundationPose 逻辑"""

        # 加载网格和纹理
        mesh = trimesh.load(self.mesh_file)
        tex_img = PIL.Image.open(self.tex_file).convert("RGB")
        mesh.visual.material.image = tex_img

        # 计算网格的边界框
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        # 初始化估计器
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=self.debug_dir,
            debug=self.debug,
            glctx=glctx,
        )
        logging.info("Estimator initialization done")

        # 加载数据
        reader = YcbineoatReader(video_dir=self.base_dir, shorter_side=None, zfar=np.inf)

        # 遍历每一帧进行估计
        for i in range(len(reader.color_files)):
            logging.info(f'i:{i}')
            color = reader.get_color(i)
            depth = reader.get_depth(i)

            if i == 0:
                # 初始帧：注册位姿
                mask = reader.get_mask(0).astype(bool)
                pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=self.est_refine_iter)

                if self.debug >= 3:
                    # 保存调试信息
                    m = mesh.copy()
                    m.apply_transform(pose)
                    m.export(f'{self.debug_dir}/model_tf.obj')
                    xyz_map = depth2xyzmap(depth, reader.K)
                    valid = depth >= 0.001
                    pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                    o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply', pcd)
            else:
                # 后续帧：跟踪位姿
                pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=self.track_refine_iter)

            if self.debug >= 1:
                # 可视化位姿
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(
                    color,
                    ob_in_cam=center_pose,
                    scale=0.1,
                    K=reader.K,
                    thickness=3,
                    transparency=0,
                    is_input_rgb=True,
                )
                cv2.imshow('1', vis[..., ::-1])
                cv2.waitKey(10)

            if self.debug >= 2:
                # 保存可视化结果
                os.makedirs(f'{self.debug_dir}/track_vis', exist_ok=True)
                imageio.imwrite(f'{self.debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

        # 关闭所有窗口
        cv2.destroyAllWindows()
        logging.info("🎯 FoundationPose estimation finished successfully.")


def main(args=None):
    rclpy.init(args=args)
    node = FoundationPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🟥 FoundationPoseNode interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
