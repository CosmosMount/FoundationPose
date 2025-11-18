
import os
import cv2

# os.environ['CUDA_HOME'] = '/usr/local/cuda'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# cuda_lib_path = '/usr/local/cuda/lib64'
# if cuda_lib_path not in os.environ.get('LD_LIBRARY_PATH', ''):
#     os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

import rospy
import shutil
import numpy as np
import trimesh
import PIL.Image
import torch
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from ultralytics import YOLO
import supervision as sv
from collections import deque
import message_filters

print(f"🔍 CUDA Debug Info:")
print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"   torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"   torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    print(f"   torch.version.cuda: {torch.version.cuda}")

from estimater import *
from datareader import *

class FoundationPoseEstimator:
    def __init__(self):
        rospy.init_node('foundation_pose_estimator', anonymous=True)
        
        # device_count = torch.cuda.device_count()
        # rospy.loginfo(f"🖥️  CUDA devices after ROS init: {device_count}")

        # ==================== 配置参数 ====================
        self.base_dir = rospy.get_param('~base_dir', 'demo_data/data_realtime')
        self.color_dir = os.path.join(self.base_dir, 'rgb')
        self.depth_dir = os.path.join(self.base_dir, 'depth')
        self.masks_dir = os.path.join(self.base_dir, 'masks')
        
        for d in [self.color_dir, self.depth_dir, self.masks_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        
        mesh_dir = rospy.get_param('~mesh_dir', 'meshes/chair3/')
        self.mesh_file = os.path.join(mesh_dir, 'chair.obj')
        self.tex_file = os.path.join(mesh_dir, 'chair_tex0.png')
        
        self.yolo_dir = "yolo/detect/train2"
        self.model_path = os.path.join(self.yolo_dir, "weights/best.pt")
        
        from datetime import datetime
        self.timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join('results', self.timestamp_prefix)
        os.makedirs(self.results_dir, exist_ok=True)
        rospy.loginfo(f"📂 Results will be saved to: {self.results_dir}")
    
        # 用于记录本次运行保存了多少张图片
        self.saved_vis_count = 0

        # YOLO安全加载
        try:
            from torch.nn.modules.container import Sequential
            from torch.nn.modules.conv import Conv2d
            from torch.nn.modules.batchnorm import BatchNorm2d
            from torch.nn.modules.activation import SiLU
            from ultralytics.nn.modules import Detect, C2f, SPPF, Concat, Conv
            
            torch.serialization.add_safe_globals([
                Sequential, Conv2d, BatchNorm2d, SiLU,
                Detect, C2f, SPPF, Concat, Conv
            ])
            self.model = YOLO(self.model_path)
        except Exception as e:
            rospy.logwarn(f"Method 1 failed: {e}, trying method 2...")
            import torch.serialization
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
            self.model = YOLO(self.model_path)
            torch.load = original_load
        
        self.num_frames = rospy.get_param('~num_frames', 5)
        self.scale = rospy.get_param('~scale', 0.5)
        self.est_refine_iter = rospy.get_param('~est_refine_iter', 5)
        self.track_refine_iter = rospy.get_param('~track_refine_iter', 2)
        self.debug = rospy.get_param('~debug', 1)
        self.debug_dir = rospy.get_param('~debug_dir', 'debug')
        
        # ==================== 状态变量 ====================
        self.bridge = CvBridge()
        self.detection_triggered = False
        self.frame_count = 0
        self.orig_K = None
        self.intrinsics_saved = False
        self.pose_queue = deque(maxlen=self.num_frames)
        
        # YOLO标注器
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=1)
        
        # ==================== ROS订阅与同步回调 ====================
        rospy.Subscriber('/chair/detected', Bool, self.detection_callback, queue_size=10)
        
        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback, queue_size=10)
        
        # ApproximateTimeSynchronizer 同步 RGB+Depth+CameraInfo
        ats = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=30, slop=0.5)
        ats.registerCallback(self.synced_callback)
        
        self.pose_pub = rospy.Publisher('/object_pose', PoseStamped, queue_size=10)
        
        set_logging_format()
        set_seed(0)
        
        rospy.loginfo("🟢 FoundationPoseEstimator initialized. Waiting for /detected_msg...")

    def detection_callback(self, msg):
        if msg.data and not self.detection_triggered:
            self.detection_triggered = True
            self.frame_count = 0
            rospy.loginfo(f"✅ Detection triggered! Starting to record {self.num_frames} frames...")

    def camera_info_callback(self, msg):
        """保存相机内参矩阵 K"""
        if not self.intrinsics_saved:
            K = np.array(msg.K).reshape(3, 3)
            self.orig_K = K.copy()
            intr_path = os.path.join(self.base_dir, "cam_K.txt")
            np.savetxt(intr_path, K, fmt="%.6f")
            self.intrinsics_saved = True
            rospy.loginfo(f"📸 Camera intrinsics received:\n{self.orig_K}")
    
    # def synced_callback(self, color_msg, depth_msg):

    #     if not self.detection_triggered or self.frame_count >= self.num_frames:
    #         return

    #     rospy.loginfo("🔔 synced_callback triggered")

    #     # 转CV图像
    #     color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
    #     depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        
    #     results = self.model.predict(color, imgsz=640, conf=0.5, verbose=False, device='cpu')[0]
    #     detections = sv.Detections.from_ultralytics(results)
        
    #     if len(detections.xyxy) == 0:
    #         rospy.logwarn("⚠️ No object detected in original frame, skipping...")
    #         return
        
    #     # 获取原始图像上的检测框
    #     x1_orig, y1_orig, x2_orig, y2_orig = map(int, detections.xyxy[0])
    #     rospy.loginfo(f"🎯 Detected bbox (original): [{x1_orig}, {y1_orig}, {x2_orig}, {y2_orig}]")

    #     # 缩放
    #     new_w = int(color.shape[1] * self.scale)
    #     new_h = int(color.shape[0] * self.scale)
    #     color_resized = cv2.resize(color, (new_w, new_h), interpolation=cv2.INTER_AREA)
    #     depth_resized = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
    #     # cv2.imshow('Camera RGB', color_resized)

    #     # YOLO检测
    #     # results = self.model.predict(color_resized, imgsz=640, conf=0.5, verbose=False)[0]
    #     # detections = sv.Detections.from_ultralytics(results)
    #     # if len(detections.xyxy) == 0:
    #     #     rospy.logwarn("⚠️ No object detected in frame, skipping...")
    #     #     return
        
    #     # mask
    #     x1 = int(x1_orig * self.scale)
    #     y1 = int(y1_orig * self.scale)
    #     x2 = int(x2_orig * self.scale)
    #     y2 = int(y2_orig * self.scale)
    #     mask = np.zeros((new_h, new_w), dtype=np.uint8)
    #     mask[y1:y2, x1:x2] = 255
        
    #     # 保存
    #     timestamp = f"{self.frame_count:06d}"
    #     cv2.imwrite(os.path.join(self.color_dir, f"{timestamp}.png"), color_resized)
    #     cv2.imwrite(os.path.join(self.depth_dir, f"{timestamp}.png"), depth_resized)
    #     cv2.imwrite(os.path.join(self.masks_dir, f"{timestamp}.png"), mask)
        
    #     # 缩放内参
    #     if self.frame_count == 0 and self.orig_K is not None:
    #         K_scaled = self.orig_K.copy()
    #         K_scaled[0, 0] *= self.scale
    #         K_scaled[1, 1] *= self.scale
    #         K_scaled[0, 2] *= self.scale
    #         K_scaled[1, 2] *= self.scale
    #         np.savetxt(os.path.join(self.base_dir, 'cam_K.txt'), K_scaled, fmt='%.6f')
    #         rospy.loginfo(f"📏 Saved scaled intrinsics:\n{K_scaled}")
        
    #     self.frame_count += 1
    #     rospy.loginfo(f"💾 Saved frame {self.frame_count}/{self.num_frames}")
        
    #     if self.frame_count == self.num_frames:
    #         rospy.loginfo("🚀 All frames collected. Starting FoundationPose estimation...")
    #         self.run_foundation_pose()

    def synced_callback(self, color_msg, depth_msg):
        """同步的 RGB + Depth 回调（不缩放版本）"""
        
        if not self.detection_triggered or self.frame_count >= self.num_frames:
            return

        rospy.loginfo("🔔 synced_callback triggered")

        # 转CV图像
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        
        # YOLO检测
        results = self.model.predict(color, imgsz=640, conf=0.5, verbose=False, device='cpu')[0]
        detections = sv.Detections.from_ultralytics(results)
        
        if len(detections.xyxy) == 0:
            rospy.logwarn("⚠️ No object detected in frame, skipping...")
            return
        
        # 获取检测框
        x1, y1, x2, y2 = map(int, detections.xyxy[0])
        rospy.loginfo(f"🎯 Detected bbox: [{x1}, {y1}, {x2}, {y2}]")
        
        # 生成 mask（不缩放）
        h, w = color.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        
        # 保存原始尺寸图像
        timestamp = f"{self.frame_count:06d}"
        cv2.imwrite(os.path.join(self.color_dir, f"{timestamp}.png"), color)
        cv2.imwrite(os.path.join(self.depth_dir, f"{timestamp}.png"), depth)
        cv2.imwrite(os.path.join(self.masks_dir, f"{timestamp}.png"), mask)
        
        # 保存原始内参（仅第一帧）
        if self.frame_count == 0 and self.orig_K is not None:
            np.savetxt(os.path.join(self.base_dir, 'cam_K.txt'), self.orig_K, fmt='%.6f')
            rospy.loginfo(f"📏 Saved original intrinsics:\n{self.orig_K}")
        
        self.frame_count += 1
        rospy.loginfo(f"💾 Saved frame {self.frame_count}/{self.num_frames}")
        
        if self.frame_count == self.num_frames:
            rospy.loginfo("🚀 All frames collected. Starting FoundationPose estimation...")
            self.run_foundation_pose()
    
    def run_foundation_pose(self):
        """运行FoundationPose位姿估计"""
        try:
            
            mesh = trimesh.load(self.mesh_file)
            tex_img = PIL.Image.open(self.tex_file).convert('RGB')
            mesh.visual.material.image = tex_img
            
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
            print("Loaded")
            scorer = ScorePredictor()
            print("ScorePredictor done")
            refiner = PoseRefinePredictor()
            print("RefinePredictor done")
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
            rospy.loginfo("✅ Estimator initialized")
            
            reader = YcbineoatReader(video_dir=self.base_dir, shorter_side=None, zfar=np.inf)
            
            for i in range(min(self.num_frames, len(reader.color_files))):
                rospy.loginfo(f"📊 Processing frame {i}/{self.num_frames}")
                
                color = reader.get_color(i)
                depth = reader.get_depth(i)
                
                if i == 0:
                    mask = reader.get_mask(0).astype(bool)
                    pose = est.register(
                        K=reader.K,
                        rgb=color,
                        depth=depth,
                        ob_mask=mask,
                        iteration=self.est_refine_iter
                    )
                else:
                    pose = est.track_one(
                        rgb=color,
                        depth=depth,
                        K=reader.K,
                        iteration=self.track_refine_iter
                    )
                
                self.pose_queue.append(pose)
                
                # if self.debug >= 1:
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                    thickness=3, transparency=0, is_input_rgb=True)
                
                vis_filename = f"frame_{self.saved_vis_count:04d}.png"
                vis_path = os.path.join(self.results_dir, vis_filename)
                cv2.imwrite(vis_path, vis[..., ::-1])  # RGB to BGR
                rospy.loginfo(f"💾 Saved visualization: {vis_path}")
                self.saved_vis_count += 1
                # cv2.imshow('FoundationPose', vis[..., ::-1])
                # cv2.waitKey(1)
            
            avg_pose = self.compute_average_pose()
            self.publish_pose(avg_pose)
            
            # cv2.destroyAllWindows()
            rospy.loginfo("🎯 FoundationPose estimation completed!")
            
            self.detection_triggered = False
            self.frame_count = 0
            self.pose_queue.clear()
            
        except Exception as e:
            rospy.logerr(f"❌ Error in FoundationPose: {e}")
            import traceback
            traceback.print_exc()
    
    def compute_average_pose(self):
        """计算多个位姿的平均值"""
        if len(self.pose_queue) == 0:
            return np.eye(4)
        
        translations = [pose[:3, 3] for pose in self.pose_queue]
        avg_translation = np.mean(translations, axis=0)
        
        from scipy.spatial.transform import Rotation
        rotations = [Rotation.from_matrix(pose[:3, :3]) for pose in self.pose_queue]
        avg_rotation = Rotation.from_quat(np.mean([r.as_quat() for r in rotations], axis=0))
        avg_rotation = avg_rotation.as_matrix()
        
        avg_pose = np.eye(4)
        avg_pose[:3, :3] = avg_rotation
        avg_pose[:3, 3] = avg_translation
        
        rospy.loginfo(f"📐 Average pose computed from {len(self.pose_queue)} frames")
        return avg_pose
    
    def publish_pose(self, pose):
        """发布位姿到ROS话题"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = 'camera_color_optical_frame'
        
        pose_msg.pose.position.x = pose[0, 3]
        pose_msg.pose.position.y = pose[1, 3]
        pose_msg.pose.position.z = pose[2, 3]
        
        from scipy.spatial.transform import Rotation
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_matrix(pose[:3, :3])
        quat = rotation.as_quat()
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        # 打印旋转矩阵，检查是否需要转置或坐标系转换
        R = pose[:3, :3]
        rospy.loginfo(f"Rotation matrix:\n{R}")
        
        # 尝试不同的 yaw 计算
        yaw1 = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
        yaw2 = np.arctan2(R[0, 1], R[0, 0]) * 180 / np.pi
        yaw3 = np.arctan2(R[1, 0], R[1, 1]) * 180 / np.pi
        yaw_from_quat = np.arctan2(2.0*(quat[3]*quat[2] + quat[0]*quat[1]), 
                                1.0 - 2.0*(quat[1]**2 + quat[2]**2)) * 180 / np.pi
        
        rospy.loginfo(f"Yaw candidates: {yaw1:.2f}°, {yaw2:.2f}°, {yaw3:.2f}°")
        rospy.loginfo(f"🔍 Yaw from quat: {yaw_from_quat:.2f}° (should match)")
        euler = rotation.as_euler('xyz', degrees=True)  # [roll, pitch, yaw]
        yaw = euler[2]
        
        self.pose_pub.publish(pose_msg)
        rospy.loginfo(f"📢 Published pose: xyz=({pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}), yaw={yaw:.2f}°")
    def run(self):
        """主循环"""
        rospy.spin()


if __name__ == '__main__':
    try:
        estimator = FoundationPoseEstimator()
        estimator.run()
    except rospy.ROSInterruptException:
        pass