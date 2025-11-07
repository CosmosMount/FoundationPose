
import os
import cv2
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

from estimater import *
from datareader import *

# 新增 message_filters
import message_filters
import threading
from queue import Queue

class FoundationPoseEstimator:
    def __init__(self):
        rospy.init_node('foundation_pose_estimator', anonymous=True)
        # ... 省略前面初始化参数和YOLO加载部分 ...

        # ==================== 队列与线程 ====================
        self.frame_queue = Queue(maxsize=20)  # 最大缓存20帧
        self.worker_thread = threading.Thread(target=self.process_frame_thread, daemon=True)
        self.worker_thread.start()

        # ROS订阅
        rospy.Subscriber('/chair/detected', Bool, self.detection_callback, queue_size=10)

        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        # ApproxTimeSync
        ats = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.05)
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
    
    def synced_callback(self, color_msg, depth_msg):
        """ROS回调只放入队列，不处理"""
        if self.detection_triggered and self.frame_count < self.num_frames:
            try:
                self.frame_queue.put_nowait((color_msg, depth_msg))
            except Queue.Full:
                rospy.logwarn("⚠️ Frame queue full, dropping frame")

    def process_frame_thread(self):
        """单独线程处理队列里的图像"""
        while not rospy.is_shutdown():
            try:
                color_msg, depth_msg = self.frame_queue.get(timeout=0.1)
            except:
                continue

            # 转CV图像
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # 缩放
            new_w = int(color.shape[1] * self.scale)
            new_h = int(color.shape[0] * self.scale)
            color_resized = cv2.resize(color, (new_w, new_h), interpolation=cv2.INTER_AREA)
            depth_resized = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # YOLO检测
            results = self.model.predict(color_resized, imgsz=640, conf=0.5, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            if len(detections.xyxy) == 0:
                rospy.logwarn("⚠️ No object detected in frame, skipping...")
                continue

            # mask
            x1, y1, x2, y2 = map(int, detections.xyxy[0])
            mask = np.zeros((new_h, new_w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255

            # 保存文件
            timestamp = f"{self.frame_count:06d}"
            cv2.imwrite(os.path.join(self.color_dir, f"{timestamp}.png"), color_resized)
            cv2.imwrite(os.path.join(self.depth_dir, f"{timestamp}.png"), depth_resized)
            cv2.imwrite(os.path.join(self.masks_dir, f"{timestamp}.png"), mask)

            # 缩放内参（只在第一帧）
            if self.frame_count == 0 and self.orig_K is not None:
                K_scaled = self.orig_K.copy()
                K_scaled[0, 0] *= self.scale
                K_scaled[1, 1] *= self.scale
                K_scaled[0, 2] *= self.scale
                K_scaled[1, 2] *= self.scale
                np.savetxt(os.path.join(self.base_dir, 'cam_K.txt'), K_scaled, fmt='%.6f')
                rospy.loginfo(f"📏 Saved scaled intrinsics:\n{K_scaled}")

            self.frame_count += 1
            rospy.loginfo(f"💾 Saved frame {self.frame_count}/{self.num_frames}")

            # 如果收集完所有帧，开始位姿估计
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
                
                if self.debug >= 1:
                    center_pose = pose @ np.linalg.inv(to_origin)
                    vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                       thickness=3, transparency=0, is_input_rgb=True)
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