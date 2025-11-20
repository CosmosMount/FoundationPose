import os
import cv2
import sys
import glob
import torch
import rospy
import shutil
import numpy as np
import trimesh
import PIL.Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from std_msgs.msg import Bool, Float32MultiArray
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

# 必須與訓練資料夾 class 名稱一致、排序一致
class_names = [
    "chair_0",
    "chair_135",
    "chair_180",
    "chair_225",
    "chair_270",
    "chair_315",
    "chair_45",
    "chair_90",
]

# --------------------------
# Transform
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# --------------------------
# Load Model
# --------------------------
# model is loaded at import time to reuse between calls
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 8)
model.load_state_dict(torch.load("angle_model.pth", map_location="cpu"))
model.eval()

print("✅ Angle prediction model loaded.")

# --------------------------
# Predict Function
# --------------------------
def predict_angle(img_path):
    """Predict angle (int) for a single image path using the loaded model."""
    img = PIL.Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        _, pred = outputs.max(1)

    class_name = class_names[pred.item()]
    angle = int(class_name.split("_")[1])
    return angle

class FoundationPoseEstimator:
    def __init__(self):
        rospy.init_node('foundation_pose_estimator', anonymous=True)

        # ==================== 配置参数 ====================
        self.base_dir = rospy.get_param('~base_dir', 'demo_data/data_realtime')
        self.color_dir = os.path.join(self.base_dir, 'rgb')
        self.depth_dir = os.path.join(self.base_dir, 'depth')
        self.masks_dir = os.path.join(self.base_dir, 'masks')

        self.state_sub = rospy.Subscriber(
            '/start_detect_obj',
            Bool,
            self.state_command_callback,
            queue_size=10)
        
        for d in [self.color_dir, self.depth_dir, self.masks_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        
        mesh_dir = rospy.get_param('~mesh_dir', 'meshes/chair3/')
        self.mesh_file = os.path.join(mesh_dir, 'chair.obj')
        self.tex_file = os.path.join(mesh_dir, 'chair_tex0.png')
        
        # self.yolo_dir = "yolo/detect/train2"
        # self.model_path = os.path.join(self.yolo_dir, "weights/best.pt")
        self.model_path = "./best.pt"
        
        from datetime import datetime
        self.timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join('results', self.timestamp_prefix)
        os.makedirs(self.results_dir, exist_ok=True)
        rospy.loginfo(f"📂 Results will be saved to: {self.results_dir}")
    
        # 用于记录本次运行保存了多少张图片
        self.saved_vis_count = 0

        self.detected = False

        # YOLO安全加载
        try:
            # rospy.logwarn(f"Method 1 failed: {e}, trying method 2...")
            import torch.serialization
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
            self.model = YOLO(self.model_path)
            torch.load = original_load
        except Exception as e:
            rospy.logwarn(f"Method 1 failed: {e}, trying method 2...")
            import torch.serialization
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
            self.model = YOLO(self.model_path)
            torch.load = original_load
        
        self.num_frames = rospy.get_param('~num_frames', 20)
        self.scale = rospy.get_param('~scale', 0.5)
        self.est_refine_iter = rospy.get_param('~est_refine_iter', 5)
        self.track_refine_iter = rospy.get_param('~track_refine_iter', 2)
        self.debug = rospy.get_param('~debug', 1)
        self.debug_dir = rospy.get_param('~debug_dir', 'debug')

        self.detected_pub = rospy.Publisher('/object_detection', Bool, queue_size=10)
        self.pose_pub = rospy.Publisher('/object_6d_pose', Float32MultiArray, queue_size=10)
        self.find_object = False
        # self.test_timer = rospy.Timer(rospy.Duration(0.1), self.detection_callback)
        
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
        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback, queue_size=10)
        
        # ApproximateTimeSynchronizer 同步 RGB+Depth+CameraInfo
        ats = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=30, slop=0.5)
        ats.registerCallback(self.synced_callback)
        
        set_logging_format()
        set_seed(0)
        
        rospy.loginfo("🟢 FoundationPoseEstimator initialized.")

    def camera_info_callback(self, msg):
        """保存相机内参矩阵 K"""
        if not self.intrinsics_saved:
            K = np.array(msg.K).reshape(3, 3)
            self.orig_K = K.copy()
            intr_path = os.path.join(self.base_dir, "cam_K.txt")
            np.savetxt(intr_path, K, fmt="%.6f")
            self.intrinsics_saved = True
            rospy.loginfo(f"📸 Camera intrinsics received:\n{self.orig_K}")

    def state_command_callback(self, msg: Bool):
        """收到布尔状态命令"""
        if msg.data:  # True -> tracking
            if not self.find_object:
                self.find_object = True  
                self.detected = False
                rospy.loginfo("🔄 State: finding (开始寻找)")
                

    def synced_callback(self, color_msg, depth_msg):
        """同步的 RGB + Depth 回调（不缩放版本）"""      
        
        if not self.find_object or self.frame_count > self.num_frames + 1 or self.detected:
            rospy.loginfo("🟢 Waiting for state...")
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
        
        self.detected_pub.publish(Bool(data=True)) 
        
        # 获取检测框
        x1, y1, x2, y2 = map(int, detections.xyxy[0])
        rospy.loginfo(f"🎯 Detected bbox: [{x1}, {y1}, {x2}, {y2}]")
         
        
        # 保存原始尺寸图像
        timestamp = f"{self.frame_count:06d}"
        cv2.imwrite(os.path.join(self.color_dir, f"{timestamp}.png"), color)
        cv2.imwrite(os.path.join(self.depth_dir, f"{timestamp}.png"), depth)
        
        
        # 保存原始内参（仅第一帧）
        if self.frame_count == 0 and self.orig_K is not None:
            np.savetxt(os.path.join(self.base_dir, 'cam_K.txt'), self.orig_K, fmt='%.6f')
            rospy.loginfo(f"📏 Saved original intrinsics:\n{self.orig_K}")
            # 生成 mask（不缩放）
            h, w = color.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            cv2.imwrite(os.path.join(self.masks_dir, f"{timestamp}.png"), mask)
        
        self.frame_count += 1
        rospy.loginfo(f"💾 Saved frame {self.frame_count}/{self.num_frames}")
        
        # if self.frame_count == self.num_frames:
            
        
        # 额外采集一帧用于角度预测（裁剪后的）
        if self.frame_count == self.num_frames + 1:
            rospy.loginfo("📸 Capturing extra frame for angle prediction...")
            
            # YOLO检测
            results = self.model.predict(color, imgsz=640, conf=0.5, verbose=False, device='cpu')[0]
            detections = sv.Detections.from_ultralytics(results)
            
            if len(detections.xyxy) > 0:
                # 获取检测框
                x1, y1, x2, y2 = map(int, detections.xyxy[0])
                
                # 裁剪图像
                cropped_img = color[y1:y2, x1:x2]
                
                # 保存裁剪后的图像
                crop_path = os.path.join(self.base_dir, "angle_pred_crop.png")
                cv2.imwrite(crop_path, cropped_img)
                rospy.loginfo(f"💾 Saved cropped image for angle prediction: {crop_path}")
            else:
                rospy.logwarn("⚠️ No object detected in extra frame, using full image for angle prediction.")
                crop_path = os.path.join(self.base_dir, "angle_pred_crop.png")
                cv2.imwrite(crop_path, color)

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
                center_pose = pose @ np.linalg.inv(to_origin)
                self.pose_queue.append(center_pose)
                vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                    thickness=3, transparency=0, is_input_rgb=True)
                
                vis_filename = f"frame_{self.saved_vis_count:04d}.png"
                vis_path = os.path.join(self.results_dir, vis_filename)
                cv2.imwrite(vis_path, vis[..., ::-1])  # RGB to BGR
                rospy.loginfo(f"💾 Saved visualization: {vis_path}")
                self.saved_vis_count += 1
            
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
        """发布位姿到ROS话题 (FloatMultiArray格式)"""
        from std_msgs.msg import MultiArrayDimension, Float32MultiArray
        from scipy.spatial.transform import Rotation
        
        # 提取位置
        x = pose[0, 3]
        y = pose[1, 3]
        z = pose[2, 3]
        
        # 提取旋转并转换为四元数
        R = pose[:3, :3]
        rotation = Rotation.from_matrix(R)
        quat = rotation.as_quat()  # [qx, qy, qz, qw]
        
        # 输出3个旋转角
        # img_path = os.path.join(self.color_dir, f"{self.num_frames - 1:06d}.png")
        img_path = os.path.join(self.base_dir, "angle_pred_crop.png")
        yaw = 0.0
        try:
            if os.path.exists(img_path):
                # 调用全局定义的 predict_angle 函数
                yaw = predict_angle(img_path)
                rospy.loginfo(f"🧭 Predicted yaw angle: {yaw}")
            else:
                rospy.logwarn(f"⚠️ Angle prediction image not found: {img_path}")
        except Exception as e:
            rospy.logerr(f"❌ Angle prediction failed: {e}")
        # 构建 FloatMultiArray 消息
        # 顺序: [x, y, z, yaw, qx, qy, qz, qw]
        pose_array = Float32MultiArray()
        pose_array.data = [
            float(z),
            float(x),
            float(y),
            float(yaw*np.pi/180.0),
            float(quat[0]),  # qx
            float(quat[1]),  # qy
            float(quat[2]),  # qz
            float(quat[3])   # qw
        ]
        
        # 可选：添加维度信息（让接收方知道数组含义）
        pose_array.layout.dim.append(MultiArrayDimension())
        pose_array.layout.dim[0].label = "pose"
        pose_array.layout.dim[0].size = 8
        pose_array.layout.dim[0].stride = 8
        
        # 发布
        self.pose_pub.publish(pose_array)
        
        # 调试信息
        rospy.loginfo(f"📢 Published pose array:")
        rospy.loginfo(f"   Position (x, y, z): ({z:.3f}, {x:.3f}, {y:.3f})")
        rospy.loginfo(f"   Yaw: {yaw:.2f}°")
        rospy.loginfo(f"   Quaternion (x, y, z, w): ({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")
        rospy.loginfo(f"   Raw array: {pose_array.data}")

        self.detected = True
        self.find_object = False  
    
    
    def run(self):
        """主循环"""
        rospy.spin()


if __name__ == '__main__':
    try:
        estimator = FoundationPoseEstimator()
        estimator.run()
    except rospy.ROSInterruptException:
        pass