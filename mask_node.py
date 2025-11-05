import os
import cv2
import rclpy
import shutil
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ultralytics import YOLO
import supervision as sv
import numpy as np

class MaskAndIntrinsicsNode(Node):
    def __init__(self):
        super().__init__('mask_and_intrinsics_node')

        # 初始化路径（FoundationPose 直接会读取这里）
        self.base_dir = "demo_data/data8"
        self.color_dir = os.path.join(self.base_dir, "rgb")
        self.depth_dir = os.path.join(self.base_dir, "depth")
        self.masks_dir = os.path.join(self.base_dir, "masks")
        for d in [self.color_dir, self.depth_dir, self.masks_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        # YOLO 模型加载
        self.yolo_dir = "yolo/detect/train2"
        self.model_path = os.path.join(self.yolo_dir, "weights/best.pt")
        self.model = YOLO(self.model_path)

        # 订阅相机话题
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.K = None

        self.rgb_done = False
        self.depth_done = False
        self.intrinsics_saved = False
        self.first_frame_saved = False

        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)

        # 发布 /mask_ready
        self.ready_pub = self.create_publisher(Bool, '/mask_ready', 10)

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=1)

        self.color_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')

        ats = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.05)
        ats.registerCallback(self.synced_callback)

        self.get_logger().info("✅ MaskAndIntrinsicsNode started and waiting for first frame...")

    def camera_info_callback(self, msg):
        """保存相机内参矩阵 K"""
        if not self.intrinsics_saved:
            K = np.array(msg.k).reshape(3, 3)
            intr_path = os.path.join(self.base_dir, "cam_K.txt")
            np.savetxt(intr_path, K, fmt="%.6f")
            self.intrinsics_saved = True
            self.get_logger().info(f"📸 已保存相机内参到 {intr_path}\n{K}")

    def synced_callback(self, color_msg: Image, depth_msg: Image):
        if self.first_frame_saved:
            return
        
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        frame = color.copy()
        results = self.model.predict(frame, imgsz=640, conf=0.5, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections.xyxy) == 0:
            self.get_logger().info("⚠️ 未检测到目标，等待下一帧...")
            return
        
        x1, y1, x2, y2 = map(int, detections.xyxy[0])
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        
        timestamp = f"{color_msg.header.stamp.sec % 100000}_{color_msg.header.stamp.nanosec // 1000000:03d}"
        color_path = os.path.join(self.color_dir, f"{timestamp}.png")
        depth_path = os.path.join(self.depth_dir, f"{timestamp}.png")
        masks_path = os.path.join(self.masks_dir, f"{timestamp}.png")

        cv2.imwrite(color_path, color)
        cv2.imwrite(depth_path, depth)
        cv2.imwrite(masks_path, mask)

        self.first_frame_saved = True
        self.get_logger().info(f"💾 已保存 mask, rgb, depth 到 {self.base_dir}")

        # 发布完成信号
        msg = Bool()
        msg.data = True
        self.ready_pub.publish(msg)
        self.get_logger().info("📢 发布 /mask_ready = True")  

def main(args=None):
    rclpy.init(args=args)
    node = MaskAndIntrinsicsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
