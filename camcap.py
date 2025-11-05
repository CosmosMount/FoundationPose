import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
import os
import shutil
import numpy as np
from datetime import datetime


class RealsenseSaver(Node):
    def __init__(self):
        super().__init__('realsense_saver')
        self.bridge = CvBridge()

        # 保存路径
        self.base_dir = "demo_data/data8"
        self.color_dir = os.path.join(self.base_dir, "rgb")
        self.depth_dir = os.path.join(self.base_dir, "depth")
        for d in [self.color_dir, self.depth_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        self.intrinsics_saved = False
        self.color_image = None
        self.depth_image = None
        self.frame_id = 0

        # 订阅 CameraInfo（单独）
        self.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)

        # 分别订阅 color / depth
        self.color_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')

        ats = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.05)
        ats.registerCallback(self.synced_callback)

        self.get_logger().info("✅ RealsenseSaver 已启动，等待图像流...")

    def synced_callback(self, color_msg: Image, depth_msg: Image):
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        timestamp = f"{color_msg.header.stamp.sec % 100000}_{color_msg.header.stamp.nanosec // 1000000:03d}"
        color_path = os.path.join(self.color_dir, f"{timestamp}.png")
        depth_path = os.path.join(self.depth_dir, f"{timestamp}.png")

        cv2.imwrite(color_path, color)
        cv2.imwrite(depth_path, depth)

        self.get_logger().info(f"✅ 保存帧 {self.frame_id}: {color_path}, {depth_path}")
        self.color_image = None
        self.depth_image = None
        self.frame_id += 1

    def camera_info_callback(self, msg):
        if not self.intrinsics_saved:
            K = np.array(msg.k).reshape(3, 3)
            intr_path = os.path.join(self.base_dir, "cam_K.txt")
            np.savetxt(intr_path, K, fmt="%.6f")
            self.intrinsics_saved = True
            self.get_logger().info(f"📸 已保存相机内参到 {intr_path}\n{K}")


def main(args=None):
    rclpy.init(args=args)
    node = RealsenseSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
