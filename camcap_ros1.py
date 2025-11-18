import rospy
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge
import cv2
import os
import shutil
import numpy as np
from datetime import datetime


class RealsenseSaver:
    def __init__(self):
        rospy.init_node('realsense_saver', anonymous=True)
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
        rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.camera_info_callback, queue_size=10)

        # 分别订阅 color / depth
        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)

        # 同步 color 和 depth
        ats = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.05)
        ats.registerCallback(self.synced_callback)

        rospy.loginfo("✅ RealsenseSaver 已启动，等待图像流...")

    def synced_callback(self, color_msg, depth_msg):
        """同步回调函数：保存 RGB 和深度图像"""
        try:
            # 转换图像
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # 生成时间戳文件名
            timestamp = f"{color_msg.header.stamp.secs % 100000}_{color_msg.header.stamp.nsecs // 1000000:03d}"
            color_path = os.path.join(self.color_dir, f"{timestamp}.png")
            depth_path = os.path.join(self.depth_dir, f"{timestamp}.png")

            # 保存图像
            cv2.imwrite(color_path, color)
            cv2.imwrite(depth_path, depth)

            rospy.loginfo(f"✅ 保存帧 {self.frame_id}: {color_path}, {depth_path}")
            self.color_image = None
            self.depth_image = None
            self.frame_id += 1
        except Exception as e:
            rospy.logerr(f"❌ 保存图像失败: {e}")

    def camera_info_callback(self, msg):
        """保存相机内参"""
        if not self.intrinsics_saved:
            try:
                K = np.array(msg.K).reshape(3, 3)
                intr_path = os.path.join(self.base_dir, "cam_K.txt")
                np.savetxt(intr_path, K, fmt="%.6f")
                self.intrinsics_saved = True
                rospy.loginfo(f"📸 已保存相机内参到 {intr_path}\n{K}")
            except Exception as e:
                rospy.logerr(f"❌ 保存相机内参失败: {e}")

    def run(self):
        """主循环"""
        rospy.spin()


def main():
    try:
        saver = RealsenseSaver()
        saver.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()