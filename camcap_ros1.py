import rospy
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge
import cv2
import os
import shutil
import numpy as np
from datetime import datetime
from ultralytics import YOLO


class RealsenseSaver:
    def __init__(self):
        rospy.init_node('realsense_saver', anonymous=True)
        self.bridge = CvBridge()

        # 保存路径
        self.base_dir = "demo_data/data11"
        self.color_dir = os.path.join(self.base_dir, "rgb")
        self.depth_dir = os.path.join(self.base_dir, "depth")
        self.mask_dir = os.path.join(self.base_dir, "masks")
        for d in [self.color_dir, self.depth_dir, self.mask_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        self.intrinsics_saved = False
        self.color_image = None
        self.depth_image = None
        self.frame_id = 0
        self.first_frame_mask_saved = False
        
        # 加载 YOLO 模型
        self.yolo_model_path = "yolo/runs/train/chair_detection/weights/best.pt"
        if not os.path.exists(self.yolo_model_path):
            # 尝试备用路径
            alt_path = "runs/train/chair_detection/weights/best.pt"
            if os.path.exists(alt_path):
                self.yolo_model_path = alt_path
            else:
                rospy.logwarn(f"⚠️  YOLO 模型未找到: {self.yolo_model_path}, mask 保存将被跳过")
                self.yolo_model = None
        
        if os.path.exists(self.yolo_model_path):
            try:
                self.yolo_model = YOLO(self.yolo_model_path)
                rospy.loginfo(f"✅ 已加载 YOLO 模型: {self.yolo_model_path}")
            except Exception as e:
                rospy.logerr(f"❌ 加载 YOLO 模型失败: {e}")
                self.yolo_model = None
        else:
            self.yolo_model = None

        # 订阅 CameraInfo（单独）
        rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.camera_info_callback, queue_size=10)

        # 分别订阅 color / depth
        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)

        # 同步 color 和 depth
        ats = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=1.0)
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

            # 第一帧：使用 YOLO 检测并保存 mask
            if self.frame_id == 0 and not self.first_frame_mask_saved and self.yolo_model is not None:
                self.save_first_frame_mask(color, timestamp)

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

    def save_first_frame_mask(self, color_image, timestamp):
        """使用 YOLO 检测第一帧并保存 mask"""
        try:
            # 运行 YOLO 推理
            results = self.yolo_model.predict(color_image, imgsz=640, conf=0.5, verbose=False, device='cpu')[0]
            
            # 创建空白 mask
            mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
            
            # 检查是否有检测结果
            if len(results.boxes) > 0:
                # 获取第一个检测框（置信度最高）
                box = results.boxes[0].xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                
                # 在检测框区域填充白色
                mask[y1:y2, x1:x2] = 255
                
                # 保存 mask
                mask_path = os.path.join(self.mask_dir, f"{timestamp}.png")
                cv2.imwrite(mask_path, mask)
                
                self.first_frame_mask_saved = True
                rospy.loginfo(f"🎯 YOLO 检测成功！保存 mask 到: {mask_path}")
                rospy.loginfo(f"   检测框: [{x1}, {y1}, {x2}, {y2}], 置信度: {results.boxes[0].conf[0]:.2f}")
            else:
                rospy.logwarn("⚠️  第一帧未检测到椅子，跳过 mask 保存")
                
        except Exception as e:
            rospy.logerr(f"❌ YOLO 检测或保存 mask 失败: {e}")
            import traceback
            traceback.print_exc()

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