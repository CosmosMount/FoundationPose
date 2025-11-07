#!/usr/bin/env python3
import os
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool  # ✅ 改为Bool
from cv_bridge import CvBridge
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

class RealSenseYoloNode:
    def __init__(self):
        rospy.init_node('realsense_yolo_node', anonymous=True)

        # YOLO模型
        self.train_dir = "detect/train2"
        self.model_path = os.path.join(self.train_dir, "weights/best.pt")
        self.model = YOLO(self.model_path)

        # 订阅彩色图像
        self.subscription = rospy.Subscriber(
            '/camera/color/image_raw',
            Image,
            self.image_callback,
            queue_size=10)

        # ✅ 订阅状态命令（Bool）
        self.state_sub = rospy.Subscriber(
            '/chair/state_command',
            Bool,
            self.state_command_callback,
            queue_size=10)

        # ✅ 发布检测结果（Bool）
        self.detected_pub = rospy.Publisher(
            '/chair/detected',
            Bool,
            queue_size=10)

        # 初始化
        self.bridge = CvBridge()
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=1)
        
        # 状态机变量
        # True -> tracking, False -> finding/detecting
        self.state_tracking = False
        self.detected = False

        rospy.loginfo("✅ RealSense YOLOv8 detection node started")

    def state_command_callback(self, msg: Bool):
        """收到布尔状态命令"""
        if msg.data:  # True -> tracking
            if not self.state_tracking:
                self.state_tracking = True
                self.detected = False
                rospy.loginfo("🔄 State: tracking (停止检测)")
        else:  # False -> finding
            if self.state_tracking:
                self.state_tracking = False
                self.detected = False
                rospy.loginfo("🔄 State: finding (重新开始寻找)")

    def image_callback(self, msg):
        if self.state_tracking:
            return  # tracking状态下不检测

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model.predict(frame, imgsz=640, conf=0.5, verbose=False, device='cpu')[0]
        detections = sv.Detections.from_ultralytics(results)

        # 状态机逻辑
        if len(detections.xyxy) > 0:
            # 发布检测结果（只发布一次）
            self.detected = True
            self.detected_pub.publish(Bool(data=True))
            rospy.loginfo("📢 Published chair detected message")
        elif len(detections.xyxy) == 0:
            self.detected = False

        # 可视化
        annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections)
        state_str = "tracking" if self.state_tracking else "detecting/finding"
        cv2.putText(annotated_frame, f"State: {state_str}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("RealSense YOLOv8 Detection", annotated_frame)
        cv2.waitKey(1)

def main():
    node = RealSenseYoloNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
