import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

class RealSenseYoloNode(Node):
    def __init__(self):
        super().__init__('realsense_yolo_node')

        # ✅ 加载自定义 YOLO 模型
        self.train_dir = "detect/train2"
        self.model_path = os.path.join(self.train_dir, "weights/best.pt")
        self.model = YOLO(self.model_path)

        # 订阅 RealSense 彩色图像
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)

        # 初始化
        self.bridge = CvBridge()
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=1)
        self.first_frame_saved = False

        # mask 保存路径
        self.mask_dir = "/home/cosmosmount/Desktop/object_pose_estimation/masks"
        os.makedirs(self.mask_dir, exist_ok=True)

        self.get_logger().info("✅ RealSense YOLOv8 detection node started")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # ✅ 推理
        results = self.model.predict(frame, imgsz=640, conf=0.5, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # ✅ 绘制检测结果
        annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections)

        # ✅ 保存第一帧 mask
        if not self.first_frame_saved and len(detections.xyxy) > 0:
            self.first_frame_saved = True

            # 取第一个检测框
            x1, y1, x2, y2 = map(int, detections.xyxy[0])

            # 生成 mask：白色前景、黑色背景
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255

            # 保存 mask（与 RGB 同名）
            filename = "0000_mask.png"
            save_path = os.path.join(self.mask_dir, filename)
            cv2.imwrite(save_path, mask)
            self.get_logger().info(f"💾 Saved first-frame mask to {save_path}")

        # ✅ 可视化
        cv2.imshow("RealSense YOLOv8 Detection", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
