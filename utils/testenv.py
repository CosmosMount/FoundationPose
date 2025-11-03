import rclpy
from rclpy.node import Node
# from std_msgs.msg import Bool
from sensor_msgs.msg import Image
import subprocess
import threading
import signal
from collections import deque

# FoundationPose 的依赖
import trimesh
import torch
import cv2
from estimater import FoundationPose
from datareader import YcbineoatReader
import Utils
import PIL

class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.callback, 10)

    def callback(self, msg):
        self.get_logger().info('Received an image')

def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()