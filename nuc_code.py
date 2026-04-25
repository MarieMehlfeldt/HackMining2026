import json
from urllib.request import Request, urlopen
from urllib.error import URLError

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

# Flask server endpoint to receive matrices
FLASK_SERVER_URL = "http://127.0.0.1:5001/matrices"


class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        self.subscription = self.create_subscription(
            PointCloud2,
            '/lidar/cloud/device_id47',
            self.lidar_callback,
            rclpy.qos.qos_profile_sensor_data
        )

        self.flask_url = FLASK_SERVER_URL

    def push_matrices_to_flask(self, coords, intensity, reflectivity):
        """Send the three matrices to the Flask server."""
        try:
            payload = json.dumps({
                "coords": coords.tolist(),
                "intensity": intensity.tolist(),
                "reflectivity": reflectivity.tolist()
            }).encode("utf-8")

            request = Request(
                self.flask_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            urlopen(request, timeout=5.0).read()
            self.get_logger().info("Successfully pushed matrices to Flask server.")

        except URLError as error:
            self.get_logger().warn(
                f"Could not push matrices to Flask server: {error}"
            )
        except Exception as error:
            self.get_logger().warn(
                f"Flask POST error: {error}"
            )

    def lidar_callback(self, msg):
        """Receive LiDAR PointCloud2 message, extract matrices, and push to Flask."""
        field_names = [field.name for field in msg.fields]

        points = pc2.read_points(
            msg,
            field_names=field_names,
            skip_nans=True
        )

        points_list = list(points)

        if len(points_list) == 0:
            self.get_logger().warn("All points are NaN.")
            return

        structured_array = np.array(points_list)

        point_matrix = np.stack(
            [structured_array[name] for name in field_names],
            axis=1
        )

        try:
            coords = point_matrix[:, :3].reshape((16, 720, 3), order="C")
            intensity = point_matrix[:, 4].reshape((16 * 720, 1), order="C")
            reflectivity = point_matrix[:, 5].reshape((16, 720, 1), order="C")

        except Exception as error:
            self.get_logger().error(f"Could not extract and reshape matrices: {error}")
            return

        self.push_matrices_to_flask(coords, intensity, reflectivity)


def main(args=None):
    rclpy.init(args=args)

    lidar_processor = LidarProcessor()

    try:
        rclpy.spin(lidar_processor)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()