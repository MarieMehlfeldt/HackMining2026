import json
from threading import Thread
from urllib.request import Request, urlopen
from urllib.error import URLError

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Int32
import time
import numpy as np

# Flask server endpoint to receive matrices
FLASK_SERVER_URL = "http://192.168.0.101:5001/matrices"

WARNING_DIRT_PERCENT = 50
CRITICAL_DIRT_PERCENT = 75

FLASK_IP = "192.168.0.101"
FLASK_PORT = "5001"
API_URL = f"http://{FLASK_IP}:{FLASK_PORT}/api/data"

def publish_color(publisher, state, prev_color):
    msg = Int32()
    color = 4
    if state == 'CLEAN':
        color = 4
    elif state == 'SAFE':
        color = 2
    elif state == 'WARN':
        color = 3
    elif state == 'DANGER':
        if prev_color == 1:
            color = 0
        else:
            color = 1
    else:
        color = 7

    msg.data = color
    publisher.publish(msg)
    return color

def fetch_dirt_levels(publisher):
    color = 7
    while True:
        time.sleep(0.5)
        try:
            # Request the data with a 2-second timeout so it doesn't hang if the server drops
            request = Request(API_URL, method="GET")
            response = urlopen(request, timeout=2.0)
            
            # Parse the JSON payload
            data = json.loads(response.read().decode("utf-8"))
            
            # Extract just the sector percentages, ignoring the massive "clean" and "dirty" point arrays
            sectors = data.get("sectors", [])
            
            if not sectors:
                continue
            # Calculate single-number metrics (matching your frontend logic)
            avg_dirt = sum(sectors) / len(sectors)
            max_dirt = max(sectors)
            
            print(f"Raw Sectors: {sectors}")
            print(f"Average Dirtiness: {avg_dirt:.1f}% | Max Sector: {max_dirt:.1f}%")
            print("-" * 40)

            if max_dirt >= CRITICAL_DIRT_PERCENT:
                state = 'DANGER'
            elif max_dirt >= WARNING_DIRT_PERCENT:
                state = 'WARN'
            elif max_dirt > 20:
                state = 'SAFE'
            else:
                state = 'CLEAN'
            
            color = publish_color(publisher, state, color)
            
        except URLError:
            print(f"Failed to connect to {API_URL}. Is the server running?")
        except TimeoutError:
            print("Request timed out.")
        except Exception as e:
            print(f"An error occurred: {e}")
        continue


class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        self.subscription = self.create_subscription(
            PointCloud2,
            '/lidar/cloud/device_id47',
            self.lidar_callback,
            rclpy.qos.qos_profile_sensor_data
        )

        self.publisher = self.create_publisher(
            Int32,
            '/trafic_light_color',
            10
        )
        self.lamp_thread = Thread(target=fetch_dirt_levels, args=(self.publisher,), daemon=True)
        self.lamp_thread.start()

        self.flask_url = FLASK_SERVER_URL
    
    def publish_color(self):
        msg = Int32()

        if self.state == 'CLEAN':
            self.color = 4
        elif self.state == 'SAFE':
            self.color = 2
        elif self.state == 'WARN':
            self.color = 3
        elif self.state == 'DANGER':
            if self.color == 1:
                self.color = 0
            else:
                self.color = 1
        else:
            self.color = 7

        msg.data = self.color
        self.publisher.publish(msg)

    def update_state_from_dirt(self, dirt):
        max_dirt = float(np.max(dirt))

        if max_dirt >= CRITICAL_DIRT_PERCENT:
            self.state = 'DANGER'
        elif max_dirt >= WARNING_DIRT_PERCENT:
            self.state = 'WARN'
        elif max_dirt > 20:
            self.state = 'SAFE'
        else:
            self.state = 'CLEAN'

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