import json
from urllib.request import Request, urlopen
from urllib.error import URLError

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from std_msgs.msg import Int32


DASHBOARD_URL = "http://192.168.0.101:5001/api/sectors"

N_SECTORS = 5

# You can change these later
WARNING_DIRT_PERCENT = 50
CRITICAL_DIRT_PERCENT = 75

DIRT_MULTIPLIER = 4.0


def find_dirt_perc(
    frame_coords,
    frame_reflectivity,
    frame_coords_pre,
    threshold_distance,
    threshold_deriv,
    threshold_reflect,
    n_sectors
):
    perc_dirt = np.zeros(n_sectors)

    for sector in range(n_sectors):
        slice_start = (sector * frame_coords.shape[1]) // n_sectors
        slice_end = ((sector + 1) * frame_coords.shape[1]) // n_sectors

        dist = np.linalg.norm(
            frame_coords[:, slice_start:slice_end, :],
            axis=2
        ).flatten()

        reflect_flat = frame_reflectivity[:, slice_start:slice_end].reshape(-1)

        if frame_coords_pre.size == 0:
            dirt_mask = (
                (dist < threshold_distance)
                & (reflect_flat < threshold_reflect)
            )
        else:
            dist_pre = np.linalg.norm(
                frame_coords_pre[:, slice_start:slice_end, :],
                axis=2
            ).flatten()

            dist_deriv = dist - dist_pre

            dirt_mask = (
                (dist < threshold_distance)
                & (
                    (dist_deriv < threshold_deriv)
                    | (reflect_flat < threshold_reflect)
                )
            )

        perc_dirt[sector] = (np.sum(dirt_mask) / dirt_mask.size) * 100

    return perc_dirt


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

        self.timer = self.create_timer(1, self.publish_color)

        self.color = 3
        self.state = 'CLEAN'

        self.old_data = 0
        self.counter = 0

        self.dashboard_url = DASHBOARD_URL

    def post_dirt_to_dashboard(self, dirt):
        try:
            payload = json.dumps({
                "sectors": dirt.tolist()
            }).encode("utf-8")

            request = Request(
                self.dashboard_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            urlopen(request, timeout=0.2).read()

        except URLError as error:
            self.get_logger().warn(
                f"Could not post dirt sectors to dashboard: {error}"
            )
        except Exception as error:
            self.get_logger().warn(
                f"Dashboard POST error: {error}"
            )

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

    def lidar_callback(self, msg):
        raw_point_count = msg.width * msg.height

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

        if self.counter > 0:
            try:
                old_data = self.old_data[:, :3].reshape((16, 720, 3), order="C")
                current_data = point_matrix[:, :3].reshape((16, 720, 3), order="C")
                current_reflectivity = point_matrix[:, 5].reshape((16, 720), order="C")

            except Exception as error:
                self.get_logger().warn(f"Could not reshape lidar data: {error}")

                old_data = np.zeros((16, 720, 3), order="C")
                current_data = np.zeros((16, 720, 3), order="C")
                current_reflectivity = np.zeros((16, 720), order="C")

            dirt = find_dirt_perc(
                current_data,
                current_reflectivity,
                old_data,
                threshold_distance=0.1,
                threshold_deriv=0,
                threshold_reflect=100,
                n_sectors=N_SECTORS
            )

            # Apply the multiplier to the numpy array
            dirt = dirt * DIRT_MULTIPLIER

            self.get_logger().info(f"Dirt sectors: {dirt}")

            self.update_state_from_dirt(dirt)
            self.post_dirt_to_dashboard(dirt)

        self.old_data = point_matrix
        self.counter = 1


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