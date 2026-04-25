import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from std_msgs.msg import Int32

def find_dirt_perc(frame_coords, frame_reflectivity, frame_coords_pre, threshold_distance, threshold_deriv, threshold_reflect, n_sectors):
    perc_dirt = np.zeros(n_sectors)
    for sector in range(n_sectors):
        slice_start = (sector * frame_coords.shape[1]) // n_sectors
        slice_end = ((sector + 1) * frame_coords.shape[1]) // n_sectors
        dist = np.linalg.norm(frame_coords[:, slice_start:slice_end, :], axis=2).flatten()
        reflect_flat = frame_reflectivity[:, slice_start:slice_end].reshape(-1)
        if frame_coords_pre.size == 0:
            dirt_mask = (dist < threshold_distance) & (reflect_flat < threshold_reflect)
        else:
            dist_pre = np.linalg.norm(frame_coords_pre[:, slice_start:slice_end, :], axis=2).flatten()
            dist_deriv = dist - dist_pre
            dirt_mask = (dist < threshold_distance) & ((dist_deriv < threshold_deriv) | (reflect_flat < threshold_reflect))
        perc_dirt[sector]=(np.sum(dirt_mask) / dirt_mask.size) * 100 # calculate percentage of dirt points in this sector and append to list
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

        # Timer to publish every 1 second
        self.timer = self.create_timer(1, self.publish_color)

        self.color = 3  # example value

        self.state='CLEAN'

        self.old_data=0
        self.counter=0

    def publish_color(self):
        msg = Int32()
        if self.state=='CLEAN':
            self.color=4
        elif self.state=='SAFE':
            self.color=2
        elif self.state=='WARN':
            self.color=3
        elif self.state=='DANGER':
            if self.color==1:
                self.color=0
            else:
                self.color=1
        else:
            self.color=7


        msg.data = self.color



        self.publisher.publish(msg)

        #self.get_logger().info(f'Published traffic light color: {msg.data}')

    def lidar_callback(self, msg):
        # 1. Total raw points from metadata
        raw_point_count = msg.width * msg.height

        # 2. Get all field names dynamically
        field_names = [field.name for field in msg.fields]
        #self.get_logger().info(f"Fields: {field_names}")

        # 3. Read points (skip NaNs)
        # Read points (structured array)
        points = pc2.read_points(
            msg,
            field_names=field_names,
            skip_nans=True
            )

        points_list = list(points)

        if len(points_list) == 0:
            self.get_logger().warn("All points are NaN.")
            return

            # Convert to structured NumPy array
        structured_array = np.array(points_list)

            # Build n x m matrix correctly
        point_matrix = np.stack(
            [structured_array[name] for name in field_names],
            axis=1
        )

        
        if(self.counter>0):
            old_data=self.old_data[:,:3].reshape((16,720, 3), order="C")
            current_data=point_matrix[:,:3].reshape((16,720, 3), order="C")
            current_reflectivity=point_matrix[:,5].reshape((16,720), order="C")
            dirt=find_dirt_perc(current_data,current_reflectivity,old_data,0.1,0,100,5)
            self.get_logger().info(f"{dirt}")
        self.old_data=point_matrix
        self.counter=1
        


        """
        self.get_logger().info(f"\nFirst 5 points:\n{point_matrix[:5]}")
        
        # 6. Parsed points
        parsed_point_count = point_matrix.shape[0]

        # 7. Logs
        self.get_logger().info(
            f"Raw points in msg: {raw_point_count} "
            f"(width: {msg.width}, height: {msg.height})"
        )

        self.get_logger().info(
            f"Valid points in matrix: {parsed_point_count} "
            f"(Matrix shape: {point_matrix.shape})"
        )

        # 8. Check dropped points
        if raw_point_count != parsed_point_count:
            dropped_points = raw_point_count - parsed_point_count
            self.get_logger().info(
                f"Difference: {dropped_points} points were dropped due to NaN values."
            )
        else:
            self.get_logger().info(
                "All raw points were successfully converted to the matrix."
            )
        """
        # 9. Example: access data
        # point_matrix[:, 0] -> x
        # point_matrix[:, 1] -> y
        # point_matrix[:, 2] -> z
        # point_matrix[:, 3] -> intensity (if available)


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