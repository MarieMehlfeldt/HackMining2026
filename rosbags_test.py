from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from pathlib import Path
from enum import Enum
import numpy as np

class RosTypeEnum(Enum):
    np.float32 = 7
    np.float64 = 8
    np.int16 = 3
    np.int32 = 5
    np.int8 = 1
    np.uint16 = 4
    np.uint32 = 6
    np.uint8 = 2

def msg_to_block_size(msg):
    dtypes = [field.datatype for field in msg.fields]
    block_size = sum([RosTypeEnum[dtype].itemsize for dtype in dtypes])
    return block_size


if __name__ == "__main__":
    path = Path("D:\\minehack")
    for folder in path.iterdir():
        print(folder)
    typestore = get_typestore(Stores.ROS2_JAZZY)
    for folder in path.iterdir():
        for file in folder.iterdir():
            if file.suffix != ".mcap":
                continue
            print(file)
            with AnyReader([file], default_typestore=typestore) as reader:
                lidar_sets = [x for x in reader.connections if "lidar" in x.topic]
                print(lidar_sets)
                for connection in lidar_sets:
                    print(connection.topic)
                    for connection, timestamp, raw_data in reader.messages(lidar_sets):
                        msg = reader.deserialize(raw_data, connection.msgtype)
                        blocksize = msg_to_block_size(msg)
                        print(blocksize)
                        print(msg.header.frame_id)
                        print(msg.header.stamp)