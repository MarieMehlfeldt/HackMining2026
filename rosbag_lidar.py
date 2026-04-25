"""Rosbag reading utilities for Lidar data."""
from pathlib import Path

import numpy as np
from rosbags.rosbag2.reader import Reader
from rosbags.typesys import Stores, get_typestore


def get_lidar_data(file:Path):
    """Loads lidar data from a rosbag file and yields the relevant components as numpy arrays.

    The arrays are returned as images with shape (H, W) and potentially a third dimension for
    coordinates.
    
    All arrays are returned as views into the original data buffer and cannot be written to.

    
    Args:
        file (Path): Path to the rosbag file to read from.
        
    Yields:
        out (np.ndarray, np.ndarray, np.ndarray): A tuple containing:
            - coords: An array (H, W, 3) float32 coordinates of the lidar points.
            - intensity: An array (H, W) float32 intensity values.
            - reflectivity: An array (H, W) uint16 reflectivity values.
    """
    size_lut = {
        1: np.int8,
        2: np.uint8,
        3: np.int16,
        4: np.uint16,
        5: np.int32,
        6: np.uint32,
        7: np.float32,
        8: np.float64
    }
    typestore = get_typestore(Stores.ROS2_JAZZY)
    with Reader(file) as reader:
        lidar_sets = [x for x in reader.connections if "lidar" in x.topic]

        blocksize = None
        intensity_offset = None
        intensity_dtype = None
        reflectivity_offset = None
        reflectivity_dtype = None
        for i, msg in enumerate(reader.messages(lidar_sets)):
            connection, timestamp, raw_data = msg
            msg = typestore.deserialize_cdr(raw_data, connection.msgtype)
            # blocksize = msg_to_block_size(raw_data)
            # print(blocksize)
            field_names = [field.name for field in msg.fields]
            # define and check assumptions about the data layout based on the header
            # we assume that the layout is the same for all messages
            if i == 0:
                blocksize = sum(size_lut[field.datatype]().itemsize for field in msg.fields)
                intensity_offset = msg.fields[field_names.index("intensity")].offset
                intensity_dtype = size_lut[msg.fields[field_names.index("intensity")].datatype]
                reflectivity_offset = msg.fields[field_names.index("reflectivity")].offset
                reflectivity_dtype = size_lut[
                    msg.fields[field_names.index("reflectivity")].datatype]
                assert field_names[:3] == ["x", "y", "z"],\
                    ValueError("Expected first three fields to be x, y, z,"
                               f" but got {field_names[:3]}")
                assert sum(size_lut[field.datatype]().itemsize for field in msg.fields[:3]) == 12,\
                    ValueError(f"Expected blocksize to be {blocksize}, but got"
                               f" {sum(size_lut[field.datatype] for field in msg.fields)}")

            # get and check the number of points in the dataset
            n_blocks = msg.data.size // blocksize
            assert n_blocks == msg.width * msg.height,\
                ValueError(f"Expected {msg.width * msg.height} blocks, but got {n_blocks}")
            # assert that data is contiguous in memory so the stride tricks work as intended
            # if speed becomes a problem, this can probably be removed because I think the
            # data is already loaded as a contiguous numpy array, so it should be fine
            buffer = np.ascontiguousarray(msg.data)
            # offset the array and stride along it to create views into the data for
            # coordinates, intensity, and reflectivity without copying the data
            coords = np.lib.stride_tricks.as_strided(
                buffer,
                shape=(n_blocks, 3*4),  # 3 coordinates, each 4 bytes (float32)
                strides=(blocksize * buffer.strides[0], buffer.strides[0]),
                writeable=False,
                ).view(np.float32).reshape((msg.height, msg.width, 3), order="C")
            intensity = np.lib.stride_tricks.as_strided(
                buffer[intensity_offset:], shape=(n_blocks, 4),
                strides=(blocksize * buffer.strides[0], buffer.strides[0]),
                writeable=False).view(intensity_dtype).reshape(
                    (msg.height, msg.width), order="C")
            reflectivity = np.lib.stride_tricks.as_strided(
                buffer[reflectivity_offset:], shape=(n_blocks, 2),
                strides=(blocksize * buffer.strides[0], buffer.strides[0]),
                writeable=False).view(reflectivity_dtype).reshape(
                    (msg.height, msg.width), order="C")
            # time = np.lib.stride_tricks.as_strided(
            #     buffer[16:], shape=(n_blocks, 4),
            #     strides=(blocksize * buffer.strides[0], buffer.strides[0]),
            #     writeable=False).view(np.uint32).reshape(
            #        (msg.height, msg.width), order="C")
            yield coords, intensity, reflectivity

if __name__ == "__main__":
    # this is a minimal example showing the usage of the get_lidar_data function.
    path = Path("D:\\minehack")
    for folder in path.iterdir():
        for file_ in folder.iterdir():
            if file_.suffix != ".mcap":
                continue
            try:
                print(len(list(get_lidar_data(file_))))
            except Exception as e:
                print(f"Error processing {file_}: {e}")
