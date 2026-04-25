from threading import Lock
pointcloud_state = {"clean": [], "dirty": []}
pointcloud_lock = Lock()