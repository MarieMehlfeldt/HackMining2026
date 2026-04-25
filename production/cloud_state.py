from threading import Lock

# Unified state dict matching what the frontend expects
data_state = {
    "sectors": [],
    "dirty": [],
    "clean": []
}
data_lock = Lock()