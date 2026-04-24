How to connect via vs code 
Install the extension: In VS Code, install Remote - SSH (by Microsoft) if not already installed.
Add one of the following connections:
	• ssh ros@192.168.0.33 -p 2221
	• ssh ros@192.168.0.33 -p 2222
	• ...
	
Display data via Foxglove 
Install foxglove desktop, connect to ws://192.168.0.33:8765.
use layout

Display data via lichtblick 
Connect to lichtblick to see data, if visionary is shown:
* ssh -X ros@192.168.0.33 -p 2221
* google-chrome --no-sandbox --ignore-gpu-blocklist
	• then open http://localhost:8080/?ds=foxglove-websocket&ds.url=ws://localhost:8765 
for lighter data one can access lichtlick over websocket: http://192.168.0.33:8080/?ds=foxglove-websocket&ds.url=ws://192.168.0.33:8765

Change status of the traffic light:
ros2 topic pub /traffic_light_switch std_msgs/msg/Int32 "{data: 0}"
ros2 topic pub /traffic_light_switch std_msgs/msg/Int32 "{data: 1}"
Change the color of the traffic light:
ros2 topic pub /trafic_light_color std_msgs/msg/Int32 "{data: 3}"
