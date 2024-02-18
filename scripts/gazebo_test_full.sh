num_runs=100
# Loop to run the command multiple times
for ((i = 1; i <= num_runs; i++)); do
    echo "Run $i: Starting ROS Node..."
    rosrun vit_formation_control gazebo_test_full.py
done