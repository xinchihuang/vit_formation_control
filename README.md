# Learning Decentralized Formation Using Transformer codes
## System requirement
1. Ubuntu 20.04
2. ROS noetic

## Dependency
1. Get the simulation environment at:

   [A simple platform for multi-robot control with ROS](https://github.com/SIT-Robotics-and-Automation-Laboratory/robomaster_description.git)
 and place it under your working space.
2. Install dependent python packages by run:

   `pip install -r requirements.txt`
## Validation with pretrained model and 7 robots in gazebo simulator
   Start the simulation environment  and testing by run

   `roslaunch vit_formation_control robomaster_gazebo.launch`
   
   You can change the number of robots for testing by change the parameter robot_num i.e 13 robots:
   
   `roslaunch vit_formation_control robomaster_gazebo.launch robot_num:=13`

   Above code will start the simulation environment and load the pretrained model for testing. The simulation run will last for 50 seconds

## Retrain the model
   1. Prepare your training data. Or get the data we used while training at https://drive.google.com/file/d/1W7qcYFSZWF3qwU3ZK7PjQT3_lXx_Qk4P/view?usp=sharing
      
      Make sure you place the data in the right place.
      
      You can change the path to training data by editing parameter *data_path_list*
   2. Retrain the model with default parameter
     `python Train_vit_full.py`
      
      You can always change the training parameters by editing the parameters in the  Train_vit_full.py
## Real robot experiment
   This is for the specific robot used for real robot testing 
   Please refer to 

  [Customized robomaster](/scripts/realrobots/README.md)
   

