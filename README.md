# InverseKinematicsANN
Inverse kinematics for 6DOF planar robot achieved separately with two algorithms FABRIK and Artifical Neural Network. User is free to choose which one he wants to use.



## Description
Project is written purely in Python.

There are two main user/programmer interfaces prepared. 

1. CLI that first allows you to generate .csv file with trajectory coordinates. It also enables way to generate inverse kinamtics for robot and visualize output. 

2. Scalable RPC server based on RabbitMQ message broker. User can run many brokers and split large datasets between them to obtain more efficient calculations and shorter response time.

## How to use it
### CLI
CLI has two main operations.

Generating robot effector single destination or trajectory. 

> usage: cli [-h] --generate-data --shape {circle,cube,cube_random,random,spring,random_dist} [--verbose] [--example] [--to-file TO_FILE]

Example usage for shape is returned from '--example' subcommand:

> cli.py --generate-data --shape random_dist --example
>> --generate-data --shape random_dist --dist normal --samples 100 --std_dev 0.35 --limits 0,3;0,4;0,5

To save points to .csv file use **--to-file** parameter with filename argument i.e. **circle.csv**. Generated points can be printed in commandline and printer if **--verbose** was set.

> cli.py --generate-data --shape circle --radius 3 --samples 20 --center 1,11,2 --to-file circle.csv

Second operation is calculating robotic arm inverse kinematics. Robot joints angles are returned in the same order as coresponding destination points. Inverse kinematics method must be set explicitly, **ann** or **fabrik**.

> usage: cli [-h] --inverse-kine --method {ann,fabrik} [--list-models] --points POINTS [--to-file TO_FILE] [--verbose] [--show-path] [--separate-plots

**--verbose** and **--to-file** parameters work as well here. **--separate-plots** is used to separate plots for input trajcetory and trajectory based on predicted angles. ANN mandatory parameter is model name, all available models in repository are in **models** directory, they can be listed by setting **--list-models** if ann algorithm was choosed.

> python .\cli.py --inverse-kine --method ann --list-models
>>Available models:
>>>roboarm_model_1674153800-982793.h5

**--model** parameter must be set explicitly
> cli.py --inverse-kine --method ann --model models/roboarm_model_1674153800-982793.h5 --points spring.csv --verbose

<br>

### RPC broker
RPC broker works by default on localhost:5672 (default RabbitMQ address and port). Broker has single queue named 'ikine_queue'. Example Client can be found in examples directory (rpc_client.py). To start IK broker first RabbitMQ service must be working.

## Demonstration

Robot IK for trajectory obtained with ANN model 'roboarm_model_1674153800-982793.h5'. Model is available in 'models' directory. Trajectory was generated with CLI command for 'spring' shape.

![](sample.gif)