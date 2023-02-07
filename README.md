# InverseKinematicsANN
Inverse kinematics for 6DOF planar robot achieved two ways FABRIK algorithm and Artifical Neural Network. User is able to choose which one to use.

## Description
Project is written purely in Python.

There are two main user/programmer interfaces prepared. 

1. CLI that allow you to generate .csv file with trajectory coordinates and provide a way to generate inverse kinamtics for robot and visualize output.

2. Scalable RPC server based on RabbitMQ message broker. User can run many brokers and split large datasets between them to obtain more efficient calculations and shorter response time.

## How to use it
### CLI
CLI has two main operations. First is generating robot effector trajectory.

> usage: cli [-h] --generate-data --shape {circle,cube,cube_random,random,spring,random_dist} [--verbose] [--example] [--to-file TO_FILE]

Example usage for shape can be listed by adding **--example** parameter:

> cli --generate-data --shape random_dist --example
>> --generate-data --shape random_dist --dist normal --samples 100 --std_dev 0.35 --limits 0,3;0,4;0,5

To save points as .csv file use **--to-file** parameter with csv filename as argument. If **--verbose** parameter was set generated points will be printed as command output and then plotted.

> cli --generate-data --shape circle --radius 3 --samples 20 --center 1,11,2 --to-file circle.csv --verbose

Second CLI main operation is robotic arm inverse kinematics calculation. Robot joints angles are returned in the same order as coresponding destination points. Inverse kinematics method must be set explicitly, **ann** or **fabrik**.

> usage: cli [-h] --inverse-kine --method {ann,fabrik} [--list-models] --points POINTS [--to-file TO_FILE] [--verbose] [--show-path] [--separate-plots]

**--verbose** and **--to-file** parameters works here as well. ANN mandatory parameter is neural network model name, all available models in repository are in **models** directory, they can be listed by setting **--list-models** if ann algorithm was choosed. **--separate-plots** is used to separately plot input trajcetory and trajectory based on predicted angles.

> cli --inverse-kine --method ann --list-models
>> Available models:
>>> roboarm_model_1674153800-982793.h5

For ANN algorithm **--model** parameter is mandatory:
> cli --inverse-kine --method ann --model models/roboarm_model_1674153800-982793.h5 --points spring.csv --verbose

Same operation for **fabrik**:

> cli --inverse-kine --method fabrik --points spring.csv --verbose

<br>

### RPC broker
RPC broker run by default on localhost:5672 (default RabbitMQ address and port). Broker has single queue named 'ikine_queue'. Example Client can be found in examples directory (examples/rpc_client.py). To start inverse kinematics broker first RabbitMQ must be installed and its service started.

Running broker is simple:

> python rpc_broker.py

If at least one instance of broker is running example client can be used:

> python examples/rpc_client.py

Command output will show example usage.

## Demonstration

Demonstrated Robot IK for trajectory below was obtained with ANN model 'roboarm_model_1674153800-982793.h5'. Model is available in **models** directory in repo. Trajectory was generated via CLI command for 'spring' shape.

![](sample.gif)

## TODOs
1. Support for custom planar robots.
2. Interface to train ANN model for custom robots.
3. Add remote RPC.
