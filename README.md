# InverseKinematicsANN
Inverse kinematics for 6DOF planar robot achieved with FABRIK algorithm and Artifical Neural Network.



## Description
There are two main user/programmer interfaces. 

1. CLI that allows to generate .csv file with trajectory coordinates and provide a convenient way to generate inverse kinamtics for robot. In both cases output can be visualized with matplotlib plots.

2. Scalable RPC server based on RabbitMQ message broker. User can run many brokers and split large datasets between them to obtain more efficient calculations and shorter response time. Project is set up to work within Docker container.



## How to use it
### CLI
CLI can generates robot effector trajectory, save it to file, print to output and visualize using plots.

> usage: cli [-h] --generate-data --shape {circle,cube,cube_random,random,spring,random_dist} [--verbose] [--example] [--to-file TO_FILE]

**--shape** mandatory parameter to choose from available trajectory shapes,

**--example** show example shape command,

**--to-file** redirect shape points list to .csv file, command argument is filename,

**--verbose** print calculated angles to output and show plot.

Spring shape example:

> cli --generate-data --shape spring --samples 50 --dim 2,3,6

CLI generates robotic arm inverse kinematics using **ann** or **fabrik** method.

> usage: cli [-h] --inverse-kine --method {ann,fabrik} [--list-models] --points POINTS [--to-file TO_FILE] [--verbose] [--show-path] [--separate-plots]

**--method** set inverse kinematics algorithm, **ann** or **fabrik**,

**--model** load model of pretrained neural network from .h5 file, settable only if **ann** method was choosed,

**--separate-plots** used to separately plot input trajcetory and trajectory based on predicted inverse kinematics,

**--verbose** and **--to-file** parameters works the same as for trajectory.

Example inverse kinematics calculated with **ann**:

> cli --inverse-kine --method ann --model models/roboarm_model_1674153800-982793.h5 --points spring.csv --verbose

same operation using **fabrik**:

> cli --inverse-kine --method fabrik --points spring.csv --verbose

angles are returned in the same order as coresponding effector destination points.



### RPC broker

#### Docker

Docker container uses **ann** method and default model **roboarm_model_1674153800-982793.h5**.

Start container:

> docker-compose up

if container is up, using example client is simple:

> python examples/rpc_client.py

command used without argument will show example usage.

#### Command line

To start broker within commandline first RabbitMQ service must be working. 

Then start broker:

> python rpc_broker.py --method ann --model models/roboarm_model_1674153800-982793.h5 

client command stays the same:

> python examples/rpc_client.py



## Demonstration

Robot inverse kinematics for trajectory below was calculated with **ann** model **roboarm_model_1674153800-982793.h5**.

![](sample.gif)

## TODOs
1. Support for custom planar robots.
2. Interface to train ANN model for custom robots.
3. Add remote RPC.
4. Add angles limits for FABRIK algorithm.
