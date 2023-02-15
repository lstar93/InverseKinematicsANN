# InverseKinematicsANN
Inverse kinematics for 6DOF planar robot achieved with FABRIK algorithm and Artifical Neural Network.



## Description
There are two main user/programmer interfaces. 

1. CLI that allows to generate .csv file with trajectory coordinates and provide a convenient way to generate inverse kinamtics for robot. In both cases output can be visualized with matplotlib plots.

2. Scalable RPC server based on RabbitMQ message broker. User can run many brokers and split large datasets between them to obtain more efficient calculations and shorter response time. Project is set up to work within Docker container.



## How to use it
### CLI
You can use CLI to generate robot effector trajectory, save it to file, print to output and visualize using plots.

```shell
cli --generate-data --shape {circle,cube,cube_random,random,spring,random_dist} [--verbose] [--example] [--to-file TO_FILE]
```

Full list of arguments for trajectory generator
```shell
--shape                             TEXT             Generator trajectory shape.
--example                                            Show an example shape command.
# -> python cli.py --generate-data --shape circle --example
# ... --generate-data --shape circle --radius 3 --samples 20 --center 1,5,2

--to-file                           FILENAME.csv     Redirect trajectory to csv file.
--verbose                                            Plot trajectory and print points in command line.
```

You can also use CLI to calculate inverse kinematics for set of points.

```shell
cli --inverse-kine --method {ann,fabrik} [--example] --points POINTS [--to-file TO_FILE] [--verbose] [--show-path] [--separate-plots] --model MODEL
```

Full list of arguments for inverse kinematics
```shell
--method                            TEXT             Inverse kinematics engine, ann or fabrik.
--example                                            Show an example shape command.
--points                            FILENAME.csv     csv file with trajectory points.
--to-file                           FILENAME.csv     Redicrect angles to csv file.
--verbose                                            Plot trajectory and print angles in command line.
--show-path                                          Join trajectory points on plot and show path.
--separate-plots                                     Separate plots for assigned and predicted trajectory.
--model                             model.h5         h5 file with pretrained ANN model.
```

Example inverse kinematics command with **ann** method:
```shell
python cli.py --inverse-kine --method ann --model models/roboarm_model_1674153800-982793.h5 --points spring.csv --verbose
```

same operation using **fabrik**:
```shell
python cli.py --inverse-kine --method fabrik --points spring.csv --verbose
```

### RPC broker

#### Command line

To start broker within commandline first RabbitMQ service must be working on local machine. 

```shell
python rpc_broker.py --method ann --model models/roboarm_model_1674153800-982793.h5 
```

example client command:

```shell
python examples/rpc_client.py
```

#### Docker
Use the provided `docker-compose.yml` to run a container:
```shell
docker-compose up
```

Docker container use `ann` method and default model `roboarm_model_1674153800-982793.h5`.

If container is up, using example client is simple:

```shell
python examples/rpc_client.py
```

command used without argument will show example usage.


## Demonstration

Robot inverse kinematics for trajectory below was calculated with `ann` model `roboarm_model_1674153800-982793.h5`.

![](sample.gif)

## TODOs
1. Support for custom planar robots.
2. Interface to train ANN model for custom robots.
3. Add remote RPC.
4. Add angles limits for FABRIK algorithm.
