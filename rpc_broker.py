""" Robot joints angles data broker """
#!/usr/bin/env python

# pylint: disable=W0212 # suppress protected access
# pylint: disable=W0105 # unnecesary strings
# pylint: disable=W0238 # unused private member

import sys
import os
import argparse
import json
from pika import BlockingConnection, ConnectionParameters, BasicProperties
from kinematics.inverse import FabrikInverseKinematics, AnnInverseKinematics
from robot.robot import OutOfRobotReachException
from robot.robot import SixDOFRobot as Robot


DEBUG_MSG = False
def debug_msg_print(msg):
    """ broker debug printouts """
    if DEBUG_MSG:
        print(msg)


def get_ikine_engine_cli():
    """ Setup ikine engine via CLI """
    # first check inverse kinematics method from CLI
    cliparser = argparse.ArgumentParser(prog='cli')
    cliparser.add_argument('--method', required=True, type=str, choices=['ann', 'fabrik'],
                            help='select inverse kinematics method, Neural Network or Fabrik')

    known_args, _ = cliparser.parse_known_args()
    if known_args.method == 'ann':
        cliparser.add_argument('--model', type=str, required=True,
            help='select .h5 file with saved model, required only if ann ikine method was choosed')

    args = cliparser.parse_args()
    ikine_method = args.method

    if ikine_method == 'ann':
        ann_model = args.model
        engine = AnnInverseKinematics(Robot.dh_matrix,
                                      Robot.links_lengths,
                                      Robot.effector_workspace_limits)
        engine.load_model(ann_model)

    if ikine_method == 'fabrik':
        engine = FabrikInverseKinematics(Robot.dh_matrix,
                                         Robot.links_lengths,
                                         Robot.effector_workspace_limits)
    return engine


class IkineRPCBroker:
    """ RPC channel callback wrapper """
    def __init__(self, ikine, host_ip = 'localhost', queue_name = 'ikine_queue'):
        """ Init ikine engine """
        # set engine
        self.__ikine = ikine
        # Open connection and channel
        self.__connection = BlockingConnection(ConnectionParameters(host=host_ip))
        self.__channel = self.__connection.channel()
        # define queue
        self.__queue = self.__channel.queue_declare(queue=queue_name)
        # one task at a time
        self.__channel.basic_qos(prefetch_count=1)
        self.__channel.basic_consume(queue=queue_name, on_message_callback=self.callback)

    def __exception_response(self, status, reason, corr_id):
        """ Exception response builder """
        return {'status' : status,
                'reason' : reason,
                'correlation_id' : corr_id}

    def callback(self, chan, method, props, body):
        """ Ikine rpc server callback """
        positions_json = json.loads(body.decode())
        positions = list(positions_json['positions'])

        # create empty response dictionary
        angles_dict = dict()

        # calculate inverse kinematics
        try:
            angles = self.__ikine.ikine(positions)
            angles_dict['status'] = 'OK'
        except OutOfRobotReachException as exception:
            debug_msg_print(str(exception))
            angles = None
            angles_dict = self.__exception_response('ERROR', str(exception), props.correlation_id)
        except TypeError as type_exception:
            debug_msg_print(str(type_exception))
            angles = None
            angles_dict = self.__exception_response('ERROR', str(exception), props.correlation_id)

        # create response json from dict
        angles_dict['angles'] = angles
        angles_json = json.dumps(angles_dict)

        # return angles to client
        chan.basic_publish(exchange='',
                           routing_key=props.reply_to,
                           properties=BasicProperties(correlation_id=props.correlation_id),
                           body=angles_json)
        chan.basic_ack(delivery_tag=method.delivery_tag)

    def start(self):
        """ Start consuming RPC clients requests """
        self.__channel.start_consuming()


def main():
    """ Start broker """
    try:
        ikine_engine = get_ikine_engine_cli()
        broker = IkineRPCBroker(ikine_engine)
        broker.start()

    except KeyboardInterrupt:
        print('CTRL+C interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0) # posix


if __name__ == '__main__':
    main()
