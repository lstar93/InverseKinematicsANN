""" Example ikine api client """
#!/usr/bin/env python

# pylint: disable=W0212 # suppress protected access
# pylint: disable=W0105 # unnecesary strings warning
# pylint: disable=W0613 # unused argument

import sys
import json
from uuid import uuid4
from pika import BlockingConnection, ConnectionParameters, BasicProperties


class IkineRPCClient:
    """ Ikine RPC client """
    def __init__(self, host_ip = 'localhost'):
        self.connection = BlockingConnection(ConnectionParameters(host=host_ip))
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
                     queue=self.callback_queue,
                     on_message_callback=self.__on_response,
                     auto_ack=True)

        self.response = None
        self.corr_id = None

    def __on_response(self, chan, method, props, body):
        """ Request/response data handler """
        # if response correlation id match atach body to response
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, pos_json, routing_key = 'ikine_queue'):
        """ Call to ikine server """
        self.response = None
        self.corr_id = str(uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key=routing_key,
            properties=BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id
            ),
            body=pos_json
        )
        self.connection.process_data_events(time_limit=None)
        return str(self.response)


def main():
    """ Example command line usage """
    # create rpc client
    client = IkineRPCClient()
    usage = "CLI Usage: rpc_client.py x,y,z;x,y,z;... "\
        "example: rpc_client.py 1,2,3;4,-2.1,3;1,-1,0.12"

    # assign postions
    positions_list = []
    try:
        if len(sys.argv) == 2:
            positions_string = list(str(sys.argv[1]).split(';'))
            positions_list = \
                [[float(x) for x in position.split(',')] for position in positions_string]
        else:
            raise ValueError
    except ValueError:
        print(usage)
        sys.exit(0)

    # create and send request
    positions_dict = dict()
    positions_dict['positions'] = positions_list
    positions_json = json.dumps(positions_dict)
    print(f'Requesting ikine for positions {positions_json}')
    print(f'Ikine response is {client.call(positions_json)}')


if __name__ == '__main__':
    main()
