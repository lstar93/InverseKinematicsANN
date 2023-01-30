""" Example ikine api client """
#!/usr/bin/env python

# pylint: disable=W0212 # suppress protected access
# pylint: disable=W0105 # disable unnecesary strings warning

import json
from uuid import uuid4
from pika import BlockingConnection, ConnectionParameters, BasicProperties

class IkineRPCClient:
    """ Ikine RPC client """
    def __init__(self):
        self.connection = BlockingConnection(ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
                     queue=self.callback_queue,
                     on_message_callback=self.on_response,
                     auto_ack=True)

        self.response = None
        self.corr_id = None

    def on_response(self, chan, method, props, body):
        """ Request/response data handler """
        # if response correlation id match arrach body to response
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, pos_json):
        """ Call to ikine server """
        self.response = None
        self.corr_id = str(uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='ikine_queue',
            properties=BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id
            ),
            body=pos_json
        )
        self.connection.process_data_events(time_limit=None)
        return str(self.response)

if __name__ == '__main__':
    client = IkineRPCClient()
    positions_list = [[1,2,3], [4,2,2], [2,-1,-0]]
    positions_dict = dict()
    positions_dict['positions'] = positions_list
    positions_json = json.dumps(positions_dict)
    print(f'Requesting ikine for positions {positions_json}')
    print(f'Ikine response is {client.call(positions_json)}')
