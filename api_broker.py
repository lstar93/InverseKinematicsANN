""" Robot joints angles data broker """
#!/usr/bin/env python

# pylint: disable=W0212 # suppress protected access

import sys
import os
from pika import BlockingConnection, ConnectionParameters

def consumer(msgch):
    """ Ikine request consumer """
    # define callback to receive messages
    def callback(ch, method, properties, body):
        print(f' [x] Received {body}')

    msgch.basic_consume(queue='hello', auto_ack=True, on_message_callback=callback)

    # start consuming messages
    msgch.start_consuming()


if __name__ == '__main__':
    try:
        # create connection to localhost broker machine
        connection = BlockingConnection(ConnectionParameters('localhost'))
        channel = connection.channel()
        consumer(channel)
    except KeyboardInterrupt:
        print('CTRL+C interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0) # posix
