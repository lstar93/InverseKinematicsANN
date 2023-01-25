""" Robot joints angles data broker """
#!/usr/bin/env python

# pylint: disable=W0212 # suppress protected access

import sys
import os
from pika import BlockingConnection, ConnectionParameters

# def consumer(msgch):
#     """ Ikine request consumer """
#     # define callback to receive messages

def hello_callback(chan, method, properties, body):
    """ 'hello' queue callback """
    print(f' [x] Received {body}')
    # acknowledge that message from queue was processed
    chan.basic_ack(delivery_tag = method.delivery_tag)

def reverser_callback(chan, method, properties, body):
    """ 'reverser' queue callback """
    print(f' [x] Body {body} reversed is {body[::-1]}')
    # acknowledge that message from queue was processed
    chan.basic_ack(delivery_tag = method.delivery_tag)

def create_topic(name, callback):
    """ create queue """
    channel.queue_declare(queue=name, durable=True)
    # start consuming messages
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=name, on_message_callback=callback)

if __name__ == '__main__':
    try:
        # create connection to localhost broker machine and new channel
        connection = BlockingConnection(ConnectionParameters('localhost'))
        channel = connection.channel()

        # create queue
        create_topic('hello', hello_callback)

        # create queue
        create_topic('reverser', reverser_callback)

        # start consuming message
        channel.start_consuming()

    except KeyboardInterrupt:
        print('CTRL+C interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0) # posix
