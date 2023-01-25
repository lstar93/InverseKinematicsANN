""" Example ikine api client """
#!/usr/bin/env python

from pika import BlockingConnection, ConnectionParameters

# create connection to localhost broker machine
connection = BlockingConnection(ConnectionParameters('localhost'))
channel = connection.channel()

# create queue where message will be delivered
channel.queue_declare(queue='hello')

# publish simple message
channel.basic_publish(exchange='', routing_key='hello', body='Hello World!')

# clode connection
connection.close()
