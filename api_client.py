""" Example ikine api client """
#!/usr/bin/env python

from pika import BlockingConnection, ConnectionParameters, BasicProperties, spec

# create connection to localhost broker machine
connection = BlockingConnection(ConnectionParameters('localhost'))
channel = connection.channel()

# create queue where message will be delivered
channel.queue_declare(queue='hello', durable=True)
channel.queue_declare(queue='reverser', durable=True)

# publish simple message
channel.basic_publish(exchange='',
                      routing_key='hello',
                      body='Hello World!',
                      properties=BasicProperties(delivery_mode=spec.PERSISTENT_DELIVERY_MODE))

# publish simple message to different queue
channel.basic_publish(exchange='',
                      routing_key='reverser',
                      body='Hello World!',
                      properties=BasicProperties(delivery_mode=spec.PERSISTENT_DELIVERY_MODE))

# clode connection
connection.close()
