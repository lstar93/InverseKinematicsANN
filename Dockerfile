# basic python image
FROM python:3.9

# install pika to access rabbitmq
RUN pip install pika
RUN pip install numpy keras scikit-learn scipy joblib tensorflow

# Without this setting, Python never prints anything out.
ENV PYTHONUNBUFFERED=1

# declare the source directory
ENV WORKSPACE /inversekine/
RUN mkdir -p $WORKSPACE
WORKDIR $WORKSPACE

# copy the files
COPY kinematics ./kinematics
COPY models ./models
COPY robot ./robot
COPY rpc_broker.py .

# start command
CMD [ "python", "rpc_broker.py", "--method", "fabrik" ]