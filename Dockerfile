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

ARG ik_method
ENV env_ik_method=$ik_method

ARG ann_model
ENV env_ann_model=$ann_model

# start rpc broker command
CMD python rpc_broker.py --method ${env_ik_method} --model ${env_ann_model}