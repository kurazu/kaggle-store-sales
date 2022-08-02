# Start from TFX base image!
FROM gcr.io/tfx-oss-public/tfx:1.9.0

# Install additional dependencies
RUN python -m pip install -q \
    'tfx[kfp]==1.9.0' \
    'more-itertools' \
    'returns' \
    'sf-hamilton'

# Install development tools
RUN python -m pip install -q \
    ipdb \
    mypy \
    isort \
    flake8 \
    pytest \
    pytest-xdist[psutil] \
    types-protobuf \
    black==21.12b0 \
    pytest-cov

# Install our helper lib (it changes more often than rest of deps, so we put it on a separate build step)
RUN python -m pip install -q 'tfx-helper==0.2.4'

RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
    jupyter nbextension install --py --symlink tensorflow_model_analysis --sys-prefix && \
    jupyter nbextension enable --py tensorflow_model_analysis --sys-prefix

# It takes a while to download BERT, let's display a progress bar during build
ENV TFHUB_DOWNLOAD_PROGRESS=1

# Use ipdb for debugging (insert `breakpoint()` in your code)
ENV PYTHONBREAKPOINT=ipdb.set_trace

# Set the name of GCP project
# ENV GCP_PROJECT=tc-global-portal
# ENV GCP_REGION=europe-west4

# Copy MyPy configuration file
COPY mypy.ini ./mypy.ini

# We copy the pipeline creation code into the image, because we will run
# the pipeline through docker

# Copy the pipeline definition into the image
COPY pipeline ./pipeline

# Copy preprocessing code into the image
COPY preprocessing ./preprocessing

# Copy modelling code into the image
COPY models ./models

# Copy the runners into the image
COPY local_runner.py vertex_ai_runner.py ./
