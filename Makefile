commit_hash=$(shell git rev-parse HEAD)

build:
	docker build -t "$(GCP_DOCKER_IMAGE):$(commit_hash)" .

push: build
	docker push "$(GCP_DOCKER_IMAGE):$(commit_hash)"

bash:
	docker run \
		-it --rm \
		--entrypoint bash \
		--volume "$(GOOGLE_APPLICATION_CREDENTIALS):/creds.json:ro" \
		--env "GOOGLE_APPLICATION_CREDENTIALS=/creds.json" \
		"$(GCP_DOCKER_IMAGE):$(commit_hash)"

local_pipeline:
	docker run \
		-it --rm \
		--entrypoint python \
		--volume "$(shell pwd)/tfx_pipeline_output:/tfx_pipeline_output:rw" \
		--env "PIPELINE_OUTPUT=/tfx_pipeline_output" \
		--volume "$(GOOGLE_APPLICATION_CREDENTIALS):/creds.json:ro" \
		--env "GOOGLE_APPLICATION_CREDENTIALS=/creds.json" \
		"$(GCP_DOCKER_IMAGE):$(commit_hash)" \
		-m local_runner


vertex_pipeline: push
	docker run \
		-it --rm \
		--entrypoint python \
		--env "GCP_PIPELINE_OUTPUT=$(GCP_PIPELINE_OUTPUT)" \
		--env "GCP_DOCKER_IMAGE=$(GCP_DOCKER_IMAGE):$(commit_hash)" \
		--env "GCP_SERVICE_ACCOUNT=$(GCP_SERVICE_ACCOUNT)" \
		--volume "$(GOOGLE_APPLICATION_CREDENTIALS):/creds.json:ro" \
		--env "GOOGLE_APPLICATION_CREDENTIALS=/creds.json" \
		--env "GCP_REGION=$(GCP_REGION)" \
		--env "GCP_PROJECT=$(GCP_PROJECT)" \
		--env "GCP_TRAIN_DATA=$(GCP_TRAIN_DATA)" \
		--env "GCP_INFERENCE_DATA=$(GCP_INFERENCE_DATA)" \
		"$(GCP_DOCKER_IMAGE)" \
		-m vertex_ai_runner

notebook:
	docker run \
		-it --rm \
		--entrypoint jupyter \
		-p 8888:8888 \
		--volume "$(shell pwd)/nbs:/nbs:rw" \
		--volume "$(shell pwd)/tfx_pipeline_output:/tfx_pipeline_output:ro" \
		--env "PIPELINE_OUTPUT=/tfx_pipeline_output" \
		--volume "$(shell pwd)/data:/data:ro" \
		--env "TRAINING_DATA_LOCATION=/data/train" \
		--env "TEST_DATA_LOCATION=/data/test" \
		--env "GCP_PIPELINE_OUTPUT=$(GCP_PIPELINE_OUTPUT)" \
		--env "GCP_REGION=$(GCP_REGION)" \
		--env "GCP_PROJECT=$(GCP_PROJECT)" \
		--volume "$(GOOGLE_APPLICATION_CREDENTIALS):/creds.json:ro" \
		--env "GOOGLE_APPLICATION_CREDENTIALS=/creds.json" \
		--env "PYTHONPATH=/tfx/src" \
		"$(GCP_DOCKER_IMAGE):$(commit_hash)" \
		notebook \
		 -y \
		 --no-browser \
		 --allow-root \
		 --ip=0.0.0.0 --port=8888 --port-retries=0 \
		 --notebook-dir=/nbs \
		 --NotebookApp.password='' \
		 --NotebookApp.token=''

tensorboard:
	docker run \
		-it --rm \
		-p 6006:6006 \
		--entrypoint tensorboard \
		--volume "$(GOOGLE_APPLICATION_CREDENTIALS):/creds.json:ro" \
		--env "GOOGLE_APPLICATION_CREDENTIALS=/creds.json" \
		--volume "$(shell pwd)/tfx_pipeline_output:/tfx_pipeline_output:ro" \
		"$(GCP_DOCKER_IMAGE)" \
		--bind_all \
		--port 6006 \
		--logdir="/tfx_pipeline_output/spaceshiptitanic/Trainer/model_run/$(shell ls tfx_pipeline_output/spaceshiptitanic/Trainer/model_run/ | sort -n | tail -n 1)"

# example:
# make remote_tensorboard run_dir=gs://pipelines-hackathon/spaceship-titanic/tfx_pipeline_output/spaceshiptitanic/170655986839/spaceshiptitanic-20220721110503/Trainer_2731620090976927744/model_run
remote_tensorboard:
	docker run \
		-it --rm \
		-p 6006:6006 \
		--entrypoint tensorboard \
		--volume "$(GOOGLE_APPLICATION_CREDENTIALS):/creds.json:ro" \
		--env "GOOGLE_APPLICATION_CREDENTIALS=/creds.json" \
		--volume "$(shell pwd)/tfx_pipeline_output:/tfx_pipeline_output:ro" \
		"$(GCP_DOCKER_IMAGE)" \
		--bind_all \
		--port 6006 \
		--logdir="$(run_dir)"

test-preprocessing:
	pytest -n auto -v \
		--cov-branch \
		--no-cov-on-fail \
		--cov-report=html:/coverage \
		--cov-report=term:skip-covered \
		--cov=preprocessing \
		preprocessing/tests/