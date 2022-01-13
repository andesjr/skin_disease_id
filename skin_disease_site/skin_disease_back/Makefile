train_model:
	python -m minimal_image_project_back.train_model

PROJECT_ID=	skin-disease-id

IMAGE_NAME = skin_cancer
# replace by your personal project id
GCR_REGION="europe-west1"


test_api_loccally:
	uvicorn api.fast:app --reload

# there is currently no tensorflow version available for linux_buster on arm architecture
build_image_on_mac_m1_for_mac_m1_warning_not_possible_if_tensorflow:
	docker build -t arm_image_that_will_not_be_runnable_on_cloud_run_and_that_cannot_contain_tensorflow .

build_image_on_linux_amd_for_any_platform_except_m1:
	docker build -t eu.gcr.io/${PROJECT_ID}/${IMAGE_NAME} .

build_image_on_mac_m1_for_cloud_run:
	docker buildx build \
    --platform linux/amd64 \
    -t eu.gcr.io/${PROJECT_ID}/${IMAGE_NAME} \
    --load \
    .

test_container_locally_on_mac_m1:
	docker run -e PORT=8000 -p 8080:8000 arm_image_that_will_not_be_runnable_on_cloud_run_and_that_cannot_contain_tensorflow

test_container_locally_on_linux_amd:
	docker run -e PORT=8000 -p 8080:8000 eu.gcr.io/${PROJECT_ID}/${IMAGE_NAME}

push_image_to_gcr:
	docker push eu.gcr.io/${PROJECT_ID}/${IMAGE_NAME}

deploy_container_to_cloud_run:
	gcloud run deploy \
	--image eu.gcr.io/${PROJECT_ID}/${IMAGE_NAME} \
	--platform managed \
	--region ${GCR_REGION}
