.PHONY: test docker run-docker test-docker clean

# Run all unit tests locally
test:
	pytest ../tests/

# Build Docker image from Dockerfile
docker:
	docker build -t pcl-train .

# Run Docker container interactively
run-docker:
	docker run -it --rm pcl-train

# Run unit tests inside Docker container
test-docker:
	docker run -it --rm pcl-train conda run --no-capture-output -n pcl_env pytest tests/

# Clean cached Docker artifacts and unused images
clean:
	rm -f pcl-train.tar
	docker image prune -f
