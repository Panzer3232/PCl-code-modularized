.PHONY: test train docker clean

# Run all unit tests
test:
	pytest tests/

# Run the training script (override dataset path if needed)
train:
	python train1.py 

# Build Docker image from Dockerfile
docker:
	docker build -t pcl-train .

# Clean cached Docker artifacts
clean:
	rm -f pcl-train.tar
