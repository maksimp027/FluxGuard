.PHONY: infra-up infra-down train export run-producer run-engine test-go test-python

infra-up:
	cd infrastructure && docker-compose up -d

infra-down:
	cd infrastructure && docker-compose down

train:
	python3 ml/train.py

export:
	python3 ml/export_onnx.py

run-producer:
	python3 producer.py

run-engine:
	cd services/risk_engine && go run ./cmd/server/main.go

test-go:
	cd services/risk_engine && go test ./... -v

test-python:
	python3 -m unittest ml/test_pipeline.py -v
