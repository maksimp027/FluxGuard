.PHONY: build test docker-build docker-up docker-down proto help

APP_NAME = chargeback-risk-engine
DOCKER_COMPOSE = docker compose -f deploy/docker-compose.yml

help:
	@echo "Available commands:"
	@echo "  make build         - Build local Go binary"
	@echo "  make test          - Run unit tests"
	@echo "  make docker-build  - Build Docker image using deploy/Dockerfile"
	@echo "  make docker-up     - Start all services with docker-compose"
	@echo "  make docker-down   - Stop and remove all docker-compose containers"
	@echo "  make proto         - Generate gRPC code from proto definition"

build:
	go build -o bin/$(APP_NAME) ./cmd/server

test:
	go test -v ./...

docker-build:
	docker build -t $(APP_NAME):latest -f deploy/Dockerfile .

docker-up:
	$(DOCKER_COMPOSE) up --build -d

docker-down:
	$(DOCKER_COMPOSE) down

proto:
	protoc --go_out=. --go_opt=paths=source_relative \
	       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
	       api/proto/risk_service.proto
