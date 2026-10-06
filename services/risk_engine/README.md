# Risk Engine Service (`services/risk_engine`)

Go streaming consumer and ONNX Runtime inference worker for pre-authorization transaction scoring.

## Module Layout

```text
services/risk_engine/
├── cmd/server/main.go               # Kafka consumer loop, Redis enrichment, ONNX scoring, decision output
├── internal/
│   ├── config/config.go             # Environment variable loader
│   ├── domain/                      # Domain package placeholder
│   ├── engine/onnx.go               # ONNX Runtime C-shared library session wrapper
│   └── features/
│       ├── builder.go               # 19-dimensional float32 feature vector assembly
│       ├── builder_test.go          # Deterministic vector encoding unit tests
│       └── velocity.go              # Redis ZSET sliding-window velocity pipeline (3600s window)
├── Makefile                         # Local build and test targets
├── go.mod
└── go.sum
```

## Build and Test

```bash
# Execute unit tests
go test -v ./...

# Compile binary
go build -o bin/chargeback-risk-engine ./cmd/server
```
