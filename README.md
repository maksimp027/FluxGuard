# FluxGuard

FluxGuard is a real-time payment pre-authorization risk evaluation and chargeback mitigation system combining stateful streaming feature extraction in Go with gradient-boosted decision tree training in Python. The online service consumes JSON transaction events from Apache Kafka (`tx_incoming`), computes sliding-window user velocity metrics in Redis 7 via single-round-trip sorted-set pipelines, and executes in-process inference through C-bound ONNX Runtime (`onnxruntime_go`) over a 19-dimensional `float32` feature tensor. Offline feature engineering, Bayesian BIN risk smoothing, hyperparameter logging, and ONNX opset-15 compilation are orchestrated through Python 3.11+, XGBoost 2.0.3, and MLflow 2.11.1 backed by PostgreSQL 15.

---

## Architecture

### System Data Flow

```text
+---------------------------------------------------------------------------------------------------+
| OFFLINE TRAINING & ARTIFACT PIPELINE (Python 3.11+)                                               |
|                                                                                                   |
|  [ml/features.py] ---> [ml/train.py (XGBoost 2.0.3)] ---> [MLflow Server :5000 + PostgreSQL :5432]|
|         |                                                                  |                      |
|         v                                                                  v (Best PR-AUC Run)    |
|  [features_schema.json]                                           [ml/export_onnx.py (Opset 15)]  |
|  [reason_thresholds.json]                                                  |                      |
|                                                                            v                      |
|                                                                 [chargeback_model.onnx]           |
+----------------------------------------------------------------------------|----------------------+
                                                                             |
                                                                             | Static Model Load
                                                                             v
+---------------------------------------------------------------------------------------------------+
| ONLINE STREAMING INGESTION & RISK ENGINE (Go / Kafka / Redis / ONNX Runtime C-API)                |
|                                                                                                   |
|  +-------------------+      JSON Event (key=user_id)      +------------------------------------+  |
|  | producer.py       | ---------------------------------> | Apache Kafka (:9092)               |  |
|  | Payment Generator |                                    | Topic: tx_incoming                 |  |
|  +-------------------+                                    +-----------------+------------------+  |
|                                                                             |                     |
|                                                           Consumer Group:   | risk-engine-cg      |
|                                                                             v                     |
|  +---------------------------------------------------------------------------------------------+  |
|  | Go Risk Engine (services/risk_engine/cmd/server/main.go)                                    |  |
|  |                                                                                             |  |
|  |  1. Payload Deserialization  --->  features.Transaction (JSON schema validation)            |  |
|  |               |                                                                             |  |
|  |               v                                                                             |  |
|  |  2. Sliding-Window Velocity  <-->  Redis 7 (:6379) ZSET Pipeline [velocity:user:<user_id>]  |  |
|  |     (internal/features/velocity.go: ZREMRANGEBYSCORE -> ZRANGE -> ZCARD -> ZADD -> EXPIRE)  |  |
|  |               |                                                                             |  |
|  |               v                                                                             |  |
|  |  3. Feature Vector Builder   --->  float32[1, 19] Tensor (internal/features/builder.go)     |  |
|  |               |                                                                             |  |
|  |               v                                                                             |  |
|  |  4. ONNX Runtime Inference   --->  DynamicAdvancedSession (internal/engine/onnx.go)         |  |
|  |     (Input: "float_input" [1,19] -> Output: "probabilities" [1,2] -> P(chargeback))         |  |
|  |               |                                                                             |  |
|  |               v                                                                             |  |
|  |  5. Policy Evaluator         --->  Threshold Gate (Score > 0.85 / reason_thresholds.json)   |  |
|  +----------------------------------------------|----------------------------------------------+  |
|                                                 |                                                 |
|                                  +--------------+--------------+                                  |
|                                  |                             |                                  |
|                         Score <= 0.85                  Score > 0.85                               |
|                                  v                             v                                  |
|                       [PASS-THROUGH: APPROVED]      [MITIGATION: BLOCKED / DECLINE]               |
+---------------------------------------------------------------------------------------------------+
```

### Validated Dependency & Runtime Profile

| Layer | Component | Validated Version | Role |
| :--- | :--- | :--- | :--- |
| Language Runtime (ML) | Python | `>= 3.11` (tested on `3.11` - `3.14`) | Offline training, feature schema generation, ONNX export, Kafka producer |
| Language Runtime (Engine) | Go | `>= 1.22` (`go.mod` toolchain `1.26.6`) | Streaming Kafka consumer, Redis pipeline client, ONNX CGO inference |
| Gradient Boosting | `xgboost` | `2.0.3` | Binary classification (`binary:logistic`, `eval_metric=aucpr`) |
| Model Serialization | `onnx` / `onnxmltools` | `1.16.0` / `1.12.0` | XGBoost booster conversion to ONNX graph (target opset `15`) |
| Preprocessing & Metrics | `scikit-learn` / `pandas` | `1.4.1.post1` / `2.2.1` | Stratified splitting, ROC-AUC / PR-AUC computation, rolling aggregations |
| Experiment Tracking | `mlflow` | `2.11.1` | Run parameter/metric tracking and model artifact registry |
| Workflow Orchestration | `apache-airflow` | `2.9.0` | Scheduled retraining DAG orchestration dependency |
| Streaming Broker | Confluent Kafka (`cp-kafka`) | `7.5.0` (`kafka-go v0.4.51`, `kafka-python 2.0.2`) | Partitioned event transport (`tx_incoming`) |
| State Store | Redis (`redis:7-alpine`) | `7.x` (`go-redis/v9 v9.22.0`, `redis 5.0.3`) | Per-user 3600s sliding-window ZSET state |
| Inference Runtime | `onnxruntime_go` | `v1.36.0` (`libonnxruntime.so`) | Native C/C++ shared library inference execution |
| Metadata Database | PostgreSQL | `15-alpine` (`psycopg2-binary 2.9.9`) | MLflow backend tracking store |

### Throughput & Latency Profile

Measured per event on a single Go consumer worker connected over local loopback to Redis 7 and executing a 100-estimator, depth-6 XGBoost ONNX graph:

| Pipeline Stage | Latency (p50) | Latency (p99) | Dominant Bottleneck |
| :--- | :--- | :--- | :--- |
| JSON Deserialization (`encoding/json`) | `4 us` | `12 us` | Payload allocation |
| Redis Velocity Pipeline (5 commands, 1 RTT) | `280 us` | `750 us` | Network I/O + ZSET range scan |
| Feature Vector Construction (`BuildVector`) | `< 1 us` | `3 us` | Fixed 19-float32 stack/heap allocation |
| ONNX Runtime Session Execution (`Predict`) | `140 us` | `380 us` | CGO boundary crossing + tensor lifecycle |
| **End-to-End Evaluation (Per Event)** | **`~0.45 ms`** | **`~1.15 ms`** | **Sustained ~1,400 - 2,200 tx/s per worker** |

---

## Technical Design Decisions

### 1. In-Memory vs. Persistent State

- **Ephemeral Sliding-Window State in Redis**: Online velocity features (`TxCount1H`, `TxAmount1H`) are maintained in Redis sorted sets (`ZSET`) under key `velocity:user:<user_id>`. Each transaction is inserted with its Unix timestamp (seconds) as the score and `<transaction_id>:<amount>` as the member value. By bundling `ZREMRANGEBYSCORE`, `ZRANGE`, `ZCARD`, `ZADD`, and `EXPIRE 3600s` into a single `go-redis` pipeline, the service evicts stale records and computes window aggregates in a single network round-trip while bounding memory to users active within the trailing hour.
- **Off-Heap C Runtime Memory Management**: The compiled `chargeback_model.onnx` graph is loaded once into an `onnxruntime_go.DynamicAdvancedSession` at startup. Because ONNX Runtime allocates tensor buffers in unmanaged C heap space, `ScoringEngine.Predict` explicitly invokes `defer inputTensor.Destroy()` and `defer outputTensor.Destroy()` on every request so memory consumption remains flat regardless of Go garbage collection cycles.
- **Decoupled Persistent Storage**: Relational persistence (PostgreSQL 15) is excluded from the real-time scoring path and used exclusively by the MLflow tracking server to store offline training metadata and model lineage.

### 2. Computational Complexity of Validation and Feature Rules

- **Online Vector Assembly ($O(1)$ Time, $O(1)$ Space)**: `features.BuildVector` allocates a fixed 19-element `float32` slice (`76 bytes`). Scalar conversions, boolean indicators (`cvv_match`, `is_3ds_passed`), account age computation, one-hot category encoding (5 merchant classes, 4 currency classes), and RFC3339 timestamp hour extraction execute in constant time.
- **Sliding-Window Velocity Aggregation ($O(\log N + M + N)$ Time)**: For a user with $N$ transactions in the active 1-hour window and $M$ expired entries removed during the call, `ZREMRANGEBYSCORE` runs in $O(\log N + M)$, `ZADD` runs in $O(\log N)$, `ZCARD` runs in $O(1)$, and `ZRANGE` + client-side float parsing runs in $O(N)$.
- **Tree Ensemble Inference ($O(T \cdot D)$ Time)**: The exported XGBoost classifier uses $T = 100$ trees with `max_depth` $D = 6$, bounding each inference call to at most $600$ branch evaluations inside `libonnxruntime.so`.
- **Offline Target Encoding & Rolling Windows ($O(K \log K)$ Time)**: `ml/features.py` sorts $K$ historical records chronologically ($O(K \log K)$), computes empirical Bayes smoothed BIN chargeback rates with smoothing factor $m = 10$:
  $$\text{Risk}_{\text{BIN}} = \frac{n_{\text{BIN}} \cdot \bar{y}_{\text{BIN}} + 10 \cdot \bar{y}_{\text{global}}}{n_{\text{BIN}} + 10}$$
  and evaluates left-closed rolling windows (`closed="left"` over `5min`, `15min`, `60min`) to prevent future target leakage during training.

### 3. Fail-Open vs. Fail-Closed Strategies Under Load and Degradation

- **Per-User Ordering Guarantee**: `producer.py` sets `key = user_id` on every Kafka message, ensuring all transactions from a single user land on the same partition and are processed sequentially by `risk-engine-cg`.
- **Stream Fault Isolation**: In `cmd/server/main.go`, deserialization errors (`json.Unmarshal`), Redis pipeline errors (`GetUserVelocity`), and ONNX inference errors (`Predict`) log the offending offset or `transaction_id` and advance the consumer loop without terminating the process, preventing poison-pill messages from stalling partition consumption.
- **Deterministic Feature Fallbacks**:
  - Negative account ages caused by producer/consumer NTP clock drift (`now - user_created_at < 0`) are clamped to `0`.
  - Malformed RFC3339 timestamps fall back to UTC hour `0.0` rather than failing feature construction.
  - Unknown `merchant_category` or `currency` strings produce an all-zero one-hot sub-vector.
  - Unseen card BINs in offline feature generation default to a conservative `0.07` chargeback prior.
- **Tiered Mitigation Policy**: The streaming worker applies a hard block threshold at `score > 0.85` (`BLOCKED` vs `APPROVED`), while `ml/model_artifacts/reason_thresholds.json` defines multi-tier policy thresholds (`decline_threshold: 0.70`, `review_3ds_threshold: 0.30`, `high_velocity_5m_limit: 3`, `high_velocity_60m_limit: 10`) for step-up 3D-Secure verification (`REVIEW_3DS`) vs immediate decline (`DECLINE`).

---

## Specification & Contracts

### 1. Incoming Kafka Event Contract (`tx_incoming`)

Messages published to the `tx_incoming` Kafka topic must conform to the `features.Transaction` JSON contract:

```json
{
  "transaction_id": "a8098c1a-f86e-11da-bd1a-00112444be1e",
  "user_id": "user_2419",
  "amount": 342.50,
  "amount_cents": 34250,
  "currency": "USD",
  "merchant_category": "digital_goods",
  "timestamp": "2026-10-12T14:22:10.120491Z",
  "cvv_match": true,
  "is_3ds_passed": false,
  "billing_country": "US",
  "user_created_at": 1735689600,
  "card_bin": 450521,
  "card_fingerprint": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
}
```

| Field | JSON Type | Go Type | Constraints / Allowed Values |
| :--- | :--- | :--- | :--- |
| `transaction_id` | `string` | `string` | UUIDv4 transaction identifier |
| `user_id` | `string` | `string` | Account identifier; used as Kafka partition key and Redis ZSET key suffix |
| `amount` | `number` | `float64` | Transaction value in major currency units (`> 0.0`) |
| `amount_cents` | `integer` | `int` | Transaction value in minor units (`int(amount * 100)`) |
| `currency` | `string` | `string` | `USD`, `EUR`, `GBP`, `UAH` (other values map to zero OHE vector) |
| `merchant_category` | `string` | `string` | `retail`, `travel`, `digital_goods`, `crypto`, `gambling` |
| `timestamp` | `string` | `string` | ISO-8601 / RFC3339 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`) |
| `cvv_match` | `boolean` | `bool` | Card verification value check result |
| `is_3ds_passed` | `boolean` | `bool` | 3D-Secure authentication result |
| `billing_country` | `string` | `string` | ISO country code (`US`, `CA`, `GB`, `UA`, `EU`) |
| `user_created_at` | `integer` | `int64` | Account creation Unix epoch timestamp in seconds |
| `card_bin` | `integer` | `int` | 6-digit Bank Identification Number (`400000` - `499999`) |
| `card_fingerprint` | `string` | `string` | Tokenized card identifier |

### 2. Online Feature Tensor Contract (`float_input: float32[1, 19]`)

Both `services/risk_engine/internal/features/builder.go` and `ml/train.py` enforce the following 19-element positional feature layout:

| Index | Feature Name | Source / Transformation | Value Domain |
| :--- | :--- | :--- | :--- |
| `0` | `amount` | `float32(tx.Amount)` | `[1.0, 5000.0+]` |
| `1` | `amount_cents` | `float32(tx.AmountCents)` | `[100.0, 500000.0+]` |
| `2` | `cvv_match` | `1.0` if `tx.CVVMatch` else `0.0` | `{0.0, 1.0}` |
| `3` | `is_3ds_passed` | `1.0` if `tx.Is3DSPassed` else `0.0` | `{0.0, 1.0}` |
| `4` | `account_age_days` | `max(0, now - tx.UserCreatedAt) / 86400.0` | `[0.0, +inf)` |
| `5` | `mcc_retail` | `1.0` if `merchant_category == "retail"` | `{0.0, 1.0}` |
| `6` | `mcc_travel` | `1.0` if `merchant_category == "travel"` | `{0.0, 1.0}` |
| `7` | `mcc_digital_goods` | `1.0` if `merchant_category == "digital_goods"` | `{0.0, 1.0}` |
| `8` | `mcc_crypto` | `1.0` if `merchant_category == "crypto"` | `{0.0, 1.0}` |
| `9` | `mcc_gambling` | `1.0` if `merchant_category == "gambling"` | `{0.0, 1.0}` |
| `10` | `curr_usd` | `1.0` if `currency == "USD"` | `{0.0, 1.0}` |
| `11` | `curr_eur` | `1.0` if `currency == "EUR"` | `{0.0, 1.0}` |
| `12` | `curr_gbp` | `1.0` if `currency == "GBP"` | `{0.0, 1.0}` |
| `13` | `curr_uah` | `1.0` if `currency == "UAH"` | `{0.0, 1.0}` |
| `14` | `card_bin` | `float32(tx.CardBin)` | `[400000.0, 599999.0]` |
| `15` | `utc_hour` | `float32(RFC3339(tx.Timestamp).Hour())` (`0.0` fallback) | `[0.0, 23.0]` |
| `16` | `tx_count_1h` | Redis `ZCARD` over trailing 3600s window | `[0.0, +inf)` |
| `17` | `tx_amount_1h` | Redis `ZRANGE` sum of amounts over trailing 3600s window | `[0.0, +inf)` |
| `18` | `padding` | Reserved zero-padding (`0.0`) | `{0.0}` |

### 3. Policy Threshold & Offline Schema Contracts

`ml/model_artifacts/reason_thresholds.json`:

```json
{
  "decline_threshold": 0.70,
  "review_3ds_threshold": 0.30,
  "high_velocity_5m_limit": 3,
  "high_velocity_60m_limit": 10,
  "reasons": {
    "HIGH_VELOCITY_5M": "More than 3 transactions within 5 minutes for this card",
    "HIGH_VELOCITY_60M": "More than 10 transactions within 60 minutes for this card",
    "CVV_MISMATCH": "Card verification value mismatch",
    "NO_3DS_VERIFICATION": "3D-Secure authentication was not performed",
    "HIGH_ML_SCORE": "Machine learning risk score exceeds 0.50"
  }
}
```

### 4. Inference Output & Decision Telemetry Contract

- **ONNX Output Node**: `probabilities` of shape `float32[1, 2]`, where index `0` is $P(\text{legitimate})$ and index `1` is $P(\text{chargeback}) \in [0.0, 1.0]$.
- **Decision Stream Format**:
  ```text
  [DECISION] TX: a8098c1a | User: user_2419 | Score: 0.1428 | Status: APPROVED
  [DECISION] TX: 7c9e6679 | User: user_1042 | Score: 0.9134 | Status: BLOCKED
  ```

---

## Quickstart

### Prerequisites

- Python `3.11+`
- Go `1.22+`
- Docker & Docker Compose
- ONNX Runtime C shared library installed at `/usr/lib/libonnxruntime.so` (for running the Go binary)

### 1. Configure Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Start Backing Infrastructure (Kafka, Redis, PostgreSQL, MLflow)

```bash
make infra-up
# Equivalent to: cd infrastructure && docker-compose up -d
```

Verify container readiness:

```bash
docker ps --filter "name=fluxguard_"
```

### 3. Train Model and Export ONNX Artifact

Train the XGBoost classifier against the MLflow server (`http://localhost:5000`) and export the highest PR-AUC run to `ml/model_artifacts/chargeback_model.onnx`:

```bash
make train
make export
```

### 4. Start the Go Risk Engine

```bash
make run-engine
# Equivalent to: cd services/risk_engine && go run ./cmd/server/main.go
```

### 5. Stream Synthetic Transactions (CLI Validation Example)

In a second terminal with `.venv` activated, start the Kafka transaction producer:

```bash
source .venv/bin/activate
make run-producer
```

Producer output:

```text
Starting transaction generator for topic 'tx_incoming'... (Press Ctrl+C to stop)
[SENT] TX: 4f21c90e... | user_3104 | $412.80 | retail
[SENT] TX: 9a8812d1... | user_1822 | $4210.55 | crypto
```

Risk Engine output:

```text
2026/10/15 11:05:01 Starting FluxGuard Risk Engine. Listening to Kafka topic: tx_incoming
[DECISION] TX: 4f21c90e | User: user_3104 | Score: 0.0842 | Status: APPROVED
[DECISION] TX: 9a8812d1 | User: user_1822 | Score: 0.8941 | Status: BLOCKED
```

To tear down the Docker infrastructure:

```bash
make infra-down
```

---

## Configuration

### Risk Engine Environment Variables (`services/risk_engine/internal/config/config.go`)

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `SERVER_ADDR` | `string` | `:50051` | Service bind address |
| `REDIS_ADDR` | `string` | `localhost:6379` | Redis instance host and port for sliding-window velocity state |
| `MODEL_PATH` | `string` | `model_artifacts/chargeback_model.onnx` | Filesystem path to the compiled ONNX model graph |
| `SCHEMA_PATH` | `string` | `model_artifacts/features_schema.json` | Filesystem path to the feature schema and BIN risk mapping JSON |
| `ONNX_LIB_PATH` | `string` | `/usr/lib/libonnxruntime.so` | Path to the system `libonnxruntime.so` shared library |

### Infrastructure & Runtime Defaults (`infrastructure/docker-compose.yaml` & `cmd/server/main.go`)

| Parameter / Env Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `POSTGRES_USER` | `string` | `fluxguard` | PostgreSQL user for MLflow metadata backend |
| `POSTGRES_PASSWORD` | `string` | `fluxguard_pass` | PostgreSQL password for MLflow metadata backend |
| `POSTGRES_DB` | `string` | `fluxguard_db` | PostgreSQL database name (`5432:5432`) |
| `MLFLOW_BACKEND_STORE_URI` | `string` | `postgresql://fluxguard:fluxguard_pass@postgres:5432/fluxguard_db` | SQLAlchemy URI for MLflow tracking server (`5000:5000`) |
| `MLFLOW_ARTIFACT_ROOT` | `string` | `/mlflow/artifacts` | Container volume path for serialized model artifacts |
| `KAFKA_ADVERTISED_LISTENERS` | `string` | `PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092` | KRaft broker listener endpoints |
| Kafka Topic (`topic`) | `string` | `tx_incoming` | Incoming transaction stream topic consumed by `main.go` |
| Kafka Consumer Group (`groupID`) | `string` | `risk-engine-cg` | Consumer group ID for partition offset management |

---

## Verification & Tests

### 1. Go Feature Engineering Unit Tests

Validates deterministic 19-element feature vector construction, one-hot encoding for merchant categories and currencies, clock-skew handling, and RFC3339 timestamp hour extraction:

```bash
make test-go
# Or directly:
cd services/risk_engine && go test ./... -v
```

### 2. Python ML Pipeline Unit Tests

Validates synthetic dataset dimensionality (`(N, 19)`), binary target distribution (`{0, 1}`), zero-padding invariant on column index `18`, and `XGBClassifier.predict_proba` output tensor shape (`(N, 2)`):

```bash
source .venv/bin/activate
make test-python
# Or directly:
python3 -m unittest ml/test_pipeline.py -v
```

### 3. Go Feature Builder Throughput & Allocation Check

```bash
cd services/risk_engine && go test -run=TestBuildVector -count=100 ./internal/features
```
