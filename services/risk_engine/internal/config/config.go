package config

import (
	"os"
)

type Config struct {
	ServerAddr  string
	RedisAddr   string
	ModelPath   string
	SchemaPath  string
	ONNXLibPath string
}

func Load() *Config {
	return &Config{
		ServerAddr:  getEnv("SERVER_ADDR", ":50051"),
		RedisAddr:   getEnv("REDIS_ADDR", "localhost:6379"),
		ModelPath:   getEnv("MODEL_PATH", "model_artifacts/chargeback_model.onnx"),
		SchemaPath:  getEnv("SCHEMA_PATH", "model_artifacts/features_schema.json"),
		ONNXLibPath: getEnv("ONNX_LIB_PATH", "/usr/lib/libonnxruntime.so"),
	}
}

func getEnv(key, defaultVal string) string {
	if val, ok := os.LookupEnv(key); ok {
		return val
	}
	return defaultVal
}
