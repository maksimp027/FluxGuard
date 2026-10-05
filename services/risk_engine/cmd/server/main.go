package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/maksimp027/chargeback_predict/internal/engine"
	"github.com/maksimp027/chargeback_predict/internal/features"
	"github.com/segmentio/kafka-go"
)

func main() {
	topic := "tx_incoming"
	brokerAddress := "127.0.0.1:9092"
	groupID := "risk-engine-cg"

	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:  []string{brokerAddress},
		GroupID:  groupID,
		Topic:    topic,
		MaxBytes: 10e6,
	})

	velEngine := features.NewVelocityEngine("127.0.0.1:6379")

	scoringEngine, err := engine.NewScoringEngine("../../ml/model_artifacts/chargeback_model.onnx")
	if err != nil {
		log.Fatalf("Failed to initialize ONNX Scoring Engine: %v", err)
	}
	defer scoringEngine.Close()

	log.Printf("Starting FluxGuard Risk Engine. Listening to Kafka topic: %s", topic)

	sigchan := make(chan os.Signal, 1)
	signal.Notify(sigchan, syscall.SIGINT, syscall.SIGTERM)

	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		<-sigchan
		log.Println("Interrupt detected. Shutting down consumer...")
		cancel()
	}()

	for {
		msg, err := reader.ReadMessage(ctx)
		if err != nil {
			if ctx.Err() != nil {
				break
			}
			log.Printf("Read error: %v", err)
			continue
		}

		var tx features.Transaction
		if err := json.Unmarshal(msg.Value, &tx); err != nil {
			log.Printf("JSON parse error at offset %d: %v", msg.Offset, err)
			continue
		}

		txID := tx.TransactionID
		if len(txID) > 8 {
			txID = txID[:8]
		}

		velData, err := velEngine.GetUserVelocity(ctx, tx.UserID, tx.TransactionID, tx.Amount)
		if err != nil {
			log.Printf("Redis velocity error for User: %s, err: %v", tx.UserID, err)
			continue
		}

		vector := features.BuildVector(tx, velData)

		score, err := scoringEngine.Predict(vector)
		if err != nil {
			log.Printf("Prediction error for TX %s: %v", tx.TransactionID, err)
			continue
		}

		status := "APPROVED"
		if score > 0.85 {
			status = "BLOCKED"
		}

		fmt.Printf("[DECISION] TX: %s | User: %s | Score: %.4f | Status: %s\n",
			txID, tx.UserID, score, status)
	}

	if err := reader.Close(); err != nil {
		log.Fatalf("Reader close error: %v", err)
	}
	log.Println("Shutdown complete.")
}
