package engine

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

type ScoringEngine struct {
	session *ort.DynamicAdvancedSession
}

func NewScoringEngine(modelPath string) (*ScoringEngine, error) {
	ort.SetSharedLibraryPath("/usr/lib/libonnxruntime.so")
	// Ignore re-initialization errors when sharing the global ORT environment across sessions.
	_ = ort.InitializeEnvironment()

	inputNames := []string{"float_input"}
	outputNames := []string{"probabilities"}

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &ScoringEngine{
		session: session,
	}, nil
}

// Predict executes synchronous ONNX inference over a 19-element float32 feature vector
// and returns P(chargeback) from the [1, 2] class probability output tensor.
func (e *ScoringEngine) Predict(vector []float32) (float32, error) {
	if len(vector) != 19 {
		return 0.0, fmt.Errorf("invalid feature vector length: expected 19, got %d", len(vector))
	}

	inputShape := ort.NewShape(1, 19)
	inputTensor, err := ort.NewTensor(inputShape, vector)
	if err != nil {
		return 0.0, fmt.Errorf("failed to create input tensor: %w", err)
	}
	// Explicitly release C-allocated ORT tensor memory outside the Go garbage collector.
	defer inputTensor.Destroy()

	outputShape := ort.NewShape(1, 2)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return 0.0, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	err = e.session.Run(
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
	)
	if err != nil {
		return 0.0, fmt.Errorf("ONNX inference failed: %w", err)
	}

	results := outputTensor.GetData()
	if len(results) < 2 {
		return 0.0, fmt.Errorf("unexpected output tensor length: got %d", len(results))
	}

	return results[1], nil
}

func (e *ScoringEngine) Close() error {
	if e.session != nil {
		return e.session.Destroy()
	}
	return nil
}
