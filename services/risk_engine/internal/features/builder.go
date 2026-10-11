package features

import (
	"time"
)

type Transaction struct {
	TransactionID    string  `json:"transaction_id"`
	UserID           string  `json:"user_id"`
	Amount           float64 `json:"amount"`
	AmountCents      int     `json:"amount_cents"`
	Currency         string  `json:"currency"`
	MerchantCategory string  `json:"merchant_category"`
	Timestamp        string  `json:"timestamp"`
	CVVMatch         bool    `json:"cvv_match"`
	Is3DSPassed      bool    `json:"is_3ds_passed"`
	BillingCountry   string  `json:"billing_country"`
	UserCreatedAt    int64   `json:"user_created_at"`
	CardBin          int     `json:"card_bin"`
	CardFingerprint  string  `json:"card_fingerprint"`
}

// BuildVector constructs the 19-element float32 feature vector required by the ONNX model:
// [0:amount, 1:amount_cents, 2:cvv_match, 3:is_3ds_passed, 4:account_age_days,
// 5..9:merchant_ohe, 10..13:currency_ohe, 14:card_bin, 15:utc_hour, 16:tx_cnt_1h, 17:tx_amt_1h, 18:padding].
func BuildVector(tx Transaction, vel VelocityData) []float32 {
	vector := make([]float32, 19)

	vector[0] = float32(tx.Amount)
	vector[1] = float32(tx.AmountCents)

	if tx.CVVMatch {
		vector[2] = 1.0
	}
	if tx.Is3DSPassed {
		vector[3] = 1.0
	}

	now := time.Now().Unix()
	ageSeconds := now - tx.UserCreatedAt
	// Clamp negative deltas caused by producer/consumer clock skew to zero.
	if ageSeconds < 0 {
		ageSeconds = 0
	}
	vector[4] = float32(ageSeconds) / 86400.0

	switch tx.MerchantCategory {
	case "retail":
		vector[5] = 1.0
	case "travel":
		vector[6] = 1.0
	case "digital_goods":
		vector[7] = 1.0
	case "crypto":
		vector[8] = 1.0
	case "gambling":
		vector[9] = 1.0
	}

	switch tx.Currency {
	case "USD":
		vector[10] = 1.0
	case "EUR":
		vector[11] = 1.0
	case "GBP":
		vector[12] = 1.0
	case "UAH":
		vector[13] = 1.0
	}

	vector[14] = float32(tx.CardBin)

	// Fail-safe fallback to 0.0 UTC hour if incoming RFC3339 timestamp is malformed.
	if parsedTime, err := time.Parse(time.RFC3339, tx.Timestamp); err == nil {
		vector[15] = float32(parsedTime.Hour())
	}

	vector[16] = vel.TxCount1H
	vector[17] = vel.TxAmount1H
	vector[18] = 0.0

	return vector
}
