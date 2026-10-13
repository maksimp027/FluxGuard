package features

import (
	"math"
	"testing"
	"time"
)

func TestBuildVector(t *testing.T) {
	tests := []struct {
		name     string
		tx       Transaction
		vel      VelocityData
		expected []float32
	}{
		{
			name: "Standard USD Retail Transaction",
			tx: Transaction{
				TransactionID:    "tx-123",
				UserID:           "user-1",
				Amount:           125.50,
				AmountCents:      12550,
				Currency:         "USD",
				MerchantCategory: "retail",
				Timestamp:        "2023-10-04T12:00:00Z",
				CVVMatch:         true,
				Is3DSPassed:      true,
				BillingCountry:   "US",
				UserCreatedAt:    time.Now().Unix() - 86400, // 1 day ago
				CardBin:          411111,
				CardFingerprint:  "fingerprint-123",
			},
			vel: VelocityData{
				TxCount1H:  2.0,
				TxAmount1H: 250.0,
			},
			expected: []float32{
				125.50,   // [0] Amount
				12550.0,  // [1] AmountCents
				1.0,      // [2] CVVMatch
				1.0,      // [3] Is3DSPassed
				1.0,      // [4] Account Age (approx 1.0 day)
				1.0,      // [5] OHE retail
				0.0,      // [6] OHE travel
				0.0,      // [7] OHE digital_goods
				0.0,      // [8] OHE crypto
				0.0,      // [9] OHE gambling
				1.0,      // [10] OHE USD
				0.0,      // [11] OHE EUR
				0.0,      // [12] OHE GBP
				0.0,      // [13] OHE UAH
				411111.0, // [14] CardBin
				12.0,     // [15] Hour
				2.0,      // [16] TxCount1H
				250.0,    // [17] TxAmount1H
				0.0,      // [18] Padding
			},
		},
		{
			name: "EUR Crypto Transaction with Failed 3DS",
			tx: Transaction{
				TransactionID:    "tx-456",
				UserID:           "user-2",
				Amount:           1000.00,
				AmountCents:      100000,
				Currency:         "EUR",
				MerchantCategory: "crypto",
				Timestamp:        "2023-10-04T23:30:00Z",
				CVVMatch:         false,
				Is3DSPassed:      false,
				BillingCountry:   "FR",
				UserCreatedAt:    time.Now().Unix() - (86400 * 5), // 5 days ago
				CardBin:          555555,
				CardFingerprint:  "fingerprint-456",
			},
			vel: VelocityData{
				TxCount1H:  5.0,
				TxAmount1H: 5000.0,
			},
			expected: []float32{
				1000.00,  // [0] Amount
				100000.0, // [1] AmountCents
				0.0,      // [2] CVVMatch
				0.0,      // [3] Is3DSPassed
				5.0,      // [4] Account Age (approx 5.0 days)
				0.0,      // [5] OHE retail
				0.0,      // [6] OHE travel
				0.0,      // [7] OHE digital_goods
				1.0,      // [8] OHE crypto
				0.0,      // [9] OHE gambling
				0.0,      // [10] OHE USD
				1.0,      // [11] OHE EUR
				0.0,      // [12] OHE GBP
				0.0,      // [13] OHE UAH
				555555.0, // [14] CardBin
				23.0,     // [15] Hour
				5.0,      // [16] TxCount1H
				5000.0,   // [17] TxAmount1H
				0.0,      // [18] Padding
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BuildVector(tt.tx, tt.vel)

			if len(got) != 19 {
				t.Fatalf("BuildVector() returned slice of length %d, want 19", len(got))
			}

			for i := 0; i < len(tt.expected); i++ {
				// Allow 0.01-day numerical tolerance on index 4 to account for wall-clock delta during test execution.
				if i == 4 {
					diff := math.Abs(float64(got[i] - tt.expected[i]))
					if diff > 0.01 {
						t.Errorf("BuildVector() element %d (Age) = %v, want approx %v", i, got[i], tt.expected[i])
					}
					continue
				}

				if got[i] != tt.expected[i] {
					t.Errorf("BuildVector() element %d = %v, want %v", i, got[i], tt.expected[i])
				}
			}
		})
	}
}
