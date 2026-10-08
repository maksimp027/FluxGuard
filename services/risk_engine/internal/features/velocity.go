package features

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

type VelocityData struct {
	TxCount1H  float32
	TxAmount1H float32
}

type VelocityEngine struct {
	client *redis.Client
}

func NewVelocityEngine(redisAddr string) *VelocityEngine {
	rdb := redis.NewClient(&redis.Options{
		Addr: redisAddr,
	})
	return &VelocityEngine{
		client: rdb,
	}
}

// GetUserVelocity maintains a 3600-second sliding window per user using a single-RTT Redis ZSET pipeline.
// Expired entries are evicted by Unix timestamp score before computing cardinality and cumulative spend,
// and a 1-hour key TTL bounds memory growth for inactive users.
func (ve *VelocityEngine) GetUserVelocity(ctx context.Context, userID, txID string, amount float64) (VelocityData, error) {
	key := fmt.Sprintf("velocity:user:%s", userID)
	now := time.Now().Unix()
	oneHourAgo := now - 3600

	pipe := ve.client.Pipeline()
	pipe.ZRemRangeByScore(ctx, key, "-inf", strconv.FormatInt(oneHourAgo, 10))
	zRangeCmd := pipe.ZRange(ctx, key, 0, -1)
	zCardCmd := pipe.ZCard(ctx, key)

	member := fmt.Sprintf("%s:%f", txID, amount)
	pipe.ZAdd(ctx, key, redis.Z{
		Score:  float64(now),
		Member: member,
	})
	pipe.Expire(ctx, key, time.Hour)

	_, err := pipe.Exec(ctx)
	if err != nil && err != redis.Nil {
		return VelocityData{}, fmt.Errorf("pipeline execution failed: %w", err)
	}

	members, err := zRangeCmd.Result()
	if err != nil && err != redis.Nil {
		return VelocityData{}, fmt.Errorf("failed to get zrange result: %w", err)
	}

	var sum float64
	for _, m := range members {
		parts := strings.Split(m, ":")
		if len(parts) == 2 {
			amt, parseErr := strconv.ParseFloat(parts[1], 64)
			if parseErr == nil {
				sum += amt
			}
		}
	}

	count, err := zCardCmd.Result()
	if err != nil && err != redis.Nil {
		return VelocityData{}, fmt.Errorf("failed to get zcard result: %w", err)
	}

	return VelocityData{
		TxCount1H:  float32(count),
		TxAmount1H: float32(sum),
	}, nil
}
