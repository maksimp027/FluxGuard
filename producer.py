import json
import random
import time
import uuid
from datetime import datetime, timedelta
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    key_serializer=lambda k: k.encode("utf-8") if k else None,
)

TOPIC_NAME = "tx_incoming"

print(f"Starting transaction generator for topic '{TOPIC_NAME}'... (Press Ctrl+C to stop)")

try:
    while True:
        user_id = f"user_{random.randint(1000, 5000)}"
        amount = round(random.uniform(1.0, 5000.0), 2)

        tx_data = {
            "transaction_id": str(uuid.uuid4()),
            "user_id": user_id,
            "amount": amount,
            "amount_cents": int(amount * 100),
            "currency": random.choice(["USD", "EUR", "GBP", "UAH"]),
            "merchant_category": random.choice(
                ["retail", "travel", "digital_goods", "crypto", "gambling"]
            ),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "cvv_match": random.choice([True, False]),
            "is_3ds_passed": random.choice([True, False]),
            "billing_country": random.choice(["US", "CA", "GB", "UA", "EU"]),
            "user_created_at": int(
                (datetime.utcnow() - timedelta(days=random.randint(10, 1000))).timestamp()
            ),
            "card_bin": random.randint(400000, 499999),
            "card_fingerprint": str(uuid.uuid4()),
        }

        # Partition by user_id to preserve per-user event ordering for sliding-window velocity state.
        producer.send(TOPIC_NAME, key=user_id, value=tx_data)

        print(
            f"[SENT] TX: {tx_data['transaction_id'][:8]}... | {user_id} | "
            f"${tx_data['amount']} | {tx_data['merchant_category']}"
        )

        time.sleep(random.uniform(0.1, 1.5))

except KeyboardInterrupt:
    print("\nStopping generator...")
finally:
    producer.flush()
    producer.close()