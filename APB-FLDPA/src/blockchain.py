import time
import json
import hashlib


class BlockchainLedger:
    def __init__(self):
        self.chain = []

    def add(self, round_id, model_hash, client_updates, aggregation_hash):
        block = {
            "index": len(self.chain),
            "timestamp": time.time(),
            "round": round_id,
            "model_hash": model_hash,
            "client_updates": client_updates,
            "aggregation_hash": aggregation_hash,
            "previous_hash": self.chain[-1]["hash"] if self.chain else "0",
        }
        block["hash"] = hashlib.sha256(json.dumps(block, sort_keys=True, default=str).encode()).hexdigest()
        self.chain.append(block)
        return block

    def verify(self):
        return all(self.chain[i]["previous_hash"] == self.chain[i - 1]["hash"] for i in range(1, len(self.chain)))
