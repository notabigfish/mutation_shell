from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer
import lmdb

_LMDB_ENVS: dict[tuple[str, bool], lmdb.Environment] = {}


class ESMEmbedder:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def embed_sequence(self, sequence: str) -> torch.FloatTensor:
        encoded = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded)
        last_hidden_state = outputs.last_hidden_state
        length = len(sequence)
        emb = last_hidden_state[0, 1 : length + 1, :]
        if emb.shape[0] != length:
            raise ValueError(
                f"Embedding length mismatch for {self.model_name}: expected {length}, got {emb.shape[0]}"
            )
        return emb.detach().cpu().float()

def open_esm_lmdb(lmdb_path: str | Path, readonly: bool) -> lmdb.Environment:
    lmdb_path = Path(lmdb_path).resolve()
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)

    cache_key = (str(lmdb_path), readonly)
    if cache_key in _LMDB_ENVS:
        return _LMDB_ENVS[cache_key]

    env = lmdb.open(
        str(lmdb_path),
        map_size=1024**4,
        subdir=False,
        readonly=readonly,
        lock=not readonly,
        readahead=readonly,
        meminit=False,
        create=not readonly,
    )
    _LMDB_ENVS[cache_key] = env
    return env

# def save_esm_sample(out_path: Path, pdb_id: str, chain_id: str, sequence: str, embedding: torch.Tensor) -> None:
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     torch.save(
#         {
#             "pdb_id": pdb_id,
#             "chain_id": chain_id,
#             "sequence": sequence,
#             "embedding": embedding,
#         },
#         out_path,
#     )

def save_chain_esm(
    env: lmdb.Environment,
    key: str,
    pdb_id: str,
    chain_id: str,
    sequence: str,
    embedding: torch.Tensor,
) -> None:
    payload = {
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "sequence": sequence,
        "embedding": embedding.to(dtype=torch.float16).contiguous(),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    with env.begin(write=True) as txn:
        txn.put(key.encode("utf-8"), buffer.getvalue())


def load_chain_esm(env: lmdb.Environment, key: str) -> dict[str, Any]:
    with env.begin(write=False) as txn:
        value = txn.get(key.encode("utf-8"))
    if value is None:
        raise FileNotFoundError(f"Missing ESM embedding in LMDB for key: {key}")
    buffer = io.BytesIO(value)
    return torch.load(buffer, map_location="cpu")