"""Memory bank of nominal patch features with exact 1-NN cosine distance.

AnomalyDINO uses ``faiss.IndexFlatL2`` on L2-normalised features and reports
``squared_L2 / 2 = 1 − cos`` (src/detection.py:122-178). We compute the identical quantity as
``1 − max_j <q, b_j>`` with a chunked matmul (fp16 on CUDA, fp32 elsewhere); faiss is optional.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

LOG = logging.getLogger(__name__)

try:  # optional backend
    import faiss  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on environment
    faiss = None


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name != "auto":
        raise ValueError(f"unknown dtype {name!r}")
    return torch.float16 if device.type == "cuda" else torch.float32


class PatchBank:
    """Nominal patch memory bank.

    Args:
        device: device holding the bank.
        dtype: matmul dtype (``torch.float16`` or ``torch.float32``).
        backend: ``"torch"`` or ``"faiss"``.
        max_chunk_elems: upper bound on the ``queries × bank`` similarity block held at once.
    """

    def __init__(self, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
                 backend: str = "torch", max_chunk_elems: int = 2**27) -> None:
        if backend not in ("torch", "faiss"):
            raise ValueError(f"backend must be 'torch' or 'faiss', got {backend!r}")
        if backend == "faiss" and faiss is None:
            raise ImportError("knn_backend='faiss' requires `pip install faiss-cpu`")
        self.device = torch.device(device)
        self.dtype = dtype
        self.backend = backend
        self.max_chunk_elems = max_chunk_elems
        self._parts: list[torch.Tensor] = []
        self._bank: torch.Tensor | None = None
        self._index = None

    def __len__(self) -> int:
        return sum(p.shape[0] for p in self._parts) if self._bank is None else self._bank.shape[0]

    def add(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Add patch features ``[..., C]``; ``mask`` (same leading shape, bool) keeps a subset."""
        if self._bank is not None:
            raise RuntimeError("bank already built; create a new PatchBank to add features")
        feats = features.reshape(-1, features.shape[-1])
        if mask is not None:
            keep = torch.as_tensor(mask, device=feats.device).reshape(-1).bool()
            if keep.shape[0] != feats.shape[0]:
                raise ValueError(f"mask has {keep.shape[0]} entries for {feats.shape[0]} patches")
            feats = feats[keep]
        if feats.shape[0]:
            self._parts.append(F.normalize(feats.float(), dim=-1).to(self.device, self.dtype))

    def build(self) -> PatchBank:
        if self._bank is not None:
            return self
        if not self._parts:
            raise ValueError("cannot build an empty PatchBank (no reference patches were added)")
        self._bank = torch.cat(self._parts, dim=0).contiguous()
        self._parts = []
        if self.backend == "faiss":
            bank = self._bank.float().cpu().numpy()
            self._index = faiss.IndexFlatIP(bank.shape[1])
            self._index.add(bank)
        LOG.debug("built PatchBank with %d patches (%s, %s)", self._bank.shape[0], self.device, self.dtype)
        return self

    @property
    def features(self) -> torch.Tensor:
        self.build()
        assert self._bank is not None
        return self._bank

    @torch.inference_mode()
    def knn_distance(self, queries: torch.Tensor) -> torch.Tensor:
        """Cosine distance of each query ``[..., C]`` to its nearest bank patch → ``[...]`` float32."""
        bank = self.features
        lead = queries.shape[:-1]
        q = F.normalize(queries.reshape(-1, queries.shape[-1]).float(), dim=-1)
        if self.backend == "faiss":
            sims, _ = self._index.search(q.cpu().numpy(), 1)
            best = torch.from_numpy(sims[:, 0]).to(queries.device)
            return (1.0 - best).clamp_min_(0.0).reshape(lead)
        q = q.to(self.device, self.dtype)
        n_bank = bank.shape[0]
        bank_chunk = min(n_bank, max(1, self.max_chunk_elems // 4096))
        query_chunk = max(1, self.max_chunk_elems // bank_chunk)
        best = torch.empty(q.shape[0], device=self.device, dtype=torch.float32)
        for qs in range(0, q.shape[0], query_chunk):
            qc = q[qs:qs + query_chunk]
            running = torch.full((qc.shape[0],), -2.0, device=self.device, dtype=torch.float32)
            for bs in range(0, n_bank, bank_chunk):
                sims = qc @ bank[bs:bs + bank_chunk].T
                running = torch.maximum(running, sims.max(dim=1).values.float())
            best[qs:qs + query_chunk] = running
        return (1.0 - best).clamp_min_(0.0).reshape(lead)
