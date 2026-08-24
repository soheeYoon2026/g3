#!/usr/bin/env python3
"""Train an experimental PointNet-style surface displacement proposer."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def best_pair_per_job(manifest: dict) -> list[dict]:
    best = {}
    for pair in manifest.get("pairs", []):
        job = pair["job_uid"]
        if job not in best or float(pair["target_cd"]) < float(best[job]["target_cd"]):
            best[job] = pair
    return [best[job] for job in sorted(best)]


def split_jobs(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    test_n = max(1, round(n * 0.1))
    val_n = max(1, round(n * 0.1))
    return {
        "train": shuffled[: n - val_n - test_n],
        "validation": shuffled[n - val_n - test_n : n - test_n],
        "test": shuffled[n - test_n :],
    }


class PairDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], points: int, seed: int):
        self.root, self.rows, self.points, self.seed = root, rows, points, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.root / row["file"]) as data:
            xyz = np.asarray(data["base_points"], np.float32)
            delta = np.asarray(data["displacement"], np.float32)
            moved = np.asarray(data["moved_mask"], np.float32)
        center = (xyz.min(0) + xyz.max(0)) * 0.5
        scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-8)
        rng = np.random.default_rng(self.seed + index)
        chosen = rng.choice(len(xyz), self.points, replace=len(xyz) < self.points)
        return (
            torch.from_numpy((xyz[chosen] - center) / scale),
            torch.from_numpy(delta[chosen] / scale),
            torch.from_numpy(moved[chosen]),
        )


class DeformationProposer(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU())
        self.global_encoder = nn.Sequential(nn.Linear(128, 256), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(128 + 256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.displacement = nn.Linear(128, 3)
        self.moved_logit = nn.Linear(128, 1)

    def forward(self, xyz):
        local = self.local(xyz)
        global_feature = self.global_encoder(local).amax(dim=1, keepdim=True).expand(-1, xyz.shape[1], -1)
        decoded = self.decoder(torch.cat([local, global_feature], dim=-1))
        return self.displacement(decoded), self.moved_logit(decoded).squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); errors, zero_errors, moved_errors, moved_zero_errors, mask_correct, count = [], [], [], [], 0, 0
    for xyz, target, moved in loader:
        xyz, target, moved = xyz.to(device), target.to(device), moved.to(device)
        raw, logits = model(xyz)
        predicted = raw * torch.sigmoid(logits).unsqueeze(-1)
        distance = torch.linalg.norm(predicted - target, dim=-1)
        zero_distance = torch.linalg.norm(target, dim=-1)
        errors.append(distance.mean().item())
        zero_errors.append(zero_distance.mean().item())
        selected = moved > 0.5
        moved_errors.append(distance[selected].mean().item())
        moved_zero_errors.append(zero_distance[selected].mean().item())
        mask_correct += ((logits > 0) == (moved > 0.5)).sum().item()
        count += moved.numel()
    error, zero_error = float(np.mean(errors)), float(np.mean(zero_errors))
    moved_error, moved_zero = float(np.mean(moved_errors)), float(np.mean(moved_zero_errors))
    return {
        "normalized_displacement_mae": error,
        "zero_prediction_mae": zero_error,
        "improvement_over_zero": (zero_error - error) / max(zero_error, 1e-12),
        "moved_point_mae": moved_error,
        "moved_point_zero_mae": moved_zero,
        "moved_point_improvement_over_zero": (moved_zero - moved_error) / max(moved_zero, 1e-12),
        "moved_mask_accuracy": mask_correct / max(count, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    manifest = json.loads((args.data / "manifest.json").read_text())
    rows = best_pair_per_job(manifest)
    split = split_jobs(rows, args.seed)
    loaders = {
        name: DataLoader(PairDataset(args.data, part, args.points, args.seed), batch_size=args.batch_size, shuffle=name == "train")
        for name, part in split.items()
    }
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = DeformationProposer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0, device=device))
    best_state, best_score, history = None, float("inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for xyz, target, moved in loaders["train"]:
            xyz, target, moved = xyz.to(device), target.to(device), moved.to(device)
            predicted, logits = model(xyz)
            per_point = nn.functional.smooth_l1_loss(predicted, target, reduction="none").mean(-1)
            displacement_loss = (per_point * moved).sum() / moved.sum().clamp_min(1)
            loss = displacement_loss * 10 + bce(logits, moved) * 0.1
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        metrics = evaluate(model, loaders["validation"], device)
        score = metrics["normalized_displacement_mae"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        if score < best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    test = evaluate(model, loaders["test"], device)
    metadata = {"format": "g2-deformation-proposer-v1", "examples": len(rows), "split": {k: [r["job_uid"] for r in v] for k, v in split.items()}, "validation_best": best_score, "test": test, "history": history}
    torch.save({"model_state": best_state, "metadata": metadata}, args.out)
    args.out.with_suffix(args.out.suffix + ".metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"saved": str(args.out), "device": str(device), "examples": len(rows), "split_sizes": {k: len(v) for k, v in split.items()}, "test": test}, indent=2))


if __name__ == "__main__":
    main()
