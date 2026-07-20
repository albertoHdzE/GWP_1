from __future__ import annotations

import logging
import math
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("gwp3")


@dataclass(frozen=True)
class ProjectConfig:
    symbol: str = "SPY"
    start: str = "2018-01-01"
    end: str = "2026-01-01"
    max_observations: int = 2000
    lookback: int = 32
    horizon: int = 1
    train_frac_single_split: float = 0.7
    seed: int = 7
    epochs_mlp: int = 20
    epochs_lstm: int = 25
    epochs_cnn: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    trading_cost_bps: float = 5.0
    lstm_hidden: int = 64


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _timed(msg: str, **fields: Any) -> tuple[float, str]:
    if fields:
        payload = " ".join(f"{k}={v}" for k, v in fields.items())
        logger.info("%s %s", msg, payload)
    else:
        logger.info("%s", msg)
    return time.perf_counter(), msg


def _timed_done(t0: float, msg: str, **fields: Any) -> None:
    elapsed_s = time.perf_counter() - t0
    if fields:
        payload = " ".join(f"{k}={v}" for k, v in fields.items())
        logger.info("%s done elapsed_s=%.3f %s", msg, elapsed_s, payload)
    else:
        logger.info("%s done elapsed_s=%.3f", msg, elapsed_s)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_run_artifacts(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: ProjectConfig,
    images_dir: Path,
) -> None:
    t0, msg = _timed("save artifacts", out_dir=str(images_dir))
    ensure_dir(images_dir)
    prices.to_csv(images_dir / "data_prices.csv")
    features.to_csv(images_dir / "data_features.csv")
    labels.to_csv(images_dir / "data_labels.csv")
    (images_dir / "run_config.json").write_text(json.dumps(cfg.__dict__, indent=2, sort_keys=True), encoding="utf-8")
    _timed_done(t0, msg)


def download_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    t0, msg = _timed("download prices", symbol=symbol, start=start, end=end)
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for symbol={symbol!r}")
    df = df.rename_axis("date").reset_index()
    df["date"] = pd.to_datetime(df["date"], utc=False)
    df = df.sort_values("date").set_index("date")
    keep_cols = [c for c in ["Close", "Volume"] if c in df.columns]
    df = df[keep_cols].copy()
    df = df.dropna()
    _timed_done(t0, msg, rows=len(df), cols=len(df.columns))
    return df


def trim_to_max_observations(df: pd.DataFrame, max_observations: int) -> pd.DataFrame:
    if len(df) <= max_observations:
        return df
    return df.iloc[-max_observations:].copy()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0.0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    t0, msg = _timed("build features", rows=len(df), cols=len(df.columns))
    close = df["Close"]
    ret = np.log(close).diff()
    macd, macd_sig = compute_macd(close)
    rsi14 = compute_rsi(close, period=14)

    feats = pd.DataFrame(index=df.index)
    feats["ret"] = ret
    feats["ret_mean_5"] = ret.rolling(5).mean()
    feats["ret_mean_10"] = ret.rolling(10).mean()
    feats["ret_mean_20"] = ret.rolling(20).mean()
    feats["ret_std_5"] = ret.rolling(5).std()
    feats["ret_std_10"] = ret.rolling(10).std()
    feats["ret_std_20"] = ret.rolling(20).std()
    feats["rsi_14"] = rsi14
    feats["macd"] = macd
    feats["macd_signal"] = macd_sig
    if "Volume" in df.columns:
        vol = df["Volume"].replace(0.0, np.nan)
        feats["log_volume"] = np.log(vol)
        feats["log_volume_z20"] = (feats["log_volume"] - feats["log_volume"].rolling(20).mean()) / feats["log_volume"].rolling(20).std()
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    _timed_done(t0, msg, rows=len(feats), cols=len(feats.columns))
    return feats


def build_labels_from_returns(features: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    t0, msg = _timed("build labels", rows=len(features), horizon=horizon)
    ret = features["ret"]
    fwd_ret = ret.shift(-horizon)
    y_class = (fwd_ret > 0.0).astype(int)
    out = pd.DataFrame(index=features.index)
    out["fwd_ret"] = fwd_ret
    out["y"] = y_class
    out = out.dropna()
    _timed_done(t0, msg, rows=len(out), pos_rate=float(out["y"].mean()) if len(out) else float("nan"))
    return out


def make_supervised_sequences(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    lookback: int,
    *,
    start_idx: int,
    end_idx_exclusive: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t0, msg = _timed("make sequences", lookback=lookback, start=start_idx, end=end_idx_exclusive)
    idx = features.index
    if not labels.index.equals(idx):
        labels = labels.reindex(idx)
    valid = labels["y"].notna()
    valid_positions = np.where(valid.values)[0]
    valid_positions = valid_positions[(valid_positions >= start_idx) & (valid_positions < end_idx_exclusive)]

    xs: list[np.ndarray] = []
    ys: list[int] = []
    fwd_rets: list[float] = []
    for pos in valid_positions:
        lb_start = pos - lookback + 1
        if lb_start < start_idx:
            continue
        window = features.iloc[lb_start : pos + 1].to_numpy(dtype=np.float32)
        if window.shape[0] != lookback:
            continue
        xs.append(window)
        ys.append(int(labels.iloc[pos]["y"]))
        fwd_rets.append(float(labels.iloc[pos]["fwd_ret"]))
    if not xs:
        _timed_done(t0, msg, n_samples=0)
        return np.empty((0, lookback, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    x = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    r = np.asarray(fwd_rets, dtype=np.float32)
    _timed_done(t0, msg, n_samples=int(x.shape[0]), n_features=int(x.shape[2]))
    return x, y, r


def gasf_image(series: np.ndarray) -> np.ndarray:
    s = np.asarray(series, dtype=np.float32)
    s_min = float(np.min(s))
    s_max = float(np.max(s))
    denom = s_max - s_min
    if denom == 0.0:
        s_scaled = np.zeros_like(s, dtype=np.float32)
    else:
        s_scaled = 2.0 * ((s - s_min) / denom) - 1.0
        s_scaled = np.clip(s_scaled, -1.0, 1.0)
    phi = np.arccos(s_scaled)
    gaf = np.cos(phi[:, None] + phi[None, :]).astype(np.float32)
    return gaf


def gasf_image_with_global_minmax(series: np.ndarray, global_min: float, global_max: float) -> np.ndarray:
    s = np.asarray(series, dtype=np.float32)
    denom = global_max - global_min
    if denom == 0.0:
        s_scaled = np.zeros_like(s, dtype=np.float32)
    else:
        s_scaled = 2.0 * ((s - global_min) / denom) - 1.0
        s_scaled = np.clip(s_scaled, -1.0, 1.0)
    phi = np.arccos(s_scaled)
    gaf = np.cos(phi[:, None] + phi[None, :]).astype(np.float32)
    return gaf


class MLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class GAFConvNet(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        h = image_size // 4
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * h * h, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return self.head(z).squeeze(-1)


def _make_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _train_binary_classifier(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> nn.Module:
    t0 = time.perf_counter()
    device = _make_device()
    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state: dict[str, Any] | None = None
    best_val = float("inf")
    patience = 5
    bad = 0

    n = x_train.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = x_train[idx]
            yb = y_train[idx].float()
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = loss_fn(val_logits, y_val.float()).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed_s = time.perf_counter() - t0
    logger.info(
        "train done arch=%s device=%s n_train=%d n_val=%d best_val=%.6f elapsed_s=%.3f",
        model.__class__.__name__,
        device.type,
        int(x_train.shape[0]),
        int(x_val.shape[0]),
        float(best_val),
        float(elapsed_s),
    )
    return model


def _predict_proba(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    t0 = time.perf_counter()
    device = _make_device()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device)).detach().cpu().numpy()
    proba = 1.0 / (1.0 + np.exp(-logits))
    elapsed_s = time.perf_counter() - t0
    logger.info("predict done arch=%s device=%s n=%d elapsed_s=%.3f", model.__class__.__name__, device.type, int(proba.shape[0]), float(elapsed_s))
    return proba.astype(np.float64)


def split_train_val_chrono(x: np.ndarray, y: np.ndarray, val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    cut = max(1, int(math.floor(n * (1.0 - val_frac))))
    x_train = x[:cut]
    y_train = y[:cut]
    x_val = x[cut:]
    y_val = y[cut:]
    return x_train, y_train, x_val, y_val


def compute_strategy_returns_from_proba(fwd_returns: np.ndarray, proba_up: np.ndarray, trading_cost_bps: float) -> pd.Series:
    pos = np.where(proba_up >= 0.5, 1.0, -1.0)
    pos_prev = np.roll(pos, 1)
    pos_prev[0] = 0.0
    turnover = np.abs(pos - pos_prev)
    cost = (trading_cost_bps / 1e4) * turnover
    strat = pos * fwd_returns - cost
    return pd.Series(strat)


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def annualized_sharpe(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna().to_numpy(dtype=float)
    if r.size < 2:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0.0:
        return float("nan")
    return float((mu / sd) * math.sqrt(periods_per_year))


def summarize_backtest(daily_returns: pd.Series) -> dict[str, float]:
    eq = equity_curve(daily_returns)
    total = float(eq.iloc[-1] - 1.0) if len(eq) else float("nan")
    mdd = max_drawdown(eq) if len(eq) else float("nan")
    sharpe = annualized_sharpe(daily_returns)
    hit = float((daily_returns > 0).mean()) if len(daily_returns) else float("nan")
    return {"total_return": total, "max_drawdown": mdd, "sharpe": sharpe, "hit_rate": hit}


def save_price_and_return_plots(prices: pd.DataFrame, features: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prices.index, prices["Close"])
    ax.set_title("Price (Close, auto-adjusted)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    fig.tight_layout()
    fig.savefig(out_dir / "data_price.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(features.index, features["ret"])
    ax.set_title("Log Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    fig.tight_layout()
    fig.savefig(out_dir / "data_returns.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(features["ret"].dropna().values, bins=60, density=True)
    ax.set_title("Return Distribution")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(out_dir / "data_return_hist.png", dpi=200)
    plt.close(fig)


def save_equity_plot(daily_returns: pd.Series, title: str, out_path: Path) -> None:
    eq = equity_curve(daily_returns)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq.index, eq.values)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Equity (start=1.0)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def train_predict_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    cfg: ProjectConfig,
    *,
    save_path: Path | None = None,
    save_payload: dict[str, Any] | None = None,
) -> np.ndarray:
    t0, msg = _timed("train+predict", arch="mlp", n_train=int(x_train.shape[0]), n_test=int(x_test.shape[0]))
    x_train_f = x_train.reshape(x_train.shape[0], -1)
    x_test_f = x_test.reshape(x_test.shape[0], -1)
    x_tr, y_tr, x_val, y_val = split_train_val_chrono(x_train_f, y_train)
    model = MLP(in_dim=x_tr.shape[1])
    model = _train_binary_classifier(
        model,
        _to_torch(x_tr),
        torch.tensor(y_tr, dtype=torch.int64),
        _to_torch(x_val),
        torch.tensor(y_val, dtype=torch.int64),
        epochs=cfg.epochs_mlp,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    if save_path is not None:
        ensure_dir(save_path.parent)
        model = model.to(torch.device("cpu"))
        payload = {} if save_payload is None else dict(save_payload)
        payload.update({"arch": "mlp", "in_dim": int(x_tr.shape[1]), "lookback": int(cfg.lookback)})
        torch.save({"state_dict": model.state_dict(), **payload}, save_path)
        logger.info("model saved arch=mlp path=%s", str(save_path))
    out = _predict_proba(model, _to_torch(x_test_f))
    _timed_done(t0, msg)
    return out


def train_predict_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    cfg: ProjectConfig,
    *,
    save_path: Path | None = None,
    save_payload: dict[str, Any] | None = None,
) -> np.ndarray:
    t0, msg = _timed("train+predict", arch="lstm", n_train=int(x_train.shape[0]), n_test=int(x_test.shape[0]))
    x_tr, y_tr, x_val, y_val = split_train_val_chrono(x_train, y_train)
    model = LSTMClassifier(n_features=x_tr.shape[2], hidden=cfg.lstm_hidden)
    model = _train_binary_classifier(
        model,
        _to_torch(x_tr),
        torch.tensor(y_tr, dtype=torch.int64),
        _to_torch(x_val),
        torch.tensor(y_val, dtype=torch.int64),
        epochs=cfg.epochs_lstm,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    if save_path is not None:
        ensure_dir(save_path.parent)
        model = model.to(torch.device("cpu"))
        payload = {} if save_payload is None else dict(save_payload)
        payload.update(
            {
                "arch": "lstm",
                "n_features": int(x_tr.shape[2]),
                "hidden": int(cfg.lstm_hidden),
                "lookback": int(cfg.lookback),
            }
        )
        torch.save({"state_dict": model.state_dict(), **payload}, save_path)
        logger.info("model saved arch=lstm path=%s", str(save_path))
    out = _predict_proba(model, _to_torch(x_test))
    _timed_done(t0, msg)
    return out


def train_predict_cnn_gaf(
    returns_windows_train: np.ndarray,
    y_train: np.ndarray,
    returns_windows_test: np.ndarray,
    cfg: ProjectConfig,
    *,
    global_min: float | None = None,
    global_max: float | None = None,
    save_path: Path | None = None,
    save_payload: dict[str, Any] | None = None,
) -> np.ndarray:
    t0, msg = _timed(
        "train+predict",
        arch="cnn_gaf",
        n_train=int(returns_windows_train.shape[0]),
        n_test=int(returns_windows_test.shape[0]),
        leakage_norm="global" if (global_min is not None and global_max is not None) else "fold",
    )
    def make_images(windows: np.ndarray) -> np.ndarray:
        imgs = []
        for w in windows:
            if global_min is None or global_max is None:
                img = gasf_image(w)
            else:
                img = gasf_image_with_global_minmax(w, global_min=global_min, global_max=global_max)
            imgs.append(img)
        x = np.stack(imgs, axis=0)
        return x[:, None, :, :]

    x_train_img = make_images(returns_windows_train)
    x_test_img = make_images(returns_windows_test)
    x_tr, y_tr, x_val, y_val = split_train_val_chrono(x_train_img, y_train)
    model = GAFConvNet(image_size=x_tr.shape[-1])
    model = _train_binary_classifier(
        model,
        _to_torch(x_tr),
        torch.tensor(y_tr, dtype=torch.int64),
        _to_torch(x_val),
        torch.tensor(y_val, dtype=torch.int64),
        epochs=cfg.epochs_cnn,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    if save_path is not None:
        ensure_dir(save_path.parent)
        model = model.to(torch.device("cpu"))
        payload = {} if save_payload is None else dict(save_payload)
        payload.update(
            {
                "arch": "cnn_gaf",
                "image_size": int(x_tr.shape[-1]),
                "lookback": int(cfg.lookback),
                "global_min": None if global_min is None else float(global_min),
                "global_max": None if global_max is None else float(global_max),
            }
        )
        torch.save({"state_dict": model.state_dict(), **payload}, save_path)
        logger.info("model saved arch=cnn_gaf path=%s", str(save_path))
    out = _predict_proba(model, _to_torch(x_test_img))
    _timed_done(t0, msg)
    return out


def single_split_indices(n: int, train_frac: float) -> tuple[int, int]:
    train_end = max(1, int(math.floor(n * train_frac)))
    return 0, train_end


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y_pred = (proba >= 0.5).astype(int)
    out: dict[str, float] = {"accuracy": float(accuracy_score(y_true, y_pred))}
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    else:
        out["roc_auc"] = float("nan")
    return out


def _fit_global_scaler_and_transform(features: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    x = scaler.fit_transform(features.values)
    scaled = pd.DataFrame(x, index=features.index, columns=features.columns)
    return scaled, scaler


def _transform_with_scaler(features: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    x = scaler.transform(features.values)
    return pd.DataFrame(x, index=features.index, columns=features.columns)


def run_step1(prices: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame, cfg: ProjectConfig, images_dir: Path) -> pd.DataFrame:
    t0, msg = _timed("step1 start", rows=int(len(features)))
    n = len(features)
    start, train_end = single_split_indices(n, cfg.train_frac_single_split)

    scaled_features, scaler = _fit_global_scaler_and_transform(features)
    x_train, y_train, r_train = make_supervised_sequences(scaled_features, labels, cfg.lookback, start_idx=start, end_idx_exclusive=train_end)
    x_test, y_test, r_test = make_supervised_sequences(scaled_features, labels, cfg.lookback, start_idx=train_end, end_idx_exclusive=n)
    logger.info("step1 split n=%d train_end=%d x_train=%d x_test=%d", int(n), int(train_end), int(x_train.shape[0]), int(x_test.shape[0]))

    ret_series = features["ret"].to_numpy(dtype=np.float32)
    global_min = float(np.min(ret_series))
    global_max = float(np.max(ret_series))
    w_train = x_train[:, :, list(features.columns).index("ret")]
    w_test = x_test[:, :, list(features.columns).index("ret")]

    model_dir = images_dir / "models"
    payload_common: dict[str, Any] = {
        "step": "1",
        "symbol": cfg.symbol,
        "feature_names": list(features.columns),
        "label": "y = 1[fwd_ret > 0]",
        "horizon": int(cfg.horizon),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
    }

    proba_mlp = train_predict_mlp(
        x_train,
        y_train,
        x_test,
        cfg,
        save_path=model_dir / "step1_mlp.pt",
        save_payload=payload_common,
    )
    proba_lstm = train_predict_lstm(
        x_train,
        y_train,
        x_test,
        cfg,
        save_path=model_dir / "step1_lstm.pt",
        save_payload=payload_common,
    )
    proba_cnn = train_predict_cnn_gaf(
        w_train,
        y_train,
        w_test,
        cfg,
        global_min=global_min,
        global_max=global_max,
        save_path=model_dir / "step1_cnn_gaf.pt",
        save_payload=payload_common,
    )
    logger.info("step1 inference complete")

    res = []
    for name, proba in [("mlp", proba_mlp), ("lstm", proba_lstm), ("cnn_gaf", proba_cnn)]:
        metrics = evaluate_predictions(y_test, proba)
        strat_rets = compute_strategy_returns_from_proba(r_test, proba, cfg.trading_cost_bps)
        summary = summarize_backtest(strat_rets)
        save_equity_plot(strat_rets, f"Step 1 Equity Curve ({name})", images_dir / f"step1_equity_{name}.png")
        res.append(
            {
                "step": "1",
                "model": name,
                **metrics,
                **summary,
            }
        )
    save_price_and_return_plots(prices, features, images_dir)
    _timed_done(t0, msg, models=3)
    return pd.DataFrame(res)


def walk_forward_splits(n: int, train_size: int, test_size: int) -> list[tuple[int, int, int, int]]:
    out = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size
        if test_end > n:
            break
        out.append((train_start, train_end, test_start, test_end))
        start = test_start
    return out


def _wf_run(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: ProjectConfig,
    images_dir: Path,
    *,
    tag: str,
    train_size: int,
    test_size: int,
    leakage_mode: Literal["leaky", "reduced"],
) -> pd.DataFrame:
    t0, msg = _timed("walk-forward start", tag=tag, train_size=train_size, test_size=test_size, leakage=leakage_mode)
    n = len(features)
    splits = walk_forward_splits(n, train_size=train_size, test_size=test_size)
    if not splits:
        raise RuntimeError("No walk-forward splits produced; check series length vs split sizes.")
    logger.info("walk-forward splits tag=%s n=%d n_splits=%d", tag, int(n), int(len(splits)))

    if leakage_mode == "leaky":
        scaled_features, _ = _fit_global_scaler_and_transform(features)
        ret_series = features["ret"].to_numpy(dtype=np.float32)
        global_min = float(np.min(ret_series))
        global_max = float(np.max(ret_series))

    model_returns: dict[str, list[pd.Series]] = {"mlp": [], "lstm": [], "cnn_gaf": []}
    model_y_true: dict[str, list[np.ndarray]] = {"mlp": [], "lstm": [], "cnn_gaf": []}
    model_proba: dict[str, list[np.ndarray]] = {"mlp": [], "lstm": [], "cnn_gaf": []}

    ret_col = list(features.columns).index("ret")

    used_folds = 0
    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(splits, start=1):
        fold_t0 = time.perf_counter()
        if leakage_mode == "leaky":
            feat_fold = scaled_features
            w_global_min = global_min
            w_global_max = global_max
            test_lb_start = test_start - cfg.lookback + 1
            test_start_seq = test_start
        else:
            scaler = StandardScaler()
            scaler.fit(features.iloc[train_start:train_end].values)
            feat_scaled = _transform_with_scaler(features, scaler)
            feat_fold = feat_scaled
            train_end_adj = max(train_start, train_end - cfg.horizon)
            test_start_strict = test_start + cfg.lookback
            test_start_seq = test_start_strict
            test_lb_start = test_start_strict - cfg.lookback + 1
            if test_start_seq >= test_end:
                continue
            train_end = train_end_adj
            ret_train = features.iloc[train_start:train_end]["ret"].to_numpy(dtype=np.float32)
            w_global_min = float(np.min(ret_train))
            w_global_max = float(np.max(ret_train))

        x_train, y_train, _ = make_supervised_sequences(feat_fold, labels, cfg.lookback, start_idx=train_start, end_idx_exclusive=train_end)
        x_test, y_test, r_test = make_supervised_sequences(feat_fold, labels, cfg.lookback, start_idx=test_start_seq, end_idx_exclusive=test_end)
        if x_train.shape[0] < 50 or x_test.shape[0] < 10:
            logger.info(
                "wf fold skip tag=%s fold=%d train=[%d,%d) test=[%d,%d) x_train=%d x_test=%d",
                tag,
                int(fold_idx),
                int(train_start),
                int(train_end),
                int(test_start_seq),
                int(test_end),
                int(x_train.shape[0]),
                int(x_test.shape[0]),
            )
            continue
        logger.info(
            "wf fold start tag=%s fold=%d train=[%d,%d) test=[%d,%d) x_train=%d x_test=%d",
            tag,
            int(fold_idx),
            int(train_start),
            int(train_end),
            int(test_start_seq),
            int(test_end),
            int(x_train.shape[0]),
            int(x_test.shape[0]),
        )

        w_train = x_train[:, :, ret_col]
        w_test = x_test[:, :, ret_col]

        proba_mlp = train_predict_mlp(x_train, y_train, x_test, cfg)
        proba_lstm = train_predict_lstm(x_train, y_train, x_test, cfg)
        proba_cnn = train_predict_cnn_gaf(w_train, y_train, w_test, cfg, global_min=w_global_min, global_max=w_global_max)

        for name, proba in [("mlp", proba_mlp), ("lstm", proba_lstm), ("cnn_gaf", proba_cnn)]:
            strat_rets = compute_strategy_returns_from_proba(r_test, proba, cfg.trading_cost_bps)
            model_returns[name].append(strat_rets)
            model_y_true[name].append(y_test)
            model_proba[name].append(proba)
        used_folds += 1
        logger.info("wf fold done tag=%s fold=%d elapsed_s=%.3f", tag, int(fold_idx), float(time.perf_counter() - fold_t0))

    rows = []
    for name in ["mlp", "lstm", "cnn_gaf"]:
        if not model_returns[name]:
            continue
        all_rets = pd.concat(model_returns[name], ignore_index=True)
        all_y = np.concatenate(model_y_true[name], axis=0)
        all_p = np.concatenate(model_proba[name], axis=0)
        metrics = evaluate_predictions(all_y, all_p)
        summary = summarize_backtest(all_rets)
        save_equity_plot(all_rets, f"Step {tag} Equity Curve ({name})", images_dir / f"step{tag}_equity_{name}.png")
        rows.append({"step": tag, "model": name, **metrics, **summary})
    _timed_done(t0, msg, used_folds=int(used_folds))
    return pd.DataFrame(rows)


def run_step2(prices: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame, cfg: ProjectConfig, images_dir: Path) -> pd.DataFrame:
    a = _wf_run(prices, features, labels, cfg, images_dir, tag="2a", train_size=500, test_size=500, leakage_mode="leaky")
    b = _wf_run(prices, features, labels, cfg, images_dir, tag="2b", train_size=500, test_size=100, leakage_mode="leaky")
    return pd.concat([a, b], ignore_index=True)


def run_step3(prices: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame, cfg: ProjectConfig, images_dir: Path) -> pd.DataFrame:
    b = _wf_run(prices, features, labels, cfg, images_dir, tag="3b", train_size=500, test_size=500, leakage_mode="reduced")
    c = _wf_run(prices, features, labels, cfg, images_dir, tag="3c", train_size=500, test_size=100, leakage_mode="reduced")
    return pd.concat([b, c], ignore_index=True)


def run_all(cfg: ProjectConfig, *, base_dir: Path) -> pd.DataFrame:
    configure_logging()
    t0, msg = _timed("run start", symbol=cfg.symbol, start=cfg.start, end=cfg.end, lookback=cfg.lookback, horizon=cfg.horizon, seed=cfg.seed)
    set_determinism(cfg.seed)
    images_dir = base_dir / "report" / "images"
    ensure_dir(images_dir)

    prices = download_prices(cfg.symbol, start=cfg.start, end=cfg.end)
    prices = trim_to_max_observations(prices, cfg.max_observations)
    features = build_features(prices)
    features = trim_to_max_observations(features, cfg.max_observations)
    labels = build_labels_from_returns(features, horizon=cfg.horizon)
    labels = labels.reindex(features.index).dropna()

    prices = prices.reindex(features.index).dropna()
    save_run_artifacts(prices, features, labels, cfg, images_dir)

    logger.info("phase start step=1")
    step1 = run_step1(prices, features, labels, cfg, images_dir)
    logger.info("phase done step=1")
    logger.info("phase start step=2")
    step2 = run_step2(prices, features, labels, cfg, images_dir)
    logger.info("phase done step=2")
    logger.info("phase start step=3")
    step3 = run_step3(prices, features, labels, cfg, images_dir)
    logger.info("phase done step=3")

    results = pd.concat([step1, step2, step3], ignore_index=True)
    results.to_csv(base_dir / "report" / "images" / "metrics_summary.csv", index=False)
    write_metrics_table_tex(results, images_dir / "metrics_summary_table.tex")
    write_report_tex_blocks(prices, features, labels, cfg, results, images_dir)
    _timed_done(t0, msg, rows=int(len(results)))
    return results


def write_metrics_table_tex(results: pd.DataFrame, out_path: Path) -> None:
    cols = ["step", "model", "accuracy", "roc_auc", "total_return", "max_drawdown", "sharpe", "hit_rate"]
    df = results.copy()
    df = df[cols]
    df = df.sort_values(["step", "model"])
    fmt = df.copy()
    for c in ["accuracy", "roc_auc", "total_return", "max_drawdown", "sharpe", "hit_rate"]:
        fmt[c] = fmt[c].astype(float).map(lambda x: f"{x:.3f}" if np.isfinite(x) else "")
    latex_tabular = fmt.to_latex(index=False, escape=True, column_format="llrrrrrr", longtable=False)
    latex = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\caption{Model performance and backtest summary across all steps.}",
            "\\label{tab:metrics}",
            latex_tabular,
            "\\end{table}",
            "",
        ]
    )
    out_path.write_text(latex, encoding="utf-8")


def _latex_escape(text: str) -> str:
    repl = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def write_report_tex_blocks(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: ProjectConfig,
    results: pd.DataFrame,
    images_dir: Path,
) -> None:
    ensure_dir(images_dir)

    start_date = prices.index.min().date()
    end_date = prices.index.max().date()
    n_obs = int(len(prices))
    feature_list = ", ".join(_latex_escape(c) for c in features.columns)
    pos_rate = float(labels["y"].mean()) if "y" in labels.columns and len(labels) else float("nan")

    step1 = results[results["step"] == "1"].copy()
    step2a = results[results["step"] == "2a"].copy()
    step2b = results[results["step"] == "2b"].copy()
    step3b = results[results["step"] == "3b"].copy()
    step3c = results[results["step"] == "3c"].copy()

    def fmt_row(df: pd.DataFrame, model: str) -> dict[str, float]:
        row = df[df["model"] == model].iloc[0].to_dict()
        return {k: float(row[k]) for k in ["accuracy", "roc_auc", "total_return", "max_drawdown", "sharpe", "hit_rate"] if k in row}

    def fmt_metrics(df: pd.DataFrame, model: str) -> str:
        r = fmt_row(df, model)
        return (
            f"Accuracy={r.get('accuracy', float('nan')):.3f}, AUC={r.get('roc_auc', float('nan')):.3f}, "
            f"TotalReturn={r.get('total_return', float('nan')):.3f}, MaxDD={r.get('max_drawdown', float('nan')):.3f}, "
            f"Sharpe={r.get('sharpe', float('nan')):.3f}"
        )

    def fmt_key(df: pd.DataFrame, model: str, key: str) -> float:
        if df.empty:
            return float("nan")
        sub = df[df["model"] == model]
        if sub.empty:
            return float("nan")
        return float(sub.iloc[0][key])

    step1a_table = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{p{0.34\\linewidth} p{0.60\\linewidth}}",
            "\\toprule",
            "\\textbf{Item} & \\textbf{Value}\\\\",
            "\\midrule",
            f"Security & \\textbf{{{_latex_escape(cfg.symbol)}}}\\\\",
            "Data source & Yahoo Finance via \\texttt{yfinance} (auto-adjusted)\\\\",
            "Frequency & Daily\\\\",
            f"Sample & {start_date} -- {end_date}\\\\",
            f"Observations & {n_obs}\\\\",
            "Project constraint & $\\leq 2000$ observations (satisfied)\\\\",
            f"Lookback window & {cfg.lookback} observations\\\\",
            f"Label horizon & {cfg.horizon} observation(s) ahead\\\\",
            f"Target definition & $y_t=\\mathbb{{1}}[r_{{t+{cfg.horizon}}}>0]$; class-1 rate={pos_rate:.3f}\\\\",
            f"Feature set & {feature_list}\\\\",
            f"Trading cost assumption & {cfg.trading_cost_bps:.1f} bps per unit turnover\\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    (images_dir / "step1a.tex").write_text(step1a_table, encoding="utf-8")

    step1b_text = "\n".join(
        [
            "We intentionally construct a leaky modeling setup to illustrate backtest overfitting due to information leakage.",
            "Specifically, we standardize features using a \\textbf{single scaler fit on the full dataset} (train+test) before splitting, so statistics from the future test period influence the training representation.",
            "For the CNN+GAF model, we also normalize returns using global min/max over the full sample when mapping windows into Gramian Angular Field images; this similarly contaminates the feature construction with future information.",
            "This leakage can inflate apparent predictive accuracy and trading performance because the model indirectly benefits from knowledge of the test distribution, a common pitfall in financial ML research and backtesting (L\\'opez de Prado; Bailey et al.).",
            "",
        ]
    )
    (images_dir / "step1b.tex").write_text(step1b_text, encoding="utf-8")

    step1c_text = "\n".join(
        [
            "We use a single chronological train/test split and train three deep learning classifiers:",
            "\\begin{itemize}",
            "\\item \\textbf{MLP}: a feed-forward network on flattened lookback windows.",
            f"\\item \\textbf{{LSTM}}: a single-layer LSTM with hidden size {cfg.lstm_hidden} followed by a small dense head (Hochreiter and Schmidhuber).",
            "\\item \\textbf{CNN on GAF}: a small 2-layer CNN trained on GAF images constructed from return windows (Wang and Oates) and a convolutional architecture (LeCun et al.).",
            "\\end{itemize}",
            f"All models use lookback length {cfg.lookback}, AdamW optimization (lr={cfg.lr}, weight decay={cfg.weight_decay}), and early stopping on a chronological validation split inside the training window.",
            "Prediction performance on the held-out test period is summarized in Table~\\ref{tab:metrics}.",
            "",
        ]
    )
    (images_dir / "step1c.tex").write_text(step1c_text, encoding="utf-8")

    step1d_header = "\n".join(
        [
            "Trading rule: go long when $\\hat p(\\text{up})\\ge 0.5$, otherwise short; transaction costs "
            f"are {cfg.trading_cost_bps:.1f} bps per unit turnover.",
            "",
        ]
    )
    step1d_rows = []
    for model in ["mlp", "lstm", "cnn_gaf"]:
        step1d_rows.append(
            " & ".join(
                [
                    _latex_escape(model),
                    f"{fmt_key(step1, model, 'accuracy'):.3f}",
                    f"{fmt_key(step1, model, 'roc_auc'):.3f}",
                    f"{fmt_key(step1, model, 'sharpe'):.3f}",
                    f"{fmt_key(step1, model, 'total_return'):.3f}",
                    f"{fmt_key(step1, model, 'max_drawdown'):.3f}",
                    f"{fmt_key(step1, model, 'hit_rate'):.3f}",
                ]
            )
            + " \\\\"
        )
    step1d_table = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Model & Acc & AUC & Sharpe & TotalRet & MaxDD & HitRate\\\\",
            "\\midrule",
            *step1d_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    (images_dir / "step1d.tex").write_text(step1d_header + step1d_table, encoding="utf-8")

    step2c_intro = "\n".join(
        [
            "Walk-forward backtests (non-anchored) are evaluated under two regimes: 500/500 (Step 2a) and 500/100 (Step 2b). "
            "Step 2 keeps leaky preprocessing (global scaling), which can increase the probability of selecting or reporting over-optimistic backtests (L\\'opez de Prado; Bailey et al.).",
            "",
        ]
    )
    step2c_rows = []
    for model in ["mlp", "lstm", "cnn_gaf"]:
        step2c_rows.append(
            " & ".join(
                [
                    _latex_escape(model),
                    f"{fmt_key(step1, model, 'sharpe'):.3f}",
                    f"{fmt_key(step2a, model, 'sharpe'):.3f}",
                    f"{fmt_key(step2b, model, 'sharpe'):.3f}",
                    f"{fmt_key(step1, model, 'total_return'):.3f}",
                    f"{fmt_key(step2a, model, 'total_return'):.3f}",
                    f"{fmt_key(step2b, model, 'total_return'):.3f}",
                ]
            )
            + " \\\\"
        )
    step2c_table = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lrrrccc}",
            "\\toprule",
            "Model & Sharpe(1) & Sharpe(2a) & Sharpe(2b) & TotalRet(1) & TotalRet(2a) & TotalRet(2b)\\\\",
            "\\midrule",
            *step2c_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    (images_dir / "step2c.tex").write_text(step2c_intro + step2c_table, encoding="utf-8")

    step2d_intro = "\n".join(
        [
            "Step 2a (500/500) uses longer test windows and fewer model re-fits than Step 2b (500/100), which uses shorter test windows and more frequent re-training. "
            "Under leaky preprocessing, repeated short-horizon evaluation can amplify backtest over-optimism (L\\'opez de Prado; Bailey et al.).",
            "",
        ]
    )

    step2d_pred_rows = []
    for model in ["mlp", "lstm", "cnn_gaf"]:
        d_acc = fmt_key(step2b, model, "accuracy") - fmt_key(step2a, model, "accuracy")
        d_auc = fmt_key(step2b, model, "roc_auc") - fmt_key(step2a, model, "roc_auc")
        step2d_pred_rows.append(
            " & ".join(
                [
                    _latex_escape(model),
                    f"{fmt_key(step2a, model, 'accuracy'):.3f}",
                    f"{fmt_key(step2b, model, 'accuracy'):.3f}",
                    f"{d_acc:+.3f}",
                    f"{fmt_key(step2a, model, 'roc_auc'):.3f}",
                    f"{fmt_key(step2b, model, 'roc_auc'):.3f}",
                    f"{d_auc:+.3f}",
                ]
            )
            + " \\\\"
        )
    step2d_pred_table = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lrrrccc}",
            "\\toprule",
            "Model & Acc(2a) & Acc(2b) & $\\Delta$ & AUC(2a) & AUC(2b) & $\\Delta$\\\\",
            "\\midrule",
            *step2d_pred_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )

    step2d_tr_rows = []
    for model in ["mlp", "lstm", "cnn_gaf"]:
        d_sh = fmt_key(step2b, model, "sharpe") - fmt_key(step2a, model, "sharpe")
        d_tr = fmt_key(step2b, model, "total_return") - fmt_key(step2a, model, "total_return")
        step2d_tr_rows.append(
            " & ".join(
                [
                    _latex_escape(model),
                    f"{fmt_key(step2a, model, 'sharpe'):.3f}",
                    f"{fmt_key(step2b, model, 'sharpe'):.3f}",
                    f"{d_sh:+.3f}",
                    f"{fmt_key(step2a, model, 'total_return'):.3f}",
                    f"{fmt_key(step2b, model, 'total_return'):.3f}",
                    f"{d_tr:+.3f}",
                ]
            )
            + " \\\\"
        )
    step2d_tr_table = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lrrrccc}",
            "\\toprule",
            "Model & Sharpe(2a) & Sharpe(2b) & $\\Delta$ & TotalRet(2a) & TotalRet(2b) & $\\Delta$\\\\",
            "\\midrule",
            *step2d_tr_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    step2d_takeaway = "\n".join(
        [
            "If Step 2 performance is materially higher than Step 3 after leakage reduction, that pattern is consistent with backtest overfitting driven by leakage.",
            "",
        ]
    )
    (images_dir / "step2d.tex").write_text(step2d_intro + step2d_pred_table + step2d_tr_table + step2d_takeaway, encoding="utf-8")

    step3a_lines = [
        "We reduce leakage by enforcing time-series-safe preprocessing and split hygiene:",
        "\\begin{itemize}",
        "\\item Fit feature scalers inside each fold using the training window only, then transform training and test separately.",
        "\\item Purge: drop the last \\(h\\) observations of each training window (where \\(h\\) is the label horizon) so labels cannot overlap the test period.",
        f"\\item Embargo for lookback: skip the first {cfg.lookback} observations of each test window so test samples cannot reuse lookback information from the training boundary.",
        "\\item For GAF images, normalize returns using training-only min/max per fold (no global min/max).",
        "\\end{itemize}",
        "These practices follow time-series cross-validation guidance for financial ML (purging/embargo and leakage control) (L\\'opez de Prado).",
        "",
    ]
    (images_dir / "step3a.tex").write_text("\n".join(step3a_lines), encoding="utf-8")

    step3d_intro = "\n".join(
        [
            "To assess whether Step 2 benefited from leakage-driven overfitting, we compare Step 2 (leaky) to Step 3 (leakage reduced) under matched walk-forward settings.",
            "",
        ]
    )
    step3d_rows = []
    for model in ["mlp", "lstm", "cnn_gaf"]:
        d_500_500 = fmt_key(step3b, model, "sharpe") - fmt_key(step2a, model, "sharpe")
        d_500_100 = fmt_key(step3c, model, "sharpe") - fmt_key(step2b, model, "sharpe")
        step3d_rows.append(
            " & ".join(
                [
                    _latex_escape(model),
                    f"{fmt_key(step2a, model, 'sharpe'):.3f}",
                    f"{fmt_key(step3b, model, 'sharpe'):.3f}",
                    f"{d_500_500:+.3f}",
                    f"{fmt_key(step2b, model, 'sharpe'):.3f}",
                    f"{fmt_key(step3c, model, 'sharpe'):.3f}",
                    f"{d_500_100:+.3f}",
                ]
            )
            + " \\\\"
        )
    step3d_table_sharpe = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lrrrccc}",
            "\\toprule",
            "Model & Sharpe(2a) & Sharpe(3b) & $\\Delta$ & Sharpe(2b) & Sharpe(3c) & $\\Delta$\\\\",
            "\\midrule",
            *step3d_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    step3d_rows_tr = []
    for model in ["mlp", "lstm", "cnn_gaf"]:
        dtr_500_500 = fmt_key(step3b, model, "total_return") - fmt_key(step2a, model, "total_return")
        dtr_500_100 = fmt_key(step3c, model, "total_return") - fmt_key(step2b, model, "total_return")
        step3d_rows_tr.append(
            " & ".join(
                [
                    _latex_escape(model),
                    f"{fmt_key(step2a, model, 'total_return'):.3f}",
                    f"{fmt_key(step3b, model, 'total_return'):.3f}",
                    f"{dtr_500_500:+.3f}",
                    f"{fmt_key(step2b, model, 'total_return'):.3f}",
                    f"{fmt_key(step3c, model, 'total_return'):.3f}",
                    f"{dtr_500_100:+.3f}",
                ]
            )
            + " \\\\"
        )
    step3d_table_tr = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lrrrccc}",
            "\\toprule",
            "Model & TotalRet(2a) & TotalRet(3b) & $\\Delta$ & TotalRet(2b) & TotalRet(3c) & $\\Delta$\\\\",
            "\\midrule",
            *step3d_rows_tr,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    step3d_takeaway = "\n".join(
        [
            "If Sharpe and total return systematically drop when moving from Step 2 to Step 3 (especially under matched windows), this is consistent with leakage-driven over-optimism in Step 2 (L\\'opez de Prado; Bailey et al.).",
            "",
        ]
    )
    (images_dir / "step3d.tex").write_text(step3d_intro + step3d_table_sharpe + step3d_table_tr + step3d_takeaway, encoding="utf-8")

    step3_only = results[results["step"].isin(["3b", "3c"])].copy()
    if not step3_only.empty:
        best = step3_only.sort_values("sharpe", ascending=False).iloc[0]
        best_step = str(best["step"])
        best_model = str(best["model"])
        best_sharpe = float(best["sharpe"])
        best_tr = float(best["total_return"])
        best_dd = float(best["max_drawdown"])
    else:
        best_step = ""
        best_model = ""
        best_sharpe = float("nan")
        best_tr = float("nan")
        best_dd = float("nan")

    conclusion_lines = [
        "\\begin{itemize}",
        "\\item Walk-forward evaluation is materially harder than a single split because it exposes the model to regime changes and repeated re-training, which typically reduces stability.",
        "\\item Comparing Step 2 (leaky scaling) to Step 3 (fold-specific scaling + purging + embargo) shows how leakage can inflate backtest performance; the Step 3 results are the preferred estimate of deployable performance.",
        f"\\item Under leakage-reduced testing, the strongest configuration in this run is \\textbf{{{_latex_escape(best_model)}}} in \\textbf{{Step {best_step}}} with Sharpe={best_sharpe:.3f}, TotalRet={best_tr:.3f}, MaxDD={best_dd:.3f}.",
        "\\item Practical deployment should add position sizing, transaction-cost stress tests, and a strict research protocol (no tuning on test folds) to reduce the probability of backtest overfitting (L\\'opez de Prado; Bailey et al.).",
        "\\item In broader asset-pricing contexts, neural networks and other ML methods can add economic value, but only under careful out-of-sample evaluation (Gu, Kelly, and Xiu).",
        "\\item If performance is only marginal after leakage reduction, prioritize robustness (simpler models, fewer degrees of freedom, and longer test windows) over headline returns.",
        "\\end{itemize}",
        "",
    ]
    (images_dir / "conclusion.tex").write_text("\n".join(conclusion_lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    base_dir = Path(__file__).resolve().parents[1]
    cfg = ProjectConfig()
    results = run_all(cfg, base_dir=base_dir)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print(results)


if __name__ == "__main__":
    main()
