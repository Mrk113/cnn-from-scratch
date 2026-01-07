import os
import hashlib

import matplotlib.pyplot as plt

def _append_hash(path: str, short_hash: str) -> str:

    root, ext = os.path.splitext(path)
    return f"{root}_{short_hash}{ext}"


def _new_short_hash() -> str:

    return hashlib.sha1(os.urandom(16)).hexdigest()[:10]


def log(data_history: dict, show: bool = True, save_path: str | None = None) -> None:

    epochs = range(1, len(data_history["train_loss"]) + 1)

    epoch_times = data_history.get("epoch_time")
    cumulative_minutes = []
    total_seconds = 0.0
    for t in epoch_times:
        total_seconds += float(t)
        cumulative_minutes.append(total_seconds / 60.0)

    if save_path is not None:
        suffixes = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
    
        root, ext = os.path.splitext(save_path)
        ext_lower = ext.lower()

        if ext_lower in suffixes:
            directory = os.path.dirname(save_path)
            stem = os.path.basename(root)
            loss_path = os.path.join(directory, f"{stem}_loss{ext}") if directory else f"{stem}_loss{ext}"
            acc_path = os.path.join(directory, f"{stem}_accuracy{ext}") if directory else f"{stem}_accuracy{ext}"
            time_path = (
                os.path.join(directory, f"{stem}_time{ext}")
                if directory
                else f"{stem}_time{ext}"
            )
        else:
            loss_path = f"{save_path}_loss.png"
            acc_path = f"{save_path}_accuracy.png"
            time_path = f"{save_path}_time.png"

        loss_dir = os.path.dirname(loss_path)
        acc_dir = os.path.dirname(acc_path)
        time_dir = os.path.dirname(time_path)
        if loss_dir:
            os.makedirs(loss_dir, exist_ok=True)
        if acc_dir:
            os.makedirs(acc_dir, exist_ok=True)
        if time_dir:
            os.makedirs(time_dir, exist_ok=True)

        run_hash = _new_short_hash()
        loss_path = _append_hash(loss_path, run_hash)
        acc_path = _append_hash(acc_path, run_hash)
        time_path = _append_hash(time_path, run_hash)

        fig_loss, ax_loss = plt.subplots(figsize=(6, 5))
        ax_loss.plot(epochs, data_history["train_loss"], label="Train Loss")
        ax_loss.plot(epochs, data_history["test_loss"], label="Test Loss")
        ax_loss.set_xlabel("Epochs")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss over Epochs")
        ax_loss.legend()
        fig_loss.tight_layout()
        fig_loss.savefig(loss_path)
        plt.close(fig_loss)

        fig_acc, ax_acc = plt.subplots(figsize=(6, 5))
        ax_acc.plot(epochs, data_history["test_acc"], label="Test Accuracy")
        ax_acc.set_xlabel("Epochs")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title("Accuracy over Epochs")
        ax_acc.legend()
        fig_acc.tight_layout()
        fig_acc.savefig(acc_path)
        plt.close(fig_acc)

        fig_time, ax_time = plt.subplots(figsize=(6, 5))
        ax_time.plot(epochs, cumulative_minutes, label="Cumulative Epoch Time")
        ax_time.set_xlabel("Epochs")
        ax_time.set_ylabel("Time (minutes)")
        ax_time.set_title("Cumulative Epoch Time (minutes)")
        ax_time.legend()
        fig_time.tight_layout()
        fig_time.savefig(time_path)
        plt.close(fig_time)

    if show:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        ax1.plot(epochs, data_history["train_loss"], label="Train Loss")
        ax1.plot(epochs, data_history["test_loss"], label="Test Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss over Epochs")
        ax1.legend()

        ax2.plot(epochs, data_history["test_acc"], label="Test Accuracy")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Accuracy over Epochs")
        ax2.legend()

        ax3.plot(epochs, cumulative_minutes, label="Cumulative Epoch Time")
        ax3.set_xlabel("Epochs")
        ax3.set_ylabel("Time (minutes)")
        ax3.set_title("Cumulative Epoch Time (minutes)")
        ax3.legend()

        fig.tight_layout()
        plt.show()
        plt.close(fig)