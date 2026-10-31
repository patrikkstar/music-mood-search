import csv
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path("experiments/results")
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

files = {
    "A: no backpressure": "a_unbounded.csv",
    "B: 1 worker": "b_1worker_fixed.csv",
    "C: 2 workers": "c_2workers_fixed.csv",
}

data = {}

for label, filename in files.items():
    with (RESULTS / filename).open(newline="") as f:
        data[label] = list(csv.DictReader(f))


def plot_metric(column, title, ylabel, filename, divisor=1):
    plt.figure(figsize=(10, 5))

    for label, rows in data.items():
        x = []
        y = []

        for row in rows:
            if row.get(column):
                x.append(float(row["elapsed_s"]))
                y.append(float(row[column]) / divisor)

        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel("Elapsed time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / filename, dpi=160)
    plt.close()


plot_metric(
    "ready",
    "RabbitMQ queue depth",
    "Ready messages",
    "queue_depth.png",
)

plot_metric(
    "unacked",
    "Unacknowledged messages",
    "Messages",
    "unacked.png",
)

plot_metric(
    "queue_memory_bytes",
    "Queue memory usage",
    "Memory (KiB)",
    "queue_memory.png",
    divisor=1024,
)

plot_metric(
    "broker_memory_mb",
    "RabbitMQ broker memory",
    "Memory (MiB)",
    "broker_memory.png",
)

labels = ["A: no BP", "B: 1 worker", "C: 2 workers"]
rates = [30.73, 1.52, 3.01]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, rates)

for bar, rate in zip(bars, rates, strict=False):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        rate + 0.3,
        f"{rate:.2f}",
        ha="center",
    )

plt.title("Average source publish rate")
plt.ylabel("Messages per second")
plt.ylim(0, 35)
plt.tight_layout()
plt.savefig(PLOTS / "source_rate.png", dpi=160)
plt.close()

print("Graphs saved to:", PLOTS)

for label, rows in data.items():
    values = [int(row["ready"]) for row in rows if row.get("ready")]
    print(f"{label}: max ready = {max(values)}")
