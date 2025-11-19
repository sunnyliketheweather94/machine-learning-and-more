import matplotlib.pyplot as plt
import numpy as np

from chapter3.stock_utils import Process1

if __name__ == "__main__":
    plt.subplots(figsize=(10, 6))

    for start in [5, 10, 15, 20, 25, 30]:
        process = Process1(level_param=20, alpha=0.1)

        traces = process.generate_traces(
            start_price=start,
            time_steps=100,
            num_traces=10,
        )

        traces = traces.T
        traces = np.mean(traces, axis=1)

        plt.plot(
            range(1, traces.shape[0] + 1),
            traces,
            label=f"Starting price = ${start}",
        )

    plt.axhline(20, linestyle="--", color="black", label="Mean Price")

    plt.title("Stock Price Dynamics in Process 1")
    plt.xlabel("Time $t$")
    plt.ylabel("Stock Price $X_t$")
    plt.legend()

    plt.savefig("process1.png")
    plt.close()
