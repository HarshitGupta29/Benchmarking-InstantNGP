import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def benchmark(funcs, inputs, labels, metrics):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize metric dictionaries
    metric_values = {}
    for func in funcs:
        metric_values[func.__class__.__name__] = {metric.__class__.__name__: [] for metric in metrics}

    # Run each function and compute metrics for each input
    for func in funcs:
        for i in range(len(inputs)):
            input_tensor = torch.Tensor(inputs[i]).to(device)
            label_tensor = torch.Tensor(labels[i]).to(device)

            start_time = time.time()
            output = func(input_tensor)
            elapsed_time = time.time() - start_time

            for metric in metrics:
                metric_values[func.__class__.__name__][metric.__class__.__name__].append(metric(output.detach(), label_tensor.detach()).item())

            print("%s - Input %d: %.4f seconds" % (func.__class__.__name__, i, elapsed_time))

    # Print metric values for each function and each input
    for func in funcs:
        for metric in metrics:
            print("%s - %s: %.4f" % (func.__class__.__name__, metric.__class__.__name__, sum(metric_values[func.__class__.__name__][metric.__class__.__name__]) / len(metric_values[func.__class__.__name__][metric.__class__.__name__])))

    return metric_values

def plot_benchmark(metric_values, metrics, n=10):
    # Create directory to save plots
    save_dir = time.strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Plot metric values for each metric
    for metric in metrics:
        metric_dir = os.path.join(save_dir, metric.__class__.__name__)
        if not os.path.exists(metric_dir):
            os.mkdir(metric_dir)

        fig, ax = plt.subplots(figsize=(8, 6))
        metric_name = metric.__class__.__name__

        # Get the length of the values from the first function
        index = np.arange(len(next(iter(metric_values[metric_name].values()))))

        for func_name, func_metric_values in metric_values[metric_name].items():
            ax.plot(index, func_metric_values, label=func_name)

        ax.set_xlabel('Input')
        ax.set_ylabel('Metric value')
        ax.set_title('Benchmark results')
        ax.set_xticks(index[::n])
        ax.set_xticklabels(index[::n])
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(metric_dir, f"{save_dir}_{metric_name}.png"))
        plt.close()
