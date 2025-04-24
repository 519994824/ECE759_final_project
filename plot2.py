import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def extract_time_cost(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    match = re.search(r'time cost:\s*([\d.]+)\s*s', text)
    return float(match.group(1)) if match else None

time_costs = {
    "Numpy (CPU)": extract_time_cost("./result_np.md"),
    "PyTorch (GPU)": extract_time_cost("./result_torch.md"),
    "CUDA Optimized\n(Multiple Techniques)": extract_time_cost("./result_c.md"),
}

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(time_costs.keys(), time_costs.values(), width=0.6)

ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.0f}s'))

ax.set_title("Comparison of Training Time Across Methods", fontsize=16, weight='bold')
ax.set_xlabel("Training Strategy", fontsize=12)
ax.set_ylabel("Total Training Time (log scale, seconds)", fontsize=12)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height * 1.05, f"{height:.2f}s",
            ha='center', va='bottom', fontsize=11)

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

ax.grid(True, axis='y', which='both', linestyle='--', linewidth=0.6, alpha=0.7)

plt.tight_layout()
plt.savefig("Performance_Comparison.png", dpi=300)
plt.show()