# 重新导入依赖，因为执行状态被重置
import re
import matplotlib.pyplot as plt

# 重新定义提取 time cost 的函数
def extract_time_cost(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    match = re.search(r'time cost:\s*([\d.]+)\s*s', text)
    return float(match.group(1)) if match else None

time_costs = {
    "Numpy (CPU)": extract_time_cost("./result_np.md"),
    "PyTorch": extract_time_cost("./result_base.md"),
    "Custom Cuda Operator": extract_time_cost("./result_c_v1.md"),
    "Custom Cuda Operators with Multiple Optimizations": extract_time_cost("./result_c_v2.md"),
}


# time_costs = {
#     "Based on Numpy (CPU)": extract_time_cost("./result_np.md"),
#     "Based on PyTorch (using GPU)": extract_time_cost("./result_base.md"),
#     "Based on PyTorch with optimized CUDA (GPU + CUDA)": extract_time_cost("./result_c_v1.md"),
#     "Based on PyTorch with optimized CUDA and optimized softmax (GPU + CUDA)": extract_time_cost("./result_c_v2.md"),
# }

plt.figure(figsize=(16, 8))
bars = plt.bar(time_costs.keys(), time_costs.values())
plt.xlabel("Strategy")
plt.ylabel("Total Training Time (s)")
plt.yscale("log")
plt.title("Total Time Cost (Log Scale)")
plt.grid(axis="y", which="both", linestyle='--', linewidth=0.5)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}s",
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
