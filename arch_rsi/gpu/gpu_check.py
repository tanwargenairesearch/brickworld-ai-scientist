import time, torch
torch.set_num_threads(1)
dev = torch.device("mps")
# 1. A hand-written Metal kernel, compiled at runtime
src = """
kernel void add_scaled(device float* out [[buffer(0)]], device const float* a [[buffer(1)]],
                       device const float* b [[buffer(2)]], constant float& s [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) { out[i] = a[i] + s * b[i]; }
"""
lib = torch.mps.compile_shader(src)
a = torch.randn(1_000_000, device=dev); b = torch.randn(1_000_000, device=dev)
out = torch.empty_like(a)
lib.add_scaled(out, a, b, 0.5)
torch.mps.synchronize()
print("metal kernel max error vs torch:", (out - (a + 0.5 * b)).abs().max().item())

# 2. Speed on one big matrix multiply (not representative of our small networks)
def bench(device, n=2048, reps=10):
    x = torch.randn(n, n, device=device); y = torch.randn(n, n, device=device)
    for _ in range(3): x @ y
    if device.type == "mps": torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(reps): x @ y
    if device.type == "mps": torch.mps.synchronize()
    dt = (time.perf_counter() - t) / reps
    return 2 * n**3 / dt / 1e9
print(f"2048x2048 matmul: GPU {bench(dev):.0f} GFLOP/s, CPU 1 thread {bench(torch.device('cpu')):.0f} GFLOP/s")

# 3. Small-op overhead: many tiny launches, like a small RL network
x = torch.randn(64, 128, device=dev); w = torch.randn(128, 128, device=dev)
for _ in range(50): torch.relu(x @ w)
torch.mps.synchronize(); t = time.perf_counter()
for _ in range(2000): torch.relu(x @ w)
torch.mps.synchronize(); gpu_us = (time.perf_counter() - t) / 2000 * 1e6
xc, wc = x.cpu(), w.cpu(); t = time.perf_counter()
for _ in range(2000): torch.relu(xc @ wc)
cpu_us = (time.perf_counter() - t) / 2000 * 1e6
print(f"tiny 64x128 layer: GPU {gpu_us:.0f} us/call, CPU 1 thread {cpu_us:.0f} us/call")
