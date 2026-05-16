import os
import sys 
sys.path.append(os.path.abspath(".."))
import time
import torch
import torch.profiler
from PIL import Image
import torchvision.transforms as T
from model.model import OrientationEstimator

dummy_paths = [
    "../example/panorama.jpg"
]

dummy_query = Image.open(
    "../example/query.jpg"
).convert("RGB")

WARMUP_ITERS = 100

tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,) * 3, (0.5,) * 3),
])

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_cuda_time(e):
    return getattr(e, "cuda_time_total", 0.0) or getattr(e, "self_cuda_time_total", 0.0)

model = OrientationEstimator(tf=tf, device=device).to(device)
model.eval()

dummy_query = tf(dummy_query).unsqueeze(0).to(device)

# warm up
for _ in range(WARMUP_ITERS):
    with torch.no_grad():
        _ = model(dummy_query, dummy_paths)

if device == "cuda":
    torch.cuda.synchronize()

# profiling
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    record_shapes=True,
    profile_memory=True
) as prof:
    with torch.no_grad():
        _ = model(dummy_query, dummy_paths)

events = prof.key_averages()

total_cuda_time = sum(get_cuda_time(e) for e in events) / 1e6

print("\n===== GPU PROFILING (seconds) =====")

gpu_rows = []
for e in events:
    cuda_time = get_cuda_time(e)

    if cuda_time > 0:
        t = cuda_time / 1e6
        p = (t / total_cuda_time) * 100 if total_cuda_time > 0 else 0
        gpu_rows.append((e.key, t, p))

gpu_rows.sort(key=lambda x: x[1], reverse=True)

for name, t, p in gpu_rows[:20]:
    print(f"{name:35s} | {t:.6f} s | {p:6.2f}%")

print(f"\nTOTAL GPU TIME: {total_cuda_time:.6f} s")

print("\n===== TOP CUDA MEMORY OPS =====")
print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage",
    row_limit=1000
))

if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    mem_before = torch.cuda.memory_allocated()

start = time.time()

with torch.no_grad():
    _ = model(dummy_query, dummy_paths)

if device == "cuda":
    torch.cuda.synchronize()

end = time.time()

print(f"\nEND-TO-END INFERENCE TIME: {end - start:.6f} s")

if device == "cuda":
    mem_after = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()
    mem_reserved_peak = torch.cuda.max_memory_reserved()

    print("\n===== GPU MEMORY =====")
    print(f"Memory before:        {mem_before / 1024**2:.2f} MB")
    print(f"Memory after:         {mem_after / 1024**2:.2f} MB")
    print(f"Peak allocated:       {mem_peak / 1024**2:.2f} MB")
    print(f"Peak reserved:        {mem_reserved_peak / 1024**2:.2f} MB")
