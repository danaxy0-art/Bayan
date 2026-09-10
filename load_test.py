"""Lab 7 - Step 5: HTTP load test with 16 concurrent clients."""

import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


URL = "http://127.0.0.1:8000/v1/classify"
PAYLOAD = {"text": "الخدمة ممتازة ولكن التأخير طويل"}
N_CONCURRENT = 16
N_REQUESTS = 200
TIMEOUT_S = 10


def send_one_request():
    start = time.perf_counter()
    response = requests.post(URL, json=PAYLOAD, timeout=TIMEOUT_S)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if response.status_code != 200:
        raise RuntimeError(f"Request failed with status {response.status_code}: {response.text}")
    return elapsed_ms


def main():
    print(f"Load test: {N_REQUESTS} requests, {N_CONCURRENT} concurrent workers")
    print(f"Target: {URL}\n")

    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=5)
        print(f"Health check: {health.status_code} {health.json()}\n")
    except Exception as exc:
        print(f"ERROR: could not reach the API server. Is uvicorn running?\n{exc}")
        return

    latencies = []
    errors = 0
    start_wall = time.perf_counter()

    with ThreadPoolExecutor(max_workers=N_CONCURRENT) as executor:
        futures = [executor.submit(send_one_request) for _ in range(N_REQUESTS)]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception as exc:
                errors += 1
                print(f"Request error: {exc}")

    total_wall_s = time.perf_counter() - start_wall

    if not latencies:
        print("No successful requests recorded.")
        return

    latencies.sort()
    p50 = statistics.median(latencies)
    p99 = latencies[int(len(latencies) * 0.99)]
    mean = statistics.mean(latencies)
    throughput = len(latencies) / total_wall_s

    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS (16 concurrent clients)")
    print("=" * 60)
    print(f"Total requests:     {N_REQUESTS}")
    print(f"Successful:         {len(latencies)}")
    print(f"Errors:             {errors}")
    print(f"Total wall time:    {total_wall_s:.2f}s")
    print(f"Throughput:         {throughput:.2f} req/s")
    print(f"p50 latency:        {p50:.2f} ms")
    print(f"p99 latency:        {p99:.2f} ms")
    print(f"mean latency:       {mean:.2f} ms")

    target_met = p99 <= 40
    print(f"\nTarget: HTTP p99 <= 40ms -> {'PASS' if target_met else 'FAIL'}")

    with open("BENCHMARKS.md", "a", encoding="utf-8") as f:
        f.write("\n## Lab 7 - HTTP Load Test (16 concurrent clients)\n\n")
        f.write(f"- Total requests: {N_REQUESTS}\n")
        f.write(f"- Successful: {len(latencies)}, Errors: {errors}\n")
        f.write(f"- Throughput: {throughput:.2f} req/s\n")
        f.write(f"- p50: {p50:.2f} ms\n")
        f.write(f"- p99: {p99:.2f} ms\n")
        f.write(f"- Target (p99 <= 40ms): {'PASS' if target_met else 'FAIL'}\n")

    print("\nAppended results to BENCHMARKS.md")


if __name__ == "__main__":
    main()