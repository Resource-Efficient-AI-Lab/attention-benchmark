import psutil

def monitor_resource_usage():
    memory_info = psutil.virtual_memory()
    used_memory_gb = memory_info.used / (1024 ** 3)

    cpu_percent = psutil.cpu_percent(interval=0)

    # Cumulative non-idle CPU time since boot
    cpu_times = psutil.cpu_times()
    cpu_time_used = sum(cpu_times) - cpu_times.idle

    return used_memory_gb, cpu_percent, cpu_time_used