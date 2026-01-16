"""
Timing Logger Utility for Performance Optimization
Provides detailed sub-operation timing for AI service endpoints
"""

import time
from typing import Optional, Callable, Any
from functools import wraps
import asyncio
from contextlib import contextmanager, asynccontextmanager


class TimingLogger:
    """Context manager and decorator for timing operations"""
    
    def __init__(self, operation_name: str, request_id: Optional[str] = None):
        self.operation_name = operation_name
        self.request_id = request_id or "---"
        self.start_time: Optional[float] = None
        self.sub_timings: dict = {}
        
    def _format_duration(self, duration_ms: float) -> str:
        """Format duration with emoji indicator"""
        if duration_ms > 10000:
            return f"🔴 {duration_ms:.2f}ms (VERY SLOW)"
        elif duration_ms > 5000:
            return f"🟠 {duration_ms:.2f}ms (SLOW)"
        elif duration_ms > 2000:
            return f"🟡 {duration_ms:.2f}ms (MEDIUM)"
        elif duration_ms > 500:
            return f"🟢 {duration_ms:.2f}ms (GOOD)"
        else:
            return f"⚡ {duration_ms:.2f}ms (FAST)"
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        print(f"   ⏱️  [{self.request_id}] Starting: {self.operation_name}")
        
    def stop(self) -> float:
        """Stop timing and return duration in ms"""
        if self.start_time is None:
            return 0
        duration_ms = (time.time() - self.start_time) * 1000
        print(f"   ✓  [{self.request_id}] {self.operation_name}: {self._format_duration(duration_ms)}")
        return duration_ms
    
    def log_sub_operation(self, name: str, duration_ms: float):
        """Log a sub-operation timing"""
        self.sub_timings[name] = duration_ms
        print(f"      └─ {name}: {self._format_duration(duration_ms)}")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


@contextmanager
def time_operation(operation_name: str, request_id: str = "---"):
    """Context manager for timing synchronous operations"""
    timer = TimingLogger(operation_name, request_id)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


@asynccontextmanager
async def async_time_operation(operation_name: str, request_id: str = "---"):
    """Context manager for timing async operations"""
    timer = TimingLogger(operation_name, request_id)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


def timed(operation_name: Optional[str] = None):
    """Decorator to time function execution"""
    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                start = time.time()
                print(f"   ⏱️  [---] Starting: {name}")
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.time() - start) * 1000
                    emoji = "🔴" if duration_ms > 10000 else "🟠" if duration_ms > 5000 else "🟡" if duration_ms > 2000 else "🟢" if duration_ms > 500 else "⚡"
                    print(f"   ✓  [---] {name}: {emoji} {duration_ms:.2f}ms")
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    print(f"   ❌ [---] {name} FAILED after {duration_ms:.2f}ms: {str(e)[:50]}")
                    raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                start = time.time()
                print(f"   ⏱️  [---] Starting: {name}")
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start) * 1000
                    emoji = "🔴" if duration_ms > 10000 else "🟠" if duration_ms > 5000 else "🟡" if duration_ms > 2000 else "🟢" if duration_ms > 500 else "⚡"
                    print(f"   ✓  [---] {name}: {emoji} {duration_ms:.2f}ms")
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    print(f"   ❌ [---] {name} FAILED after {duration_ms:.2f}ms: {str(e)[:50]}")
                    raise
            return sync_wrapper
    return decorator


def log_timing_summary(timings: dict, total_ms: float, operation: str):
    """Print a summary of all sub-operation timings"""
    print(f"\n📊 TIMING SUMMARY for {operation}:")
    print(f"   ┌{'─' * 50}┐")
    
    sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
    for name, duration in sorted_timings:
        percent = (duration / total_ms * 100) if total_ms > 0 else 0
        bar_length = int(percent / 5)  # Scale to max 20 chars
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"   │ {name[:25]:<25} │ {duration:>8.0f}ms │ {bar} {percent:>5.1f}% │")
    
    print(f"   ├{'─' * 50}┤")
    print(f"   │ {'TOTAL':<25} │ {total_ms:>8.0f}ms │ {'█' * 20} 100.0% │")
    print(f"   └{'─' * 50}┘\n")
