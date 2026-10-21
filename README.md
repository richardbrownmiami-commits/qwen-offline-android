# Qwen Offline AI (Android 32-bit ARMv7)

A 100% offline, privacy-first Android LLM chat application optimized specifically for **ARMv7 32-bit devices (Android 11 / API 30)** such as Redmi 9 (Helio G35, 4 GB RAM, 2.8 GB free).

## Highlights

- **Pure Model Freedom**: No hidden system prompt or corporate morality wrapper. Generates direct answers as trained in the GGUF model.
- **Custom Prompt Injection**: Option to add your own system prompt/injection or leave it completely blank.
- **Pinned llama.cpp (`b11791`)**: Direct C++ JNI bridge with full support for Qwen 2.5 and 3.5 architectures, embedded GGUF chat templates, and ChatML fallback.
- **Optimized for 32-bit ARMv7**:
  - `LLAMA_LOAD_MODE_MMAP` (memory-mapped weights, avoids allocating 1.5 GB in heap)
  - Memory-aware context window (default 512 tokens, max 1024)
  - Chunked prompt evaluation (32 tokens) and NEON / VFPv4 compiler optimizations
  - Dynamic CPU thread management (capped to 4 big/LITTLE cores to prevent thermal throttling)
- **Built-in Vitals & Benchmark**: Real-time app RSS memory, free device RAM monitoring, and a 1-tap benchmark testing token generation speed.

## Recommended Models

1. **Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf** (~1.27 GB (reported size varies by source))
2. **Qwen2.5-1.5B-Instruct-uncensored-Q4_K_S.gguf** (~1.1 GB)
3. Any standard GGUF model in Q4_K_M or Q4_K_S format.

## Build and device testing

GitHub Actions builds an ARMv7 APK on pushes. Wait for the workflow to pass, then download its artifact from the **Actions** tab. A successful CI build verifies compilation and ABI packaging only; it cannot prove that a specific 1.3 GB GGUF will fit or generate quickly on your phone. Test with a small Qwen GGUF first, then the target model. The phone has a 32-bit process address space and memory pressure can still cause load failures, even with memory-mapped weights.

## How to Install & Run

1. Download the latest successful APK artifact from the **Actions** tab.
2. Install the APK on your Android 11 ARMv7 device.
3. Transfer your `.gguf` model file to your phone's storage (e.g. `Download` or `Models` folder).
4. Launch **Qwen Offline AI**, tap **📁 Pick .gguf**, and select your model file.
5. Tap **Send** to start chatting offline!
