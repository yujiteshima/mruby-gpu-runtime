# mruby-gpu-runtime

> **The Vulkan Compute runtime shared by mruby's GPU gems.**

🚧 **Being extracted from [mruby-gpu-narray](https://github.com/yujiteshima/mruby-gpu-narray)** as part of the Ruby Association Grant 2026 project "Expanding GPU computing libraries for mruby" ([selection result](https://www.ruby.or.jp/en/news/20261102)). Nothing is usable here yet.

## What it will provide

- Device selection and feature reporting (`GPU.info`)
- Buffers that carry a dtype and a shape, and a batch that records dispatches and submits them once
- Pipelines created from SPIR-V, with the shaders embedded in the binary
- A C API with a single header for gem authors (`include/gpu_runtime.h`)
- A profiler: `GPU.profile { ... }` with per-dispatch GPU time and Chrome trace output
- A GitHub Actions template that runs the tests on Mesa lavapipe, so no GPU is needed

[mruby-gpu-narray](https://github.com/yujiteshima/mruby-gpu-narray) and [mruby-gpu-kernel](https://github.com/yujiteshima/mruby-gpu-kernel) will be rebuilt on top of it. Like them, it will link only against the Vulkan loader.

Progress is tracked in the [GitHub Project](https://github.com/users/yujiteshima/projects/3).

## License

MIT
