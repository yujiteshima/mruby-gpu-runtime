# mruby-gpu-runtime

> **The Vulkan Compute runtime shared by mruby's GPU gems.**

🚧 **Being extracted from [mruby-gpu-narray](https://github.com/yujiteshima/mruby-gpu-narray)** as part of the Ruby Association Grant 2026 project "Expanding GPU computing libraries for mruby" ([selection result](https://www.ruby.or.jp/en/news/20261102)). The API is not stable yet.

## Status

The Vulkan core of mruby-gpu-narray now lives here: device and queue setup, buffers and their memory, the descriptor layouts, pipeline creation, and the deferred batch that records dispatches and submits them once. So do the `GPU` module functions about the device and the batch: `GPU.info`, `GPU.device_name`, `GPU.sync`, `GPU.pending` and `GPU.sync_mode`.

- The C header, `include/gpu_runtime.h`, still exposes the context and buffer structs as they were in narray. Opaque handles and `gpu_rt_` functions replace them before the first release, so do not build on it yet.
- Pipelines use three fixed layouts (one, two or three storage buffers plus 16 bytes of push constants).
- Each buffer has a `VkDeviceMemory` of its own; there is no memory pool yet.

## What it will provide

- Device selection and feature reporting (`GPU.info`)
- Buffers that carry a dtype and a shape, and a batch that records dispatches and submits them once
- Pipelines created from SPIR-V, with the shaders embedded in the binary
- A C API with a single header for gem authors (`include/gpu_runtime.h`)
- A profiler: `GPU.profile { ... }` with per-dispatch GPU time and Chrome trace output
- A GitHub Actions template that runs the tests on Mesa lavapipe, so no GPU is needed

[mruby-gpu-narray](https://github.com/yujiteshima/mruby-gpu-narray) and [mruby-gpu-kernel](https://github.com/yujiteshima/mruby-gpu-kernel) will be rebuilt on top of it. Like them, it links only against the Vulkan loader.

Progress is tracked in the [GitHub Project](https://github.com/users/yujiteshima/projects/3).

## Depending on it

mruby-gpu-narray builds on this gem, and mruby-gpu-kernel and the coming image and camera gems will too. Until this gem is on mgem-list, such a gem names the GitHub repository and the branch in its `mrbgem.rake`:

```ruby
spec.add_dependency 'mruby-gpu-runtime', github: 'yujiteshima/mruby-gpu-runtime', branch: 'main'
```

Keep `branch: 'main'`. Without it, mruby clones `master`, which this repository does not have, and the build stops.

mruby reads this line only when the build configuration does not list mruby-gpu-runtime itself. A build that lists it by path never fetches it, so a wrong line goes unnoticed there. Check the line once with a build configuration that leaves mruby-gpu-runtime out, as mruby-gpu-narray's CI does.

## Build and test

To build this gem on its own, add it to `build_config.rb`:

```ruby
MRuby::Build.new do |conf|
  conf.toolchain :gcc          # use :clang on macOS
  conf.gembox 'default'
  conf.gem '/path/to/mruby-gpu-runtime'
  # On macOS, point at Homebrew's Vulkan (skip on Linux/Pi, where it's standard):
  # conf.cc.include_paths     << '/opt/homebrew/include'
  # conf.linker.library_paths << '/opt/homebrew/lib'
end
```

```sh
cd /path/to/mruby && MRUBY_CONFIG=build_config.rb rake
./build/host/bin/mruby /path/to/mruby-gpu-runtime/test/runtime_test.rb   # ALL TESTS PASSED
```

Buffers and dispatches are tested through mruby-gpu-narray's tests, which run on top of this gem.

## License

MIT
