# mruby-gpu-runtime

> **The Vulkan Compute runtime shared by mruby's GPU gems.**

🚧 **Being extracted from [mruby-gpu-narray](https://github.com/yujiteshima/mruby-gpu-narray)** as part of the Ruby Association Grant 2026 project "Expanding GPU computing libraries for mruby" ([selection result](https://www.ruby.or.jp/en/news/20261102)). The API is not stable yet.

## Status

The Vulkan core of mruby-gpu-narray now lives here: device and queue setup, buffers and their memory, the descriptor layouts, pipeline creation, and the deferred batch that records dispatches and submits them once. So do the `GPU` module functions about the device and the batch: `GPU.info`, `GPU.device_name`, `GPU.sync`, `GPU.pending` and `GPU.sync_mode`.

- The C API is `include/gpu_runtime.h`: opaque handles for the context, buffers and pipelines, and functions that start with `gpu_rt_`. It is at API version 0.1 and may still change; see [Writing a gem on the C API](#writing-a-gem-on-the-c-api).
- A pipeline's layout is described by value, and today it can be one, two or three storage buffers plus up to 16 bytes of push constants.
- Each buffer has a `VkDeviceMemory` of its own; there is no memory pool yet.
- The device can be chosen, and `GPU.info[:features]` says what it can do; see [Devices and features](#devices-and-features).

## What it will provide

- Device selection and feature reporting (`GPU.info`): done, see [Devices and features](#devices-and-features)
- Buffers that carry a dtype and a shape, and a batch that records dispatches and submits them once
- Pipelines created from SPIR-V, with the shaders embedded in the binary
- A C API with a single header for gem authors (`include/gpu_runtime.h`)
- A profiler: `GPU.profile { ... }` with per-dispatch GPU time and Chrome trace output
- A GitHub Actions template that runs the tests on Mesa lavapipe, so no GPU is needed

[mruby-gpu-narray](https://github.com/yujiteshima/mruby-gpu-narray) and [mruby-gpu-kernel](https://github.com/yujiteshima/mruby-gpu-kernel) will be rebuilt on top of it. Like them, it links only against the Vulkan loader.

Progress is tracked in the [GitHub Project](https://github.com/users/yujiteshima/projects/3).

## Devices and features

A machine can have several Vulkan devices. A Raspberry Pi 5 has two, the V3D GPU and llvmpipe, which runs on the CPU.

```ruby
GPU.devices
# => [{index: 0, name: "V3D 7.1.10.2", type: :integrated},
#     {index: 1, name: "llvmpipe (LLVM 20.1.2, 128 bits)", type: :cpu}]

GPU.device = 1            # by index
GPU.device = "llvmpipe"   # or by name
GPU.device = nil          # the default again
```

The device starts at the first GPU operation, and it is chosen then, by the first of these that says something:

1. `GPU.device = index_or_name`, called before the first GPU operation.
2. The environment variable `MRUBY_GPU_DEVICE`, an index or a name. A script that makes no choice then runs unchanged on another device: `MRUBY_GPU_DEVICE=llvmpipe mruby script.rb`.
3. The first device that is not a CPU, or device 0 when every device is one.

- An index is the position in `GPU.devices`, which is the order Vulkan lists the devices in. That order can differ between machines and drivers; a name does not depend on it.
- A name matches the device whose name is equal to it, ignoring case. If there is none, it matches a device whose name contains it, so `"V3D"` finds `"V3D 7.1.10.2"` whatever the driver version. When several devices match, the one with the lowest index is used.
- `type` is `:integrated`, `:discrete`, `:virtual`, `:cpu` (a software rasterizer such as llvmpipe or lavapipe) or `:other`.
- `GPU.devices` and `GPU.device=` do not start the device.
- When no device matches, `GPU.device=` raises `ArgumentError`, and a `MRUBY_GPU_DEVICE` that matches nothing raises `RuntimeError` when the device starts. Both messages list the devices.
- Once the device has started it cannot change, because the buffers and pipelines of every GPU gem live on it. `GPU.device=` then accepts only the device in use, and raises `RuntimeError` for another one.

`GPU.info` describes the device in use:

```ruby
GPU.info
# => {device: "V3D 7.1.10.2", api_version: "1.3.305", backend: "Vulkan", max_workgroups: 65535,
#     device_index: 0, device_type: :integrated,
#     features: {storage_8bit: true, timestamp_query: true, subgroups: true, memory_budget: false}}
```

The values above are an example. `GPU.info[:features]` always has these four keys, and each value is `true` or `false`. The keys are stable: code may rely on them.

| Key | `true` when | C API |
|---|---|---|
| `:storage_8bit` | The device has `VK_KHR_8bit_storage` with `storageBuffer8BitAccess`: a shader may declare 8-bit integers in a storage buffer. | `GPU_RT_FEATURE_STORAGE_8BIT` |
| `:timestamp_query` | The compute queue writes timestamps (`timestampValidBits` is not 0, and `timestampPeriod` is positive). | `GPU_RT_FEATURE_TIMESTAMP_QUERY` |
| `:subgroups` | Compute shaders may use subgroup operations, at least the basic ones (`VkPhysicalDeviceSubgroupProperties`). | `GPU_RT_FEATURE_SUBGROUPS` |
| `:memory_budget` | The device has `VK_EXT_memory_budget`. | `GPU_RT_FEATURE_MEMORY_BUDGET` |

`true` means that the device has the feature and that the runtime has turned it on, so it can be used. The runtime turns on 8-bit storage and the memory-budget extension whenever the device has them; the other two need nothing turned on. A gem asks with `gpu_rt_context_has_feature`, which gives the same answers.

## Depending on it

mruby-gpu-narray builds on this gem, and mruby-gpu-kernel and the coming image and camera gems will too. Until this gem is on mgem-list, such a gem names the GitHub repository and the branch in its `mrbgem.rake`:

```ruby
spec.add_dependency 'mruby-gpu-runtime', github: 'yujiteshima/mruby-gpu-runtime', branch: 'main'
```

Keep `branch: 'main'`. Without it, mruby clones `master`, which this repository does not have, and the build stops.

mruby reads this line only when the build configuration does not list mruby-gpu-runtime itself. A build that lists it by path never fetches it, so a wrong line goes unnoticed there. Check the line once with a build configuration that leaves mruby-gpu-runtime out, as mruby-gpu-narray's CI does.

## Writing a gem on the C API

A GPU gem includes one header, `<gpu_runtime.h>`, and never calls Vulkan itself. The header starts with a complete example and the conventions; this is the outline.

| | |
|---|---|
| Version | `GPU_RT_API_VERSION`, `gpu_rt_require(mrb, major, minor)` |
| Context (`gpu_rt_context`) | `gpu_rt_context_get`, `gpu_rt_context_ready`, `gpu_rt_context_info`, `gpu_rt_context_has_feature` |
| Buffers (`gpu_rt_buffer`) | `gpu_rt_buffer_create`, `gpu_rt_buffer_wrap`, `gpu_rt_buffer_release`, `gpu_rt_buffer_get`, `gpu_rt_buffer_check`, `gpu_rt_buffer_size`, `gpu_rt_buffer_map`, `gpu_rt_buffer_unmap` |
| Pipelines (`gpu_rt_pipeline`) | `gpu_rt_pipeline_create`, `gpu_rt_pipeline_release` |
| Dispatch | `gpu_rt_dispatch` |
| Sync | `gpu_rt_sync`, `gpu_rt_buffer_sync`, `gpu_rt_pending` |

- `gpu_rt_context_get(mrb)` starts Vulkan the first time it is called, so a program that never uses the GPU never pays for it. The device is chosen there, as [Devices and features](#devices-and-features) describes; a gem does not choose it.
- `gpu_rt_context_info` fills a `gpu_rt_device_info` with what `GPU.info` reports, including the device's index and its `gpu_rt_device_type`. `gpu_rt_context_has_feature` answers for one `gpu_rt_feature`.
- A buffer is created with a size in bytes and a usage, and wrapped as a Ruby object at once, so that the GC frees it. `gpu_rt_buffer_get` reads the buffer of a `GPU::NArray` that Ruby passes in.
- A pipeline is created from SPIR-V bytes and a layout described by value.
- `gpu_rt_dispatch` records a dispatch, with a label that names the operation, into the batch that all GPU gems share. Nothing runs until a result is needed: `gpu_rt_buffer_map` waits for the dispatches that touch that buffer, and `gpu_rt_sync` for all of them.
- Every function takes `mrb` first and raises a Ruby exception when it fails.
- `gpu_rt_require(mrb, 0, 1)` in the gem's init function stops the gem with a message that names both versions when the runtime provides another API version.

In `mrbgem.rake` the gem needs only the dependency. The include path and the Vulkan loader come with it:

```ruby
spec.add_dependency 'mruby-gpu-runtime', github: 'yujiteshima/mruby-gpu-runtime', branch: 'main'
```

[`test/capi`](test/capi) is a small gem written this way, with its shaders embedded as SPIR-V.

## Build and test

To build this gem on its own, add it to `build_config.rb`:

```ruby
MRuby::Build.new do |conf|
  conf.toolchain :gcc          # use :clang on macOS
  conf.gembox 'default'
  conf.gem '/path/to/mruby-gpu-runtime'
  conf.gem '/path/to/mruby-gpu-runtime/test/capi'   # test bindings for the C API
  # On macOS, point at Homebrew's Vulkan (skip on Linux/Pi, where it's standard):
  # conf.cc.include_paths     << '/opt/homebrew/include'
  # conf.linker.library_paths << '/opt/homebrew/lib'
end
```

```sh
cd /path/to/mruby && MRUBY_CONFIG=build_config.rb rake
./build/host/bin/mruby /path/to/mruby-gpu-runtime/test/runtime_test.rb   # ALL TESTS PASSED
```

`test/capi` gives the test script a way to call the C API. Without it, the script tests the `GPU` module only, and says that it skipped the rest. Its shaders are committed as SPIR-V (`test/capi/src/*_spv.h`); after editing one, run `make -C test/capi/shader`, which needs `glslangValidator`.

mruby-gpu-narray's tests run on top of this gem as well.

## License

MIT
