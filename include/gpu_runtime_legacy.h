/* gpu_runtime_legacy.h -- the declarations mruby-gpu-narray's gpu_internal.h
 * used to have, kept until the gems built on them have moved.
 *
 * DO NOT BUILD ON THIS. It exposes the context (g_ctx) and the buffer as plain
 * structs and pipelines as raw VkPipeline handles against three fixed layouts.
 * gpu_runtime.h includes it, unless GPU_RT_NO_LEGACY is defined, only so that
 * the main branches of mruby-gpu-narray and mruby-gpu-kernel keep building
 * while they move to the gpu_rt_ functions. Once both have moved, this file is
 * deleted and its contents become src/gpu_internal.h.
 */
#ifndef GPU_RUNTIME_LEGACY_H
#define GPU_RUNTIME_LEGACY_H

#include <mruby.h>
#include <mruby/data.h>
#include <vulkan/vulkan.h>
#include <stddef.h>
#include <stdint.h>

/* ---- Descriptor-set layouts ----
 *
 * Every pipeline binds one, two or three storage buffers, plus up to
 * GPU_PUSH_CONSTANT_SIZE bytes of push constants:
 *
 *   LAYOUT_3BUF takes (a, b, c): element-wise binary ops.
 *   LAYOUT_2BUF takes (a, b):    scalar ops, reductions, out-of-place passes.
 *   LAYOUT_1BUF takes (a):       in-place passes.
 */
typedef enum { LAYOUT_3BUF = 0, LAYOUT_2BUF, LAYOUT_1BUF, LAYOUT_COUNT } LayoutId;

/* Every push block so far is 8 bytes ({ uint n; float scalar; },
 * { uint n; uint h; }, ...); 16 leaves headroom for the next one. */
#define GPU_PUSH_CONSTANT_SIZE 16

/* ---- GPU Buffer ----
 *
 * The struct behind the public gpu_rt_buffer handle. `n` is what the legacy
 * callers read; the gpu_rt_ functions go by `size`. Both are kept in step, so
 * a buffer made through one API can be read through the other. */
typedef struct gpu_rt_buffer {
  VkBuffer buffer;
  VkDeviceMemory memory;
  uint32_t n;          /* float32 element count: size / 4 */
  VkDeviceSize bytes;  /* bytes allocated and bound: size, or 4 when size is 0 */
  uint32_t epoch;      /* batch this buffer was last bound in (0 = never); see dispatch_pipeline */
  VkDeviceSize size;   /* bytes the creator asked for */
  uint32_t usage;      /* GPU_RT_BUFFER_* bits */
} GpuBuffer;

/* The data type of every Ruby object that owns a GpuBuffer (GPU::NArray and
 * its subclasses). Its free function is the buffer's GC finalizer. */
extern const struct mrb_data_type gpu_buffer_type;

/* Descriptor sets the pool holds. A batch that needs more is flushed early,
 * which is just an earlier sync point, not an error. */
#define GPU_MAX_DESC_SETS 256

/* ---- GPU Context (singleton) ---- */
typedef struct {
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family;
  VkCommandPool cmd_pool;
  VkDescriptorSetLayout desc_layouts[LAYOUT_COUNT];
  VkPipelineLayout pipe_layouts[LAYOUT_COUNT];
  VkDescriptorPool desc_pool;
  uint32_t max_workgroups;  /* maxComputeWorkGroupCount[0] */
  int initialized;
  int init_failed;          /* a previous gpu_init raised; do not retry */

  /* ---- Deferred submission (see dispatch_pipeline / gpu_flush) ----
   *
   * Dispatches are recorded into one long-lived command buffer and submitted
   * together the first time the host needs a result. */
  VkCommandBuffer batch_cmd;     /* allocated at init, reset by every begin */
  VkFence         batch_fence;   /* reused across flushes */
  int      batch_recording;      /* batch_cmd is between vkBegin and vkEnd */
  uint32_t batch_dispatches;     /* recorded since the last flush (GPU.pending) */
  uint32_t batch_sets;           /* descriptor sets taken since the last pool reset */
  uint32_t epoch;                /* batch number; bumped by every flush, never 0 */
  int      eager;                /* GPU.sync_mode = :eager -> flush after every dispatch */
  GpuBuffer **graveyard;         /* buffers whose Ruby owner died while a batch still referenced them */
  size_t graveyard_len;
  size_t graveyard_cap;
} GpuCtx;

extern GpuCtx g_ctx;

/* ---- Error handling ----
 *
 * Every Vulkan call that returns a VkResult goes through VK_CHECK, which turns
 * a failure into an mruby exception. Unchecked, a failed call leaves an
 * unusable handle behind and the next call either segfaults the VM or -- worse
 * -- returns whatever happened to be in the buffer as if it were a result.
 *
 * gpu_check raises, so it does not return on failure: anything the caller
 * allocated must be released *before* the call. */
void gpu_check(mrb_state *mrb, VkResult r, const char *call);
const char *gpu_result_name(VkResult r);
#define VK_CHECK(mrb, expr) gpu_check((mrb), (expr), #expr)

/* ---- gpu_vulkan.c ---- */

/* Create the instance, device, queue, command pool, descriptor layouts and
 * pool. Raises on failure, and is attempted once per process. */
void gpu_init(mrb_state *mrb);

/* gpu_init unless it has already run. Every entry point that needs the device
 * calls this, so initialization is lazy. */
void gpu_ensure_initialized(mrb_state *mrb);

/* Tear everything down. Whatever is still recorded is dropped, not run.
 * Called from the gem's final, after the gems that depend on this one have
 * released their own Vulkan objects. */
void gpu_shutdown(mrb_state *mrb);

/* Create a compute pipeline from SPIR-V (`size` bytes of `code`) against one of
 * the fixed layouts. Returns the VkResult instead of raising, so a gem that
 * builds its pipelines up front can record which ones failed and report it
 * when they are used. On failure *out is VK_NULL_HANDLE. */
VkResult gpu_create_pipeline(const uint32_t *code, size_t size, LayoutId lid,
                             VkPipeline *out);

/* Record one compute dispatch of `pipeline` (created against
 * g_ctx.pipe_layouts[lid]) over `bufs`, in binding order. Nothing is submitted
 * here unless GPU.sync_mode is :eager; see gpu_flush. */
void dispatch_pipeline(mrb_state *mrb, VkPipeline pipeline, LayoutId lid,
                       GpuBuffer **bufs, int num_buffers,
                       const void *push_data, uint32_t push_size,
                       uint32_t group_x, uint32_t group_y, uint32_t group_z);

/* Submit everything recorded so far, wait for it, and reset for the next batch.
 * Also frees buffers parked in the graveyard. Safe to call with nothing
 * pending. */
void gpu_flush(mrb_state *mrb);

/* Flush only if the pending batch references `buf`. Every host-side read or
 * write of a buffer goes through this (map_buffer calls it). */
void gpu_sync_buffer(mrb_state *mrb, GpuBuffer *buf);

/* ---- gpu_buffer.c ---- */
GpuBuffer *create_buffer(mrb_state *mrb, uint32_t n);
void       destroy_buffer(mrb_state *mrb, GpuBuffer *buf);  /* GC finalizer + create_buffer unwind */
mrb_value  wrap_buffer(mrb_state *mrb, struct RClass *klass, GpuBuffer *buf);
float     *map_buffer(mrb_state *mrb, GpuBuffer *buf);
void       unmap_buffer(GpuBuffer *buf);

#endif
