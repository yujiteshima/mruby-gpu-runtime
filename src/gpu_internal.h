/* gpu_internal.h -- what the runtime's own sources share: the context
 * (g_ctx), the structs behind the public handles, and the core functions.
 *
 * Not installed, and not for other gems. They include gpu_runtime.h, which
 * shows none of this: no Vulkan type, and every handle opaque.
 */
#ifndef GPU_INTERNAL_H
#define GPU_INTERNAL_H

#include "gpu_runtime.h"
#include <mruby/data.h>
#include <vulkan/vulkan.h>

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
 * { uint n; uint h; }, ...); 16 leaves headroom for the next one. The public
 * header promises the same number as GPU_RT_MAX_PUSH_CONSTANT_SIZE. */
#define GPU_PUSH_CONSTANT_SIZE GPU_RT_MAX_PUSH_CONSTANT_SIZE

/* ---- GPU Buffer ----
 *
 * The struct behind the public gpu_rt_buffer handle. */
typedef struct gpu_rt_buffer {
  VkBuffer buffer;
  VkDeviceMemory memory;
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

/* The exception class a VkResult stands for (the table in README.md,
 * "Errors"). GPU_RT_ERROR_GENERIC for a result the table does not list. */
gpu_rt_error_kind gpu_result_kind(VkResult r);

/* gpu_check for a call made on behalf of a named operation: raises the class
 * of `r` with "<operation>: <call> failed: <reason> (VkResult <n>)". Does
 * not return. `operation` may be NULL. */
void gpu_fail(mrb_state *mrb, VkResult r, const char *call, const char *operation);

/* Raise GPU::Error or one of its subclasses with a fixed message. */
#define GPU_RAISE(mrb, kind, msg) mrb_raise((mrb), gpu_rt_error_class((mrb), (kind)), (msg))

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
 * here unless GPU.sync_mode is :eager; see gpu_flush. `label` names the
 * operation in the message of a Vulkan failure, and may be NULL. */
void dispatch_pipeline(mrb_state *mrb, VkPipeline pipeline, LayoutId lid,
                       GpuBuffer **bufs, int num_buffers,
                       const void *push_data, uint32_t push_size,
                       uint32_t group_x, uint32_t group_y, uint32_t group_z,
                       const char *label);

/* Submit everything recorded so far, wait for it, and reset for the next batch.
 * Also frees buffers parked in the graveyard. Safe to call with nothing
 * pending. */
void gpu_flush(mrb_state *mrb);

/* Flush only if the pending batch references `buf`. Every host-side read or
 * write of a buffer goes through this (map_buffer calls it). */
void gpu_sync_buffer(mrb_state *mrb, GpuBuffer *buf);

/* ---- gpu_buffer.c ---- */

/* Create a buffer of `size` bytes. Raises on failure. */
GpuBuffer *gpu_buffer_create_bytes(mrb_state *mrb, VkDeviceSize size, uint32_t usage);
void       destroy_buffer(mrb_state *mrb, GpuBuffer *buf);  /* GC finalizer + gpu_rt_buffer_release */
mrb_value  wrap_buffer(mrb_state *mrb, struct RClass *klass, GpuBuffer *buf);
void      *map_buffer(mrb_state *mrb, GpuBuffer *buf);
void       unmap_buffer(GpuBuffer *buf);

/* The context handle is the singleton. */
struct gpu_rt_context { GpuCtx *ctx; };

/* ---- Pipelines made through gpu_rt_pipeline_create (gpu_api.c) ----
 *
 * The caller owns the struct and the runtime owns the VkPipeline, which is
 * why every live pipeline is on one list: shutdown destroys the Vulkan
 * objects of pipelines nobody released, and a release that comes later only
 * frees the struct. */
struct gpu_rt_pipeline {
  VkPipeline pipeline;       /* VK_NULL_HANDLE once the device is gone */
  LayoutId   lid;            /* the fixed layout that serves this description */
  uint32_t   storage_buffers;
  uint32_t   push_constant_size;
  uint32_t   epoch;          /* batch this pipeline was last dispatched in (0 = never) */
  int        released;       /* the owner gave it back while a batch still used it */
  char      *label;          /* mrb_malloc'ed copy, or NULL */
  struct gpu_rt_pipeline *prev, *next;
};

/* Destroy the pipelines that were released while the batch used them. Called
 * by gpu_flush once nothing is pending. */
void gpu_pipelines_sweep(mrb_state *mrb);

/* Destroy every pipeline's Vulkan object. Called by gpu_shutdown before the
 * device goes. */
void gpu_pipelines_shutdown(mrb_state *mrb);

#endif
