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

/* ---- Layouts ----
 *
 * What a pipeline binds, as the public gpu_rt_layout describes it: storage
 * buffers at bindings 0.., then uniform buffers, all in set 0, and one
 * push-constant range. The descriptor-set layout and the pipeline layout of
 * each distinct description are created the first time a pipeline asks for
 * it and kept until shutdown: there are few of them, and pipelines and
 * recorded dispatches point at them. */
typedef struct GpuLayout {
  uint32_t storage_buffers;
  uint32_t uniform_buffers;
  uint32_t push_constant_size;
  VkDescriptorSetLayout desc_layout;
  VkPipelineLayout pipe_layout;
  struct GpuLayout *next;
} GpuLayout;

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

/* How many descriptors of each type the pool holds follows from the layouts
 * created so far (see pool_want in gpu_vulkan.c): a full batch of the widest
 * layout while that stays under GPU_POOL_DESCRIPTORS, and for wider layouts
 * that many, or GPU_POOL_MIN_SETS sets if that is more. */
#define GPU_POOL_DESCRIPTORS 16384
#define GPU_POOL_MIN_SETS 8

/* Bytes of push constants one batch may carry. A batch that needs more is
 * flushed early, like one that needs more descriptor sets.
 *
 * This is a limit of Mesa V3DV (seen in 25.2.8, and unchanged on Mesa main in
 * November 2026), kept on every device so that a batch behaves the same
 * everywhere. When a shader reads its push constants with an index that is
 * not constant, V3DV copies every dispatch's block into one 4096-byte buffer
 * per command buffer (check_push_constants_ubo in v3dv_uniforms.c). Its check
 * for a full buffer is off by one block, and the buffer it then adds is
 * written at the old offset, so the block that crosses 4096 bytes and every
 * one after it lands outside the buffer: first a dispatch that silently does
 * nothing, then a crash.
 *
 * The runtime counts the way V3DV places the blocks. Each block has the
 * largest size pushed so far in the command buffer (batch_push_max), and
 * V3DV moves on by that size *before* it writes, so the last block ends at
 * the sum of all blocks, minus the first one, plus the current largest size.
 * dispatch_pipeline keeps that end within the budget.
 *
 * The 256 sets of a batch times the 16 bytes that were the most a layout
 * could have are exactly 4096, which is why this never showed before layouts
 * became general. */
#define GPU_BATCH_PUSH_BYTES 4096

/* ---- GPU Context (singleton) ---- */
typedef struct {
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family;
  VkCommandPool cmd_pool;
  VkDescriptorPool desc_pool;
  uint32_t max_workgroups;  /* maxComputeWorkGroupCount[0] */

  /* ---- Layouts and what the device allows in one (see gpu_layout_get) ---- */
  GpuLayout *layouts;                /* every layout created, until shutdown */
  uint32_t max_storage_buffers;      /* in one layout */
  uint32_t max_uniform_buffers;      /* in one layout */
  uint32_t max_bindings;             /* storage and uniform buffers together */
  uint32_t max_push_constant_size;   /* bytes */
  uint32_t max_uniform_buffer_size;  /* bytes one uniform binding may cover */

  /* ---- The descriptor pool's budget, per descriptor type ---- */
  uint32_t pool_storage, pool_uniform;    /* descriptors the pool was created with */
  uint32_t want_storage, want_uniform;    /* what the layouts so far ask for; the pool
                                           * is replaced before the next set if it holds less */
  uint32_t batch_storage, batch_uniform;  /* taken since the last pool reset */
  VkDescriptorBufferInfo *set_infos;      /* scratch for one set's update, as wide */
  VkWriteDescriptorSet   *set_writes;     /* as the widest layout */
  uint32_t set_cap;
  unsigned char *push_scratch;            /* as large as the largest push-constant block; */
  uint32_t push_scratch_cap;              /* see the second push in dispatch_pipeline */
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
  uint32_t batch_push_bytes;     /* push constants recorded since the last flush, each block */
  uint32_t batch_push_max;       /* counted at the largest size so far in the batch, */
  uint32_t batch_push_first;     /* and the first block's size; see GPU_BATCH_PUSH_BYTES */
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

/* Create the instance, device, queue, command pool and descriptor pool.
 * Raises on failure, and is attempted once per process. */
void gpu_init(mrb_state *mrb);

/* gpu_init unless it has already run. Every entry point that needs the device
 * calls this, so initialization is lazy. */
void gpu_ensure_initialized(mrb_state *mrb);

/* Tear everything down. Whatever is still recorded is dropped, not run.
 * Called from the gem's final, after the gems that depend on this one have
 * released their own Vulkan objects. */
void gpu_shutdown(mrb_state *mrb);

/* The layout for a description, created on first use. The caller has
 * checked the description against the device's limits. Returns NULL and sets
 * *result instead of raising, for the same reason as gpu_create_pipeline;
 * *step then names what failed. */
GpuLayout *gpu_layout_get(uint32_t storage_buffers, uint32_t uniform_buffers,
                          uint32_t push_constant_size,
                          VkResult *result, const char **step);

/* Create a compute pipeline from SPIR-V (`size` bytes of `code`) against a
 * layout. Returns the VkResult instead of raising, so a gem that builds its
 * pipelines up front can record which ones failed and report it when they
 * are used. On failure *out is VK_NULL_HANDLE. */
VkResult gpu_create_pipeline(const uint32_t *code, size_t size, const GpuLayout *layout,
                             VkPipeline *out);

/* Record one compute dispatch of `pipeline` (created against `layout`) over
 * the layout's storage buffers `bufs` and uniform buffers `uniforms`, each in
 * binding order. Nothing is submitted here unless GPU.sync_mode is :eager;
 * see gpu_flush. */
void dispatch_pipeline(mrb_state *mrb, VkPipeline pipeline, const GpuLayout *layout,
                       GpuBuffer **bufs, GpuBuffer **uniforms,
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
 * The cache. A GpuPipeline is one VkPipeline, shared by every
 * gpu_rt_pipeline_create call that gave the same SPIR-V and the same layout,
 * and counted by `refs`. It is found again by `hash` and then a full compare
 * of the kept copy of the code, and is destroyed when its last handle is
 * released -- after the batch has run, if a recorded dispatch binds it.
 *
 * The runtime owns the VkPipeline, which is why every live GpuPipeline is on
 * one list: shutdown destroys the Vulkan objects of pipelines nobody
 * released, and a release that comes later only frees the structs. */
typedef struct GpuPipeline {
  VkPipeline pipeline;       /* VK_NULL_HANDLE once the device is gone */
  const GpuLayout *layout;   /* NULL once the device is gone */
  uint32_t  *spirv;          /* the code this was created from, for the compare */
  size_t     spirv_size;     /* bytes */
  uint64_t   hash;           /* of the code and the layout */
  uint32_t   refs;           /* handles not yet released; 0 = waiting for the batch */
  uint32_t   epoch;          /* batch this pipeline was last dispatched in (0 = never) */
  struct GpuPipeline *prev, *next;
} GpuPipeline;

/* The public handle: one per gpu_rt_pipeline_create call, so each caller has
 * a label of its own and releases only its own reference. */
struct gpu_rt_pipeline {
  GpuPipeline *shared;
  char        *label;        /* mrb_malloc'ed copy, or NULL */
};

/* Destroy the pipelines that were released while the batch used them. Called
 * by gpu_flush once nothing is pending. */
void gpu_pipelines_sweep(mrb_state *mrb);

/* Destroy every pipeline's Vulkan object. Called by gpu_shutdown before the
 * device goes. */
void gpu_pipelines_shutdown(mrb_state *mrb);

#endif
