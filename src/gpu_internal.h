/* gpu_internal.h -- what the runtime's own sources share. Not installed, and
 * not for other gems: they include gpu_runtime.h.
 *
 * The context (g_ctx), the buffer struct and the core functions are still
 * declared in include/gpu_runtime_legacy.h, because the main branches of
 * mruby-gpu-narray and mruby-gpu-kernel compile against them. They move into
 * this file when that header is deleted.
 */
#ifndef GPU_INTERNAL_H
#define GPU_INTERNAL_H

#include "gpu_runtime.h"
#include "gpu_runtime_legacy.h"   /* also when a build defines GPU_RT_NO_LEGACY */

/* The context handle is the singleton. */
struct gpu_rt_context { GpuCtx *ctx; };

/* Create a buffer of `size` bytes (gpu_buffer.c). create_buffer, the legacy
 * entry point, is this with n float32 elements. */
GpuBuffer *gpu_buffer_create_bytes(mrb_state *mrb, VkDeviceSize size, uint32_t usage);

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
