/* gpu_api.c -- the gpu_rt_ functions of include/gpu_runtime.h.
 *
 * A thin layer over the core in gpu_vulkan.c and gpu_buffer.c: it checks what
 * a gem passes in, turns the descriptions into what the core takes, and keeps
 * track of the pipelines so that their owners never see a Vulkan handle.
 */
#include "gpu_internal.h"
#include <mruby/class.h>
#include <mruby/data.h>
#include <stdio.h>
#include <string.h>

/* ---- API version ---- */

void gpu_rt_require_at(mrb_state *mrb, int major, int minor, const char *file) {
  int ok = major == GPU_RT_API_VERSION_MAJOR &&
           (major == 0 ? minor == GPU_RT_API_VERSION_MINOR
                       : minor <= GPU_RT_API_VERSION_MINOR);
  if (ok) return;

  int runtime_is_older = major > GPU_RT_API_VERSION_MAJOR ||
    (major == GPU_RT_API_VERSION_MAJOR && minor > GPU_RT_API_VERSION_MINOR);
  char msg[512];
  snprintf(msg, sizeof(msg),
    "%s needs the mruby-gpu-runtime C API %d.%d, but this mruby was built with %d.%d. %s",
    file ? file : "a gem", major, minor,
    GPU_RT_API_VERSION_MAJOR, GPU_RT_API_VERSION_MINOR,
    runtime_is_older ? "Update mruby-gpu-runtime."
                     : "Update the gem, or build with an older mruby-gpu-runtime.");
  /* mruby 3.4 closes the state when a gem's init raises and reports only
   * "Invalid mrb_state", so say it where it will be seen. */
  fprintf(stderr, "mruby-gpu-runtime: %s\n", msg);
  mrb_raise(mrb, E_RUNTIME_ERROR, msg);
}

/* ---- Context ---- */

static struct gpu_rt_context the_context = { &g_ctx };

gpu_rt_context *gpu_rt_context_get(mrb_state *mrb) {
  gpu_ensure_initialized(mrb);
  return &the_context;
}

mrb_bool gpu_rt_context_ready(mrb_state *mrb) {
  (void)mrb;
  return g_ctx.initialized ? TRUE : FALSE;
}

/* Every function that takes the handle goes through this: a NULL or a handle
 * kept from before the device was torn down is a bug in the caller, and
 * reporting it beats dereferencing a dead VkDevice. */
static void check_context(mrb_state *mrb, gpu_rt_context *ctx, const char *fn) {
  if (ctx != &the_context || !g_ctx.initialized) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "%s: not a live context; pass what gpu_rt_context_get returns", fn);
  }
}

void gpu_rt_context_info(mrb_state *mrb, gpu_rt_context *ctx, gpu_rt_device_info *out) {
  check_context(mrb, ctx, "gpu_rt_context_info");
  /* deviceName lives in a struct on the stack, and the caller may keep the
   * pointer, so the name is copied once. There is one device per process. */
  static char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(g_ctx.physical_device, &props);
  memcpy(name, props.deviceName, sizeof(name));
  name[sizeof(name) - 1] = '\0';

  out->name = name;
  out->backend = "Vulkan";
  out->api_major = VK_VERSION_MAJOR(props.apiVersion);
  out->api_minor = VK_VERSION_MINOR(props.apiVersion);
  out->api_patch = VK_VERSION_PATCH(props.apiVersion);
  out->max_workgroups = g_ctx.max_workgroups;
}

mrb_bool gpu_rt_context_has_feature(mrb_state *mrb, gpu_rt_context *ctx, gpu_rt_feature feature) {
  check_context(mrb, ctx, "gpu_rt_context_has_feature");
  /* Nothing is detected or enabled at device creation yet (runtime#6), and a
   * feature that is not enabled must not be used, so the answer is no. */
  (void)feature;
  return FALSE;
}

/* ---- Buffers ---- */

gpu_rt_buffer *gpu_rt_buffer_create(mrb_state *mrb, gpu_rt_context *ctx,
                                    const gpu_rt_buffer_desc *desc) {
  check_context(mrb, ctx, "gpu_rt_buffer_create");
  if (!desc) mrb_raise(mrb, E_ARGUMENT_ERROR, "gpu_rt_buffer_create: no description");
  if (desc->usage == 0 ||
      (desc->usage & ~(uint32_t)(GPU_RT_BUFFER_STORAGE | GPU_RT_BUFFER_UNIFORM))) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "gpu_rt_buffer_create: usage %d is not a combination of GPU_RT_BUFFER_ bits",
      (int)desc->usage);
  }
  return gpu_buffer_create_bytes(mrb, (VkDeviceSize)desc->size, desc->usage);
}

mrb_value gpu_rt_buffer_wrap(mrb_state *mrb, struct RClass *klass, gpu_rt_buffer *buf) {
  return wrap_buffer(mrb, klass, buf);
}

void gpu_rt_buffer_release(mrb_state *mrb, gpu_rt_buffer *buf) {
  destroy_buffer(mrb, buf);
}

gpu_rt_buffer *gpu_rt_buffer_check(mrb_state *mrb, mrb_value obj) {
  (void)mrb;
  if (mrb_type(obj) == MRB_TT_CDATA && DATA_TYPE(obj) == &gpu_buffer_type) {
    return (gpu_rt_buffer *)DATA_PTR(obj);
  }
  return NULL;
}

gpu_rt_buffer *gpu_rt_buffer_get(mrb_state *mrb, mrb_value obj) {
  gpu_rt_buffer *buf = gpu_rt_buffer_check(mrb, obj);
  if (!buf) {
    mrb_raisef(mrb, E_TYPE_ERROR, "expected a GPU buffer (GPU::NArray), got %s",
               mrb_obj_classname(mrb, obj));
  }
  return buf;
}

size_t gpu_rt_buffer_size(mrb_state *mrb, gpu_rt_buffer *buf) {
  (void)mrb;
  return (size_t)buf->size;
}

void *gpu_rt_buffer_map(mrb_state *mrb, gpu_rt_buffer *buf) {
  return map_buffer(mrb, buf);
}

void gpu_rt_buffer_unmap(mrb_state *mrb, gpu_rt_buffer *buf) {
  (void)mrb;
  if (g_ctx.initialized) unmap_buffer(buf);
}

void gpu_rt_context_limits(mrb_state *mrb, gpu_rt_context *ctx, gpu_rt_limits *out) {
  check_context(mrb, ctx, "gpu_rt_context_limits");
  out->max_storage_buffers = g_ctx.max_storage_buffers;
  out->max_uniform_buffers = g_ctx.max_uniform_buffers;
  out->max_push_constant_size = g_ctx.max_push_constant_size;
  out->max_uniform_buffer_size = g_ctx.max_uniform_buffer_size;
}

/* ---- Pipelines ---- */

static GpuPipeline *g_pipelines = NULL;   /* the cache: every pipeline the device holds */
static uint32_t g_pipelines_len = 0;
static uint32_t g_pipelines_unused = 0;   /* of those, with no handle left, waiting for the batch */

static void pipeline_unlink(GpuPipeline *e) {
  if (e->prev) e->prev->next = e->next; else g_pipelines = e->next;
  if (e->next) e->next->prev = e->prev;
  e->prev = e->next = NULL;
  g_pipelines_len--;
}

static void handle_free(mrb_state *mrb, gpu_rt_pipeline *p) {
  if (p->label) mrb_free(mrb, p->label);
  mrb_free(mrb, p);
}

/* Destroy a pipeline no handle and no recorded dispatch uses. */
static void pipeline_destroy(mrb_state *mrb, GpuPipeline *e) {
  vkDestroyPipeline(g_ctx.device, e->pipeline, NULL);
  pipeline_unlink(e);
  mrb_free(mrb, e->spirv);
  mrb_free(mrb, e);
}

/* Check a description against what the device allows in one layout. The
 * numbers in the messages are the device's, so they differ between devices. */
static void check_layout(mrb_state *mrb, const gpu_rt_layout *layout, const char *label) {
  if (!layout) mrb_raise(mrb, E_ARGUMENT_ERROR, "gpu_rt_pipeline_create: no layout");
  /* mrb_raisef's %i takes an mrb_int, which holds any uint32_t limit. */
  if (layout->storage_buffers < 1 || layout->storage_buffers > g_ctx.max_storage_buffers) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "gpu_rt_pipeline_create(%s): %i storage buffers, and a layout on this device takes 1 to %i",
      label, (mrb_int)layout->storage_buffers, (mrb_int)g_ctx.max_storage_buffers);
  }
  if (layout->uniform_buffers > g_ctx.max_uniform_buffers) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "gpu_rt_pipeline_create(%s): %i uniform buffers, and a layout on this device takes at most %i",
      label, (mrb_int)layout->uniform_buffers, (mrb_int)g_ctx.max_uniform_buffers);
  }
  if ((uint64_t)layout->storage_buffers + layout->uniform_buffers > g_ctx.max_bindings) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "gpu_rt_pipeline_create(%s): %i storage and %i uniform buffers, and a layout on this "
      "device binds at most %i buffers in all",
      label, (mrb_int)layout->storage_buffers, (mrb_int)layout->uniform_buffers,
      (mrb_int)g_ctx.max_bindings);
  }
  if (layout->push_constant_size > g_ctx.max_push_constant_size) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "gpu_rt_pipeline_create(%s): %i bytes of push constants, and the most on this device is %i",
      label, (mrb_int)layout->push_constant_size, (mrb_int)g_ctx.max_push_constant_size);
  }
  if (layout->push_constant_size % 4 != 0) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "gpu_rt_pipeline_create(%s): %i bytes of push constants, and the size must be a multiple of 4",
      label, (mrb_int)layout->push_constant_size);
  }
}

/* FNV-1a, 64 bits. Only a first filter: a hit is confirmed by comparing the
 * layout and every byte of the code. */
static uint64_t hash_bytes(uint64_t h, const void *data, size_t size) {
  const unsigned char *p = (const unsigned char *)data;
  for (size_t i = 0; i < size; i++) {
    h ^= p[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

/* Report a failure the way the caller asked for: fill *error_out and return
 * NULL, or raise. */
static gpu_rt_pipeline *pipeline_failed(mrb_state *mrb, const char *label,
                                        const char *operation, const char *reason,
                                        VkResult r, gpu_rt_error *error_out) {
  if (error_out) {
    error_out->operation = operation;
    error_out->reason = reason;
    error_out->vk_result = (int32_t)r;
    return NULL;
  }
  mrb_raisef(mrb, E_RUNTIME_ERROR, "gpu_rt_pipeline_create(%s): %s failed: %s (VkResult %d)",
             label, operation, reason, (int)r);
  return NULL;   /* not reached */
}

gpu_rt_pipeline *gpu_rt_pipeline_create(mrb_state *mrb, gpu_rt_context *ctx,
                                        const void *spirv, size_t spirv_size,
                                        const gpu_rt_layout *layout, const char *label,
                                        gpu_rt_error *error_out) {
  const char *shown = label ? label : "unnamed";
  check_context(mrb, ctx, "gpu_rt_pipeline_create");
  check_layout(mrb, layout, shown);
  if (error_out) memset(error_out, 0, sizeof(*error_out));

  /* Vulkan reads the code as 32-bit words. Handing it anything else is
   * undefined behaviour in the driver, so it is refused here. */
  if (!spirv || spirv_size == 0 || spirv_size % 4 != 0) {
    return pipeline_failed(mrb, shown, "the SPIR-V size check",
                           "the code is empty or not a whole number of 32-bit words",
                           VK_SUCCESS, error_out);
  }

  VkResult r;
  const char *step;
  const GpuLayout *l = gpu_layout_get(layout->storage_buffers, layout->uniform_buffers,
                                      layout->push_constant_size, &r, &step);
  if (!l) return pipeline_failed(mrb, shown, step, gpu_result_name(r), r, error_out);

  /* The cache: the same code with the same layout is the same pipeline. */
  const uint32_t key[3] = { layout->storage_buffers, layout->uniform_buffers,
                            layout->push_constant_size };
  uint64_t hash = hash_bytes(0xcbf29ce484222325ull, key, sizeof(key));
  hash = hash_bytes(hash, spirv, spirv_size);
  GpuPipeline *e;
  for (e = g_pipelines; e; e = e->next) {
    if (e->hash == hash && e->layout == l && e->spirv_size == spirv_size &&
        memcmp(e->spirv, spirv, spirv_size) == 0) {
      break;
    }
  }

  /* Only this allocation can raise, and nothing exists yet to unwind. The
   * ones below report instead (the label is only a name, so it may be lost). */
  gpu_rt_pipeline *p = mrb_malloc(mrb, sizeof(*p));
  memset(p, 0, sizeof(*p));
  if (label) {
    size_t len = strlen(label) + 1;
    p->label = mrb_malloc_simple(mrb, len);
    if (p->label) memcpy(p->label, label, len);
  }

  if (e) {
    /* One with no handle left was waiting for the batch to be destroyed.
     * It is in use again, so the sweep must leave it. */
    if (e->refs == 0) g_pipelines_unused--;
    e->refs++;
    p->shared = e;
    return p;
  }

  /* The caller's bytes may sit at any address; the words must be aligned.
   * The copy is kept, to compare later creations against. */
  e = mrb_malloc_simple(mrb, sizeof(*e));
  uint32_t *words = mrb_malloc_simple(mrb, spirv_size);
  if (!e || !words) {
    if (e) mrb_free(mrb, e);
    if (words) mrb_free(mrb, words);
    handle_free(mrb, p);
    return pipeline_failed(mrb, shown, "copying the SPIR-V", "out of host memory",
                           VK_ERROR_OUT_OF_HOST_MEMORY, error_out);
  }
  memcpy(words, spirv, spirv_size);
  memset(e, 0, sizeof(*e));
  e->layout = l;
  e->spirv = words;
  e->spirv_size = spirv_size;
  e->hash = hash;
  e->refs = 1;

  r = gpu_create_pipeline(words, spirv_size, l, &e->pipeline);
  if (r != VK_SUCCESS) {
    mrb_free(mrb, words);
    mrb_free(mrb, e);
    handle_free(mrb, p);
    return pipeline_failed(mrb, shown, "creating the compute pipeline",
                           gpu_result_name(r), r, error_out);
  }

  e->next = g_pipelines;
  if (g_pipelines) g_pipelines->prev = e;
  g_pipelines = e;
  g_pipelines_len++;
  p->shared = e;
  return p;
}

void gpu_rt_pipeline_release(mrb_state *mrb, gpu_rt_pipeline *p) {
  if (!p) return;
  GpuPipeline *e = p->shared;
  handle_free(mrb, p);
  if (--e->refs > 0) return;             /* other handles share it */
  if (e->pipeline == VK_NULL_HANDLE) {   /* the device went first (gpu_pipelines_shutdown) */
    mrb_free(mrb, e);
    return;
  }
  if (g_ctx.batch_recording && e->epoch == g_ctx.epoch) {
    /* The command buffer being recorded binds this pipeline. Destroying it
     * now would make that command buffer invalid; gpu_flush sweeps it once
     * the batch has run. Until then it stays in the cache, and a creation
     * that asks for it again takes it back. */
    g_pipelines_unused++;
    return;
  }
  pipeline_destroy(mrb, e);
}

uint32_t gpu_rt_pipeline_count(mrb_state *mrb, gpu_rt_context *ctx) {
  check_context(mrb, ctx, "gpu_rt_pipeline_count");
  return g_pipelines_len - g_pipelines_unused;
}

void gpu_pipelines_sweep(mrb_state *mrb) {
  if (g_pipelines_unused == 0) return;
  for (GpuPipeline *e = g_pipelines, *next; e; e = next) {
    next = e->next;
    if (e->refs == 0) pipeline_destroy(mrb, e);
  }
  g_pipelines_unused = 0;
}

void gpu_pipelines_shutdown(mrb_state *mrb) {
  for (GpuPipeline *e = g_pipelines, *next; e; e = next) {
    next = e->next;
    vkDestroyPipeline(g_ctx.device, e->pipeline, NULL);
    e->pipeline = VK_NULL_HANDLE;
    e->layout = NULL;
    e->prev = e->next = NULL;
    mrb_free(mrb, e->spirv);
    e->spirv = NULL;
    /* One with no handle left has no owner; the others are freed by their
     * last handle's release, which now only frees the structs. */
    if (e->refs == 0) mrb_free(mrb, e);
  }
  g_pipelines = NULL;
  g_pipelines_len = 0;
  g_pipelines_unused = 0;
}

/* ---- Dispatch ---- */

/* Check the buffers given for one kind of binding against the layout. `first`
 * is the binding number of buffers[0]. */
static void check_buffers(mrb_state *mrb, const char *label, const char *kind,
                          gpu_rt_buffer **buffers, uint32_t count, uint32_t expected,
                          uint32_t first, uint32_t usage, const char *usage_name) {
  if (count != expected || (expected > 0 && !buffers)) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "gpu_rt_dispatch(%s): %i %s given, and the pipeline's layout binds %i",
      label, (mrb_int)(buffers ? count : 0), kind, (mrb_int)expected);
  }
  for (uint32_t i = 0; i < count; i++) {
    if (!buffers[i]) {
      mrb_raisef(mrb, E_ARGUMENT_ERROR, "gpu_rt_dispatch(%s): binding %i has no buffer",
                 label, (mrb_int)(first + i));
    }
    if (!(buffers[i]->usage & usage)) {
      mrb_raisef(mrb, E_ARGUMENT_ERROR,
        "gpu_rt_dispatch(%s): the buffer at binding %i was not created with %s",
        label, (mrb_int)(first + i), usage_name);
    }
  }
}

void gpu_rt_dispatch(mrb_state *mrb, const gpu_rt_dispatch_desc *d) {
  if (!d || !d->pipeline) mrb_raise(mrb, E_ARGUMENT_ERROR, "gpu_rt_dispatch: no pipeline");
  gpu_rt_pipeline *p = d->pipeline;
  GpuPipeline *e = p->shared;
  const char *label = d->label ? d->label : (p->label ? p->label : "unnamed");

  if (e->pipeline == VK_NULL_HANDLE || !g_ctx.initialized) {
    mrb_raisef(mrb, E_RUNTIME_ERROR,
      "gpu_rt_dispatch(%s): the pipeline was released, or the device is gone", label);
  }
  const GpuLayout *l = e->layout;
  check_buffers(mrb, label, "buffers", d->buffers, d->buffer_count, l->storage_buffers,
                0, GPU_RT_BUFFER_STORAGE, "GPU_RT_BUFFER_STORAGE");
  check_buffers(mrb, label, "uniform buffers", d->uniform_buffers, d->uniform_buffer_count,
                l->uniform_buffers, l->storage_buffers,
                GPU_RT_BUFFER_UNIFORM, "GPU_RT_BUFFER_UNIFORM");
  for (uint32_t i = 0; i < d->uniform_buffer_count; i++) {
    if (d->uniform_buffers[i]->bytes > g_ctx.max_uniform_buffer_size) {
      mrb_raisef(mrb, E_ARGUMENT_ERROR,
        "gpu_rt_dispatch(%s): the uniform buffer at binding %i has %i bytes, and the most "
        "on this device is %i",
        label, (mrb_int)(l->storage_buffers + i), (mrb_int)d->uniform_buffers[i]->bytes,
        (mrb_int)g_ctx.max_uniform_buffer_size);
    }
  }
  if (d->push_constant_size > l->push_constant_size ||
      (d->push_constant_size > 0 && !d->push_constants)) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "gpu_rt_dispatch(%s): %d bytes of push constants, and the pipeline's layout has %d",
      label, (int)d->push_constant_size, (int)l->push_constant_size);
  }
  if (d->push_constant_size % 4 != 0) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "gpu_rt_dispatch(%s): %d bytes of push constants, and the size must be a multiple of 4",
      label, (int)d->push_constant_size);
  }

  dispatch_pipeline(mrb, e->pipeline, l, d->buffers, d->uniform_buffers,
                    d->push_constants, d->push_constant_size,
                    d->groups[0],
                    d->groups[1] ? d->groups[1] : 1,
                    d->groups[2] ? d->groups[2] : 1);
  /* Marked after the call: the core may flush inside it (an empty descriptor
   * pool before recording, :eager mode after), and what matters to a later
   * release is whether the batch being recorded now binds this pipeline. */
  e->epoch = g_ctx.batch_recording ? g_ctx.epoch : 0;
}

/* ---- Sync ---- */

void gpu_rt_sync(mrb_state *mrb, gpu_rt_context *ctx) {
  check_context(mrb, ctx, "gpu_rt_sync");
  gpu_flush(mrb);
}

void gpu_rt_buffer_sync(mrb_state *mrb, gpu_rt_buffer *buf) {
  if (g_ctx.initialized) gpu_sync_buffer(mrb, buf);
}

uint32_t gpu_rt_pending(mrb_state *mrb, gpu_rt_context *ctx) {
  check_context(mrb, ctx, "gpu_rt_pending");
  return g_ctx.batch_dispatches;
}
