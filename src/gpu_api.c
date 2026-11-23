/* gpu_api.c -- the gpu_rt_ functions of include/gpu_runtime.h.
 *
 * A thin layer over the core in gpu_vulkan.c and gpu_buffer.c: it checks what
 * a gem passes in, turns the descriptions into what the core takes, and keeps
 * track of the pipelines so that their owners never see a Vulkan handle.
 */
#include "gpu_internal.h"
#include <mruby/class.h>
#include <mruby/data.h>
#include <mruby/string.h>
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
  out->index = g_ctx.device_index;
  switch (g_ctx.device_type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: out->type = GPU_RT_DEVICE_INTEGRATED; break;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   out->type = GPU_RT_DEVICE_DISCRETE;   break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    out->type = GPU_RT_DEVICE_VIRTUAL;    break;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:            out->type = GPU_RT_DEVICE_CPU;        break;
    default:                                     out->type = GPU_RT_DEVICE_OTHER;      break;
  }
}

mrb_bool gpu_rt_context_has_feature(mrb_state *mrb, gpu_rt_context *ctx, gpu_rt_feature feature) {
  check_context(mrb, ctx, "gpu_rt_context_has_feature");
  /* Filled in by gpu_init: detected, and turned on where that is needed. */
  if ((int)feature < 0 || feature >= GPU_RT_FEATURE_COUNT) return FALSE;
  return g_ctx.features[feature] ? TRUE : FALSE;
}

/* ---- Buffers ---- */

static const struct { const char *name; size_t size; } dtypes[GPU_RT_DTYPE_COUNT] = {
  [GPU_RT_DTYPE_FLOAT32] = { "float32", 4 },
  [GPU_RT_DTYPE_UINT8]   = { "uint8",   1 },
  [GPU_RT_DTYPE_UINT16]  = { "uint16",  2 },
  [GPU_RT_DTYPE_INT32]   = { "int32",   4 },
};

size_t gpu_rt_dtype_size(mrb_state *mrb, gpu_rt_dtype dtype) {
  (void)mrb;
  return (unsigned)dtype < GPU_RT_DTYPE_COUNT ? dtypes[dtype].size : 0;
}

const char *gpu_rt_dtype_name(mrb_state *mrb, gpu_rt_dtype dtype) {
  (void)mrb;
  return (unsigned)dtype < GPU_RT_DTYPE_COUNT ? dtypes[dtype].name : "unknown";
}

/* For messages: mrb_raisef has no format for a size_t, and an mrb_int may be
 * too narrow for one. The strings are mruby's and die with the exception. */
static const char *size_string(mrb_state *mrb, size_t n) {
  char s[24];
  snprintf(s, sizeof(s), "%llu", (unsigned long long)n);
  return mrb_str_to_cstr(mrb, mrb_str_new_cstr(mrb, s));
}

static const char *shape_string(mrb_state *mrb, uint32_t ndim, const uint32_t *shape) {
  char s[16 * GPU_RT_MAX_DIMS];
  int at = snprintf(s, sizeof(s), "[");
  for (uint32_t i = 0; i < ndim; i++) {
    at += snprintf(s + at, sizeof(s) - (size_t)at, i ? ", %u" : "%u", (unsigned)shape[i]);
  }
  snprintf(s + at, sizeof(s) - (size_t)at, "]");
  return mrb_str_to_cstr(mrb, mrb_str_new_cstr(mrb, s));
}

gpu_rt_buffer *gpu_rt_buffer_create(mrb_state *mrb, gpu_rt_context *ctx,
                                    const gpu_rt_buffer_desc *desc) {
  check_context(mrb, ctx, "gpu_rt_buffer_create");
  if (!desc) mrb_raise(mrb, E_ARGUMENT_ERROR, "gpu_rt_buffer_create: no description");
  if (desc->usage == 0 || (desc->usage & ~(uint32_t)GPU_RT_BUFFER_STORAGE)) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "gpu_rt_buffer_create: usage %d is not a combination of GPU_RT_BUFFER_ bits",
      (int)desc->usage);
  }
  size_t elem = gpu_rt_dtype_size(mrb, desc->dtype);
  if (elem == 0) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR, "gpu_rt_buffer_create: %d is not a dtype",
               (int)desc->dtype);
  }
  const char *dtype = gpu_rt_dtype_name(mrb, desc->dtype);
  if (desc->ndim > GPU_RT_MAX_DIMS) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "gpu_rt_buffer_create: %d dimensions, and a buffer has at most %d",
      (int)desc->ndim, GPU_RT_MAX_DIMS);
  }

  size_t size = desc->size, count;
  uint32_t ndim = desc->ndim, shape[GPU_RT_MAX_DIMS] = { 0 };
  if (ndim == 0) {
    /* One dimension, as long as the bytes allow. */
    count = size / elem;
    if (size % elem != 0) {
      mrb_raisef(mrb, E_ARGUMENT_ERROR,
        "gpu_rt_buffer_create: %s bytes is not a whole number of %s elements (%d bytes each)",
        size_string(mrb, size), dtype, (int)elem);
    }
    if (count > UINT32_MAX) {
      mrb_raisef(mrb, E_ARGUMENT_ERROR,
        "gpu_rt_buffer_create: %s %s elements, and one dimension holds at most 4294967295",
        size_string(mrb, count), dtype);
    }
    ndim = 1;
    shape[0] = (uint32_t)count;
  } else {
    count = 1;
    for (uint32_t i = 0; i < ndim; i++) shape[i] = desc->shape[i];
    /* A zero extent makes the product 0 whatever the others are, so look for
     * one first: the overflow check below then only sees real products. */
    for (uint32_t i = 0; i < ndim; i++) if (shape[i] == 0) count = 0;
    for (uint32_t i = 0; count != 0 && i < ndim; i++) {
      if (count > SIZE_MAX / elem / shape[i]) {
        mrb_raisef(mrb, E_ARGUMENT_ERROR,
          "gpu_rt_buffer_create: a %s buffer of shape %s is more bytes than a size_t holds",
          dtype, shape_string(mrb, ndim, shape));
      }
      count *= shape[i];
    }
    if (size != 0 && size != count * elem) {
      mrb_raisef(mrb, E_ARGUMENT_ERROR,
        "gpu_rt_buffer_create: size is %s bytes, but shape %s of %s is %s bytes",
        size_string(mrb, size), shape_string(mrb, ndim, shape), dtype,
        size_string(mrb, count * elem));
    }
    size = count * elem;
  }

  GpuBuffer *buf = gpu_buffer_create_bytes(mrb, (VkDeviceSize)size, desc->usage);
  buf->dtype = desc->dtype;
  buf->ndim = ndim;
  buf->count = count;
  memcpy(buf->shape, shape, sizeof(shape));
  return buf;
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

gpu_rt_dtype gpu_rt_buffer_dtype(mrb_state *mrb, gpu_rt_buffer *buf) {
  (void)mrb;
  return buf->dtype;
}

uint32_t gpu_rt_buffer_ndim(mrb_state *mrb, gpu_rt_buffer *buf) {
  (void)mrb;
  return buf->ndim;
}

void gpu_rt_buffer_shape(mrb_state *mrb, gpu_rt_buffer *buf,
                         uint32_t shape_out[GPU_RT_MAX_DIMS]) {
  (void)mrb;
  memcpy(shape_out, buf->shape, sizeof(buf->shape));
}

size_t gpu_rt_buffer_count(mrb_state *mrb, gpu_rt_buffer *buf) {
  (void)mrb;
  return buf->count;
}

void *gpu_rt_buffer_map(mrb_state *mrb, gpu_rt_buffer *buf) {
  return map_buffer(mrb, buf);
}

void gpu_rt_buffer_unmap(mrb_state *mrb, gpu_rt_buffer *buf) {
  (void)mrb;
  if (g_ctx.initialized) unmap_buffer(buf);
}

/* ---- Pipelines ---- */

static gpu_rt_pipeline *g_pipelines = NULL;   /* every pipeline not yet freed */
static uint32_t g_pipelines_released = 0;     /* of those, released and waiting for the batch */

static void pipeline_unlink(gpu_rt_pipeline *p) {
  if (p->prev) p->prev->next = p->next; else g_pipelines = p->next;
  if (p->next) p->next->prev = p->prev;
  p->prev = p->next = NULL;
}

static void pipeline_free(mrb_state *mrb, gpu_rt_pipeline *p) {
  if (p->label) mrb_free(mrb, p->label);
  mrb_free(mrb, p);
}

/* Which of the fixed layouts serves a description. The pipeline layouts all
 * carry GPU_PUSH_CONSTANT_SIZE bytes of push constants, which is valid for a
 * shader that declares fewer. */
static LayoutId layout_for(mrb_state *mrb, const gpu_rt_layout *layout, const char *label) {
  if (!layout) mrb_raise(mrb, E_ARGUMENT_ERROR, "gpu_rt_pipeline_create: no layout");
  if (layout->push_constant_size > GPU_RT_MAX_PUSH_CONSTANT_SIZE) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "gpu_rt_pipeline_create(%s): %d bytes of push constants, and the most is %d",
      label, (int)layout->push_constant_size, GPU_RT_MAX_PUSH_CONSTANT_SIZE);
  }
  switch (layout->storage_buffers) {
    case 1: return LAYOUT_1BUF;
    case 2: return LAYOUT_2BUF;
    case 3: return LAYOUT_3BUF;
    default:
      mrb_raisef(mrb, E_ARGUMENT_ERROR,
        "gpu_rt_pipeline_create(%s): %d storage buffers, and a layout takes 1 to %d",
        label, (int)layout->storage_buffers, GPU_RT_MAX_STORAGE_BUFFERS);
  }
  return LAYOUT_1BUF;   /* not reached */
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
  LayoutId lid = layout_for(mrb, layout, shown);
  if (error_out) memset(error_out, 0, sizeof(*error_out));

  /* Vulkan reads the code as 32-bit words. Handing it anything else is
   * undefined behaviour in the driver, so it is refused here. */
  if (!spirv || spirv_size == 0 || spirv_size % 4 != 0) {
    return pipeline_failed(mrb, shown, "the SPIR-V size check",
                           "the code is empty or not a whole number of 32-bit words",
                           VK_SUCCESS, error_out);
  }

  /* Only this allocation can raise, and nothing exists yet to unwind. The
   * two below report instead (the label is only a name, so it may be lost). */
  gpu_rt_pipeline *p = mrb_malloc(mrb, sizeof(*p));
  memset(p, 0, sizeof(*p));
  p->lid = lid;
  p->storage_buffers = layout->storage_buffers;
  p->push_constant_size = layout->push_constant_size;
  if (label) {
    size_t len = strlen(label) + 1;
    p->label = mrb_malloc_simple(mrb, len);
    if (p->label) memcpy(p->label, label, len);
  }
  /* The caller's bytes may sit at any address; the words must be aligned. */
  uint32_t *words = mrb_malloc_simple(mrb, spirv_size);
  if (!words) {
    pipeline_free(mrb, p);
    return pipeline_failed(mrb, shown, "copying the SPIR-V", "out of host memory",
                           VK_ERROR_OUT_OF_HOST_MEMORY, error_out);
  }
  memcpy(words, spirv, spirv_size);

  VkResult r = gpu_create_pipeline(words, spirv_size, lid, &p->pipeline);
  mrb_free(mrb, words);
  if (r != VK_SUCCESS) {
    pipeline_free(mrb, p);
    return pipeline_failed(mrb, shown, "creating the compute pipeline",
                           gpu_result_name(r), r, error_out);
  }

  p->next = g_pipelines;
  if (g_pipelines) g_pipelines->prev = p;
  g_pipelines = p;
  return p;
}

void gpu_rt_pipeline_release(mrb_state *mrb, gpu_rt_pipeline *p) {
  if (!p || p->released) return;
  if (p->pipeline == VK_NULL_HANDLE) {   /* the device went first (gpu_pipelines_shutdown) */
    pipeline_free(mrb, p);
    return;
  }
  if (g_ctx.batch_recording && p->epoch == g_ctx.epoch) {
    /* The command buffer being recorded binds this pipeline. Destroying it
     * now would make that command buffer invalid; gpu_flush sweeps it once
     * the batch has run. */
    p->released = 1;
    g_pipelines_released++;
    return;
  }
  vkDestroyPipeline(g_ctx.device, p->pipeline, NULL);
  pipeline_unlink(p);
  pipeline_free(mrb, p);
}

void gpu_pipelines_sweep(mrb_state *mrb) {
  if (g_pipelines_released == 0) return;
  for (gpu_rt_pipeline *p = g_pipelines, *next; p; p = next) {
    next = p->next;
    if (!p->released) continue;
    vkDestroyPipeline(g_ctx.device, p->pipeline, NULL);
    pipeline_unlink(p);
    pipeline_free(mrb, p);
  }
  g_pipelines_released = 0;
}

void gpu_pipelines_shutdown(mrb_state *mrb) {
  for (gpu_rt_pipeline *p = g_pipelines, *next; p; p = next) {
    next = p->next;
    vkDestroyPipeline(g_ctx.device, p->pipeline, NULL);
    p->pipeline = VK_NULL_HANDLE;
    p->prev = p->next = NULL;
    /* A released one has no owner left; the others are freed by their
     * owner's release, which now only frees the struct. */
    if (p->released) pipeline_free(mrb, p);
  }
  g_pipelines = NULL;
  g_pipelines_released = 0;
}

/* ---- Dispatch ---- */

void gpu_rt_dispatch(mrb_state *mrb, const gpu_rt_dispatch_desc *d) {
  if (!d || !d->pipeline) mrb_raise(mrb, E_ARGUMENT_ERROR, "gpu_rt_dispatch: no pipeline");
  gpu_rt_pipeline *p = d->pipeline;
  const char *label = d->label ? d->label : (p->label ? p->label : "unnamed");

  if (p->pipeline == VK_NULL_HANDLE || p->released || !g_ctx.initialized) {
    mrb_raisef(mrb, E_RUNTIME_ERROR,
      "gpu_rt_dispatch(%s): the pipeline was released, or the device is gone", label);
  }
  if (d->buffer_count != p->storage_buffers || !d->buffers) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "gpu_rt_dispatch(%s): %d buffers given, and the pipeline's layout binds %d",
      label, (int)(d->buffers ? d->buffer_count : 0), (int)p->storage_buffers);
  }
  for (uint32_t i = 0; i < d->buffer_count; i++) {
    if (!d->buffers[i]) {
      mrb_raisef(mrb, E_ARGUMENT_ERROR, "gpu_rt_dispatch(%s): binding %d has no buffer",
                 label, (int)i);
    }
    if (!(d->buffers[i]->usage & GPU_RT_BUFFER_STORAGE)) {
      mrb_raisef(mrb, E_ARGUMENT_ERROR,
        "gpu_rt_dispatch(%s): the buffer at binding %d was not created with GPU_RT_BUFFER_STORAGE",
        label, (int)i);
    }
  }
  if (d->push_constant_size > p->push_constant_size ||
      (d->push_constant_size > 0 && !d->push_constants)) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "gpu_rt_dispatch(%s): %d bytes of push constants, and the pipeline's layout has %d",
      label, (int)d->push_constant_size, (int)p->push_constant_size);
  }

  dispatch_pipeline(mrb, p->pipeline, p->lid, d->buffers, (int)d->buffer_count,
                    d->push_constants, d->push_constant_size,
                    d->groups[0],
                    d->groups[1] ? d->groups[1] : 1,
                    d->groups[2] ? d->groups[2] : 1);
  /* Marked after the call: the core may flush inside it (an empty descriptor
   * pool before recording, :eager mode after), and what matters to a later
   * release is whether the batch being recorded now binds this pipeline. */
  p->epoch = g_ctx.batch_recording ? g_ctx.epoch : 0;
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
