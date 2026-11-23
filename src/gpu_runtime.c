/* gpu_runtime.c -- the GPU module: device information and the batch.
 *
 * These module functions moved here from mruby-gpu-narray together with the
 * Vulkan core, because they describe the device and the shared batch rather
 * than arrays. Their behaviour is unchanged.
 */
#include "gpu_internal.h"
#include <mruby/array.h>
#include <mruby/hash.h>
#include <mruby/string.h>
#include <stdio.h>
#include <stdlib.h>

static mrb_value gpu_s_device_name(mrb_state *mrb, mrb_value self) {
  gpu_ensure_initialized(mrb);
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(g_ctx.physical_device, &props);
  return mrb_str_new_cstr(mrb, props.deviceName);
}

static mrb_value gpu_s_info(mrb_state *mrb, mrb_value self) {
  gpu_ensure_initialized(mrb);
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(g_ctx.physical_device, &props);

  char api_ver[32];
  snprintf(api_ver, sizeof(api_ver), "%d.%d.%d",
    VK_VERSION_MAJOR(props.apiVersion),
    VK_VERSION_MINOR(props.apiVersion),
    VK_VERSION_PATCH(props.apiVersion));

  mrb_value h = mrb_hash_new(mrb);
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_cstr(mrb, "device")),
               mrb_str_new_cstr(mrb, props.deviceName));
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_cstr(mrb, "api_version")),
               mrb_str_new_cstr(mrb, api_ver));
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_cstr(mrb, "backend")),
               mrb_str_new_cstr(mrb, "Vulkan"));
  /* The dispatch bound, so callers (and bug reports) can see the largest
   * array a single operation can cover: max_workgroups * 256 elements. */
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_cstr(mrb, "max_workgroups")),
               mrb_fixnum_value((mrb_int)g_ctx.max_workgroups));
  /* Which of GPU.devices this is. */
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_lit(mrb, "device_index")),
               mrb_fixnum_value((mrb_int)g_ctx.device_index));
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_lit(mrb, "device_type")),
               mrb_symbol_value(mrb_intern_cstr(mrb, gpu_device_type_name(g_ctx.device_type))));
  /* The optional features, true when the device has one and the runtime has
   * turned it on: the same answers as gpu_rt_context_has_feature. The keys
   * are part of the interface (README.md, "Devices and features"). */
  mrb_value f = mrb_hash_new(mrb);
  mrb_hash_set(mrb, f, mrb_symbol_value(mrb_intern_lit(mrb, "storage_8bit")),
               mrb_bool_value(g_ctx.features[GPU_RT_FEATURE_STORAGE_8BIT]));
  mrb_hash_set(mrb, f, mrb_symbol_value(mrb_intern_lit(mrb, "timestamp_query")),
               mrb_bool_value(g_ctx.features[GPU_RT_FEATURE_TIMESTAMP_QUERY]));
  mrb_hash_set(mrb, f, mrb_symbol_value(mrb_intern_lit(mrb, "subgroups")),
               mrb_bool_value(g_ctx.features[GPU_RT_FEATURE_SUBGROUPS]));
  mrb_hash_set(mrb, f, mrb_symbol_value(mrb_intern_lit(mrb, "memory_budget")),
               mrb_bool_value(g_ctx.features[GPU_RT_FEATURE_MEMORY_BUDGET]));
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_lit(mrb, "features")), f);
  return h;
}

/* GPU.devices -> [{ index: 0, name: "V3D 7.1.10.2", type: :integrated }, ...]
 *
 * Every Vulkan device of this machine, in index order. type is :integrated,
 * :discrete, :virtual, :cpu (a software rasterizer) or :other. Does not start
 * the device, so it can be called before GPU.device=. */
static mrb_value gpu_s_devices(mrb_state *mrb, mrb_value self) {
  uint32_t count = 0;
  GpuDeviceEntry *entries = gpu_devices_list(mrb, &count);
  /* Copied out first: building the Ruby objects can raise, and `entries` is
   * not the GC's. A machine has a handful of devices. */
  enum { MAX_LISTED = 64 };
  static GpuDeviceEntry listed[MAX_LISTED];
  if (count > MAX_LISTED) count = MAX_LISTED;
  for (uint32_t i = 0; i < count; i++) listed[i] = entries[i];
  free(entries);

  mrb_value ary = mrb_ary_new_capa(mrb, (mrb_int)count);
  for (uint32_t i = 0; i < count; i++) {
    mrb_value d = mrb_hash_new(mrb);
    mrb_hash_set(mrb, d, mrb_symbol_value(mrb_intern_lit(mrb, "index")),
                 mrb_fixnum_value((mrb_int)i));
    mrb_hash_set(mrb, d, mrb_symbol_value(mrb_intern_lit(mrb, "name")),
                 mrb_str_new_cstr(mrb, listed[i].name));
    mrb_hash_set(mrb, d, mrb_symbol_value(mrb_intern_lit(mrb, "type")),
                 mrb_symbol_value(mrb_intern_cstr(mrb, gpu_device_type_name(listed[i].type))));
    mrb_ary_push(mrb, ary, d);
  }
  return ary;
}

/* GPU.device = index | name | nil
 *
 * Choose the device the context starts on: an index of GPU.devices, a name
 * (equal ignoring case, or else contained in the device's name), or nil for
 * the default again (MRUBY_GPU_DEVICE, or the first device that is not a
 * CPU). Call it before the first GPU operation. Once the device has started
 * only the device in use is accepted, and anything else raises RuntimeError.
 * Raises ArgumentError, with the list of devices, when none matches, and an
 * earlier choice then stays. nil raises RuntimeError when MRUBY_GPU_DEVICE
 * matches no device. */
static mrb_value gpu_s_set_device(mrb_state *mrb, mrb_value self) {
  mrb_value v;
  mrb_get_args(mrb, "o", &v);
  if (mrb_nil_p(v)) {
    gpu_device_choose(mrb, 0, 0, NULL);
  } else if (mrb_integer_p(v)) {
    gpu_device_choose(mrb, 1, mrb_integer(v), NULL);
  } else if (mrb_string_p(v)) {
    gpu_device_choose(mrb, 0, 0, mrb_string_value_cstr(mrb, &v));
  } else {
    mrb_raisef(mrb, E_TYPE_ERROR,
      "GPU.device= takes an index (Integer), a name (String) or nil, got %s",
      mrb_obj_classname(mrb, v));
  }
  return v;
}

/* GPU.sync -> nil. Run everything recorded so far and wait for it. Reading a
 * result does this implicitly; call it yourself to time a batch, or before
 * handing a buffer to something outside this library. */
static mrb_value gpu_s_sync(mrb_state *mrb, mrb_value self) {
  gpu_flush(mrb);
  return mrb_nil_value();
}

/* GPU.pending -> Integer: dispatches recorded but not yet submitted. */
static mrb_value gpu_s_pending(mrb_state *mrb, mrb_value self) {
  return mrb_fixnum_value((mrb_int)g_ctx.batch_dispatches);
}

/* GPU.sync_mode -> :deferred | :eager
 *
 * :deferred (default) records dispatches and submits them together when a
 * result is needed. :eager submits and waits after every dispatch -- the
 * behaviour before batching existed, kept for measuring the difference. */
static mrb_value gpu_s_sync_mode(mrb_state *mrb, mrb_value self) {
  /* Two literals, not one ternary: mrb_intern_lit takes sizeof its argument. */
  return mrb_symbol_value(g_ctx.eager ? mrb_intern_lit(mrb, "eager")
                                      : mrb_intern_lit(mrb, "deferred"));
}

static mrb_value gpu_s_set_sync_mode(mrb_state *mrb, mrb_value self) {
  mrb_sym mode;
  mrb_get_args(mrb, "n", &mode);
  if (mode == mrb_intern_lit(mrb, "eager")) {
    g_ctx.eager = 1;
    gpu_flush(mrb);   /* nothing may stay queued once the mode says nothing is */
  } else if (mode == mrb_intern_lit(mrb, "deferred")) {
    g_ctx.eager = 0;
  } else {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "GPU.sync_mode must be :deferred or :eager, got :%n", mode);
  }
  return mrb_symbol_value(mode);
}

void mrb_mruby_gpu_runtime_gem_init(mrb_state *mrb) {
  struct RClass *gpu = mrb_define_module(mrb, "GPU");
  mrb_define_module_function(mrb, gpu, "device_name", gpu_s_device_name, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, gpu, "info",        gpu_s_info,        MRB_ARGS_NONE());
  mrb_define_module_function(mrb, gpu, "sync",        gpu_s_sync,        MRB_ARGS_NONE());
  mrb_define_module_function(mrb, gpu, "pending",     gpu_s_pending,     MRB_ARGS_NONE());
  mrb_define_module_function(mrb, gpu, "sync_mode",   gpu_s_sync_mode,   MRB_ARGS_NONE());
  mrb_define_module_function(mrb, gpu, "sync_mode=",  gpu_s_set_sync_mode, MRB_ARGS_REQ(1));
  mrb_define_module_function(mrb, gpu, "devices",     gpu_s_devices,     MRB_ARGS_NONE());
  mrb_define_module_function(mrb, gpu, "device=",     gpu_s_set_device,  MRB_ARGS_REQ(1));
}

void mrb_mruby_gpu_runtime_gem_final(mrb_state *mrb) {
  gpu_shutdown(mrb);
}
