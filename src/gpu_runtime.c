/* gpu_runtime.c -- the GPU module: device information and the batch.
 *
 * These module functions moved here from mruby-gpu-narray together with the
 * Vulkan core, because they describe the device and the shared batch rather
 * than arrays. Their behaviour is unchanged.
 */
#include "gpu_internal.h"
#include <mruby/hash.h>
#include <mruby/string.h>
#include <stdio.h>

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
  return h;
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

  /* The exception classes of every GPU gem (gpu_runtime.h, "Errors"). Defined
   * here so that they exist before the gems that raise them are initialised. */
  struct RClass *error = mrb_define_class_under(mrb, gpu, "Error", E_STANDARD_ERROR);
  mrb_define_class_under(mrb, gpu, "DeviceError",      error);
  mrb_define_class_under(mrb, gpu, "OutOfMemoryError", error);
  mrb_define_class_under(mrb, gpu, "UnsupportedError", error);
  mrb_define_class_under(mrb, gpu, "CompileError",     error);
  mrb_define_class_under(mrb, gpu, "ShapeError",       E_ARGUMENT_ERROR);
}

void mrb_mruby_gpu_runtime_gem_final(mrb_state *mrb) {
  gpu_shutdown(mrb);
}
