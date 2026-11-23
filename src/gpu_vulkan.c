/* gpu_vulkan.c -- Vulkan initialization, pipelines, and the deferred batch.
 *
 * Moved from mruby-gpu-narray (src/gpu_vulkan.c), which had it from
 * yujiteshima/mruby-gpu. The initialization and dispatch logic is unchanged.
 * What stayed behind in narray is its own table of built-in pipelines: the
 * runtime knows the three descriptor layouts, not which shaders use them.
 */
#include "gpu_internal.h"
#include <mruby/error.h>
#include <mruby/string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

GpuCtx g_ctx = {0};

/* ---- Error handling ----
 *
 * The Vulkan SDK ships vk_enum_string_helper.h for this, but relying on it
 * would add an SDK dependency to a gem whose only link dependency is the
 * loader, so the handful of codes that can actually reach us live here. */
const char *gpu_result_name(VkResult r) {
  switch (r) {
    case VK_SUCCESS:                      return "success";
    case VK_NOT_READY:                    return "not ready";
    case VK_TIMEOUT:                      return "timeout";
    case VK_INCOMPLETE:                   return "incomplete";
    case VK_ERROR_OUT_OF_HOST_MEMORY:     return "out of host memory";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:   return "out of device memory";
    case VK_ERROR_INITIALIZATION_FAILED:  return "initialization failed";
    case VK_ERROR_DEVICE_LOST:            return "device lost";
    case VK_ERROR_MEMORY_MAP_FAILED:      return "memory map failed";
    case VK_ERROR_LAYER_NOT_PRESENT:      return "layer not present";
    case VK_ERROR_EXTENSION_NOT_PRESENT:  return "extension not present";
    case VK_ERROR_FEATURE_NOT_PRESENT:    return "feature not present";
    case VK_ERROR_INCOMPATIBLE_DRIVER:    return "incompatible driver";
    case VK_ERROR_TOO_MANY_OBJECTS:       return "too many objects";
    case VK_ERROR_FRAGMENTED_POOL:        return "descriptor pool fragmented";
    case VK_ERROR_UNKNOWN:                return "unknown error";
    default: break;
  }
  /* Results newer than Vulkan 1.0, by value: an older SDK header may not
   * name them, and a newer driver may still return them. */
  switch ((int)r) {
    case -11:         return "format not supported";      /* VK_ERROR_FORMAT_NOT_SUPPORTED */
    case -1000012000: return "invalid shader";            /* VK_ERROR_INVALID_SHADER_NV */
    case -1000069000: return "out of pool memory";        /* VK_ERROR_OUT_OF_POOL_MEMORY */
    case -1000161000: return "fragmentation";             /* VK_ERROR_FRAGMENTATION */
    default:          return "unrecognised VkResult";
  }
}

/* Which exception class a VkResult stands for. Listed by value for the same
 * reason as above. Anything not listed is GPU::Error itself, unless the
 * caller knows more (gpu_fail during initialisation, gpu_rt_pipeline_create). */
gpu_rt_error_kind gpu_result_kind(VkResult r) {
  switch ((int)r) {
    case -1:          /* VK_ERROR_OUT_OF_HOST_MEMORY */
    case -2:          /* VK_ERROR_OUT_OF_DEVICE_MEMORY */
    case -5:          /* VK_ERROR_MEMORY_MAP_FAILED: no address space left to map into */
    case -10:         /* VK_ERROR_TOO_MANY_OBJECTS: e.g. more allocations than the device allows */
    case -12:         /* VK_ERROR_FRAGMENTED_POOL */
    case -1000069000: /* VK_ERROR_OUT_OF_POOL_MEMORY */
    case -1000161000: /* VK_ERROR_FRAGMENTATION */
      return GPU_RT_ERROR_OUT_OF_MEMORY;
    case -3:          /* VK_ERROR_INITIALIZATION_FAILED */
    case -4:          /* VK_ERROR_DEVICE_LOST */
    case -9:          /* VK_ERROR_INCOMPATIBLE_DRIVER: no driver for the Vulkan asked for */
      return GPU_RT_ERROR_DEVICE;
    case -6:          /* VK_ERROR_LAYER_NOT_PRESENT */
    case -7:          /* VK_ERROR_EXTENSION_NOT_PRESENT */
    case -8:          /* VK_ERROR_FEATURE_NOT_PRESENT */
    case -11:         /* VK_ERROR_FORMAT_NOT_SUPPORTED */
      return GPU_RT_ERROR_UNSUPPORTED;
    case -1000012000: /* VK_ERROR_INVALID_SHADER_NV */
      return GPU_RT_ERROR_COMPILE;
    default:
      return GPU_RT_ERROR_GENERIC;
  }
}

void gpu_fail(mrb_state *mrb, VkResult r, const char *call, const char *operation) {
  gpu_rt_error_kind kind = gpu_result_kind(r);
  /* Before the context is ready the callers are gpu_init and the listing of
   * devices (GPU.devices, GPU.device=), and whatever stops them leaves the
   * process without a device. */
  if (!g_ctx.initialized) {
    if (kind == GPU_RT_ERROR_GENERIC) kind = GPU_RT_ERROR_DEVICE;
    if (!operation) operation = "starting the GPU";
  }
  struct RClass *c = gpu_rt_error_class(mrb, kind);
  if (operation) {
    mrb_raisef(mrb, c, "%s: %s failed: %s (VkResult %d)",
               operation, call, gpu_result_name(r), (int)r);
  }
  mrb_raisef(mrb, c, "%s failed: %s (VkResult %d)", call, gpu_result_name(r), (int)r);
}

void gpu_check(mrb_state *mrb, VkResult r, const char *call) {
  if (r == VK_SUCCESS) return;
  gpu_fail(mrb, r, call, NULL);
}

/* Portability (MoltenVK on macOS) support. Absent on the Raspberry Pi's native
 * V3D driver, so everything below is gated on runtime extension detection and
 * has no effect there. Provide fallback macros for older SDK headers. */
#ifndef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
#define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME "VK_KHR_portability_enumeration"
#endif
#ifndef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
#define VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR 0x00000001
#endif
#define GPU_PORTABILITY_SUBSET_EXT "VK_KHR_portability_subset"

/* An extension that cannot be enumerated is treated as absent: the caller only
 * uses these to opt into optional behaviour, so failing closed is correct and
 * lets the Pi's plain path keep working. */
static int has_instance_ext(const char *name) {
  uint32_t count = 0;
  if (vkEnumerateInstanceExtensionProperties(NULL, &count, NULL) != VK_SUCCESS) return 0;
  if (count == 0) return 0;
  VkExtensionProperties *props = malloc(sizeof(VkExtensionProperties) * count);
  if (!props) return 0;
  vkEnumerateInstanceExtensionProperties(NULL, &count, props);
  int found = 0;
  for (uint32_t i = 0; i < count; i++) {
    if (strcmp(props[i].extensionName, name) == 0) { found = 1; break; }
  }
  free(props);
  return found;
}

static int has_device_ext(VkPhysicalDevice dev, const char *name) {
  uint32_t count = 0;
  if (vkEnumerateDeviceExtensionProperties(dev, NULL, &count, NULL) != VK_SUCCESS) return 0;
  if (count == 0) return 0;
  VkExtensionProperties *props = malloc(sizeof(VkExtensionProperties) * count);
  if (!props) return 0;
  vkEnumerateDeviceExtensionProperties(dev, NULL, &count, props);
  int found = 0;
  for (uint32_t i = 0; i < count; i++) {
    if (strcmp(props[i].extensionName, name) == 0) { found = 1; break; }
  }
  free(props);
  return found;
}

/* ---- Pipelines ----
 *
 * A pipeline is created against one of the fixed layouts, so the caller only
 * brings the SPIR-V. Failures are returned rather than raised: mruby-gpu-narray
 * builds all of its pipelines at once and reports a missing or rejected
 * shader only when an operation needs it. */
VkResult gpu_create_pipeline(const uint32_t *code, size_t size, LayoutId lid,
                             VkPipeline *out) {
  *out = VK_NULL_HANDLE;
  if (!g_ctx.initialized) return VK_ERROR_INITIALIZATION_FAILED;
  if (lid < 0 || lid >= LAYOUT_COUNT) return VK_ERROR_UNKNOWN;

  VkShaderModuleCreateInfo sm_info = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = size,
    .pCode = code
  };
  VkShaderModule shader;
  VkResult r = vkCreateShaderModule(g_ctx.device, &sm_info, NULL, &shader);
  if (r != VK_SUCCESS) return r;

  VkComputePipelineCreateInfo cp_info = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = shader,
      .pName = "main"
    },
    .layout = g_ctx.pipe_layouts[lid]
  };
  r = vkCreateComputePipelines(g_ctx.device, VK_NULL_HANDLE, 1, &cp_info, NULL, out);
  vkDestroyShaderModule(g_ctx.device, shader, NULL);
  if (r != VK_SUCCESS) *out = VK_NULL_HANDLE;
  return r;
}

/* ---- Deferred submission ----
 *
 * A dispatch is no longer a round trip. It is recorded into one long-lived
 * command buffer, followed by a memory barrier so the next dispatch sees its
 * writes, and nothing is submitted until the host actually needs a result:
 * a host read or write of a buffer the batch touches (map_buffer), a
 * descriptor pool that has run dry, an explicit GPU.sync, or shutdown. Then
 * gpu_flush ends the command buffer, submits it once, waits once, and resets
 * the pool for the next batch.
 *
 * `a * 2 + 1 - 3 + 4` therefore costs four dispatches and one wait instead of
 * four waits; an FFT's log2(n) passes cost one wait instead of log2(n).
 * GPU.sync_mode = :eager flushes after every dispatch -- the pre-batching
 * behaviour, kept so the two can be measured against each other.
 *
 * A buffer remembers the epoch (batch number) it was last bound in. That is
 * how map_buffer knows whether a flush is due, and how the GC finalizer knows
 * a buffer must outlive its Ruby owner until the batch has run. */

static void batch_begin(mrb_state *mrb, const char *op) {
  if (g_ctx.batch_recording) return;
  VkCommandBufferBeginInfo begin = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
  };
  /* The pool was created with RESET_COMMAND_BUFFER_BIT, so beginning a
   * command buffer that has already been submitted resets it implicitly. */
  VkResult r = vkBeginCommandBuffer(g_ctx.batch_cmd, &begin);
  if (r != VK_SUCCESS) gpu_fail(mrb, r, "vkBeginCommandBuffer", op);
  g_ctx.batch_recording = 1;
}

/* One descriptor set from the pool. A full pool just means the batch is
 * flushed a little early: the sets it held are released by the reset. */
static VkDescriptorSet alloc_desc_set(mrb_state *mrb, LayoutId lid, const char *op) {
  if (g_ctx.batch_sets >= GPU_MAX_DESC_SETS) gpu_flush(mrb);
  VkDescriptorSetAllocateInfo dsai = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = g_ctx.desc_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &g_ctx.desc_layouts[lid]
  };
  VkDescriptorSet set = VK_NULL_HANDLE;
  VkResult r = vkAllocateDescriptorSets(g_ctx.device, &dsai, &set);
  if (r == VK_ERROR_FRAGMENTED_POOL
#ifdef VK_ERROR_OUT_OF_POOL_MEMORY
      || r == VK_ERROR_OUT_OF_POOL_MEMORY
#endif
  ) {
    /* Our count and the driver's disagree; a flush resets the pool either way. */
    gpu_flush(mrb);
    r = vkAllocateDescriptorSets(g_ctx.device, &dsai, &set);
  }
  if (r != VK_SUCCESS) gpu_fail(mrb, r, "vkAllocateDescriptorSets", op);
  g_ctx.batch_sets++;
  return set;
}

void dispatch_pipeline(
    mrb_state *mrb, VkPipeline pipeline, LayoutId lid,
    GpuBuffer **bufs, int num_buffers,
    const void *push_data, uint32_t push_size,
    uint32_t group_x, uint32_t group_y, uint32_t group_z,
    const char *label)
{
  /* Exceeding maxComputeWorkGroupCount is invalid usage, and what happens is
   * up to the driver: lavapipe runs the dispatch anyway and returns the right
   * answer, so the limit is easy to miss in testing, while a driver bounded by
   * a hardware register can clamp or fault instead. Reject it here rather than
   * let the answer depend on which GPU the code lands on. */
  if (group_x > g_ctx.max_workgroups) {
    /* Widened to mrb_int first: the element count overflows uint32_t on
     * devices that report a workgroup limit in the billions. */
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "array too large for one dispatch: %i workgroups needed, device allows %i "
      "(about %i elements)",
      (mrb_int)group_x, (mrb_int)g_ctx.max_workgroups,
      (mrb_int)g_ctx.max_workgroups * 256);
  }
  if (pipeline == VK_NULL_HANDLE) {
    GPU_RAISE(mrb, GPU_RT_ERROR_GENERIC, "dispatch of a pipeline that was never created");
  }
  if (num_buffers < 1 || num_buffers > 3) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR, "dispatch takes 1..3 buffers, got %d", num_buffers);
  }

  /* What a Vulkan failure below is reported under: the same prefix as the
   * checks in gpu_rt_dispatch. */
  char op[96];
  snprintf(op, sizeof(op), "gpu_rt_dispatch(%s)", label ? label : "unnamed");

  VkDescriptorSet desc_set = alloc_desc_set(mrb, lid, op);

  VkDescriptorBufferInfo buf_infos[3];
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < num_buffers; i++) {
    buf_infos[i] = (VkDescriptorBufferInfo){bufs[i]->buffer, 0, bufs[i]->bytes};
    writes[i] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = desc_set,
      .dstBinding = i,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &buf_infos[i]
    };
  }
  vkUpdateDescriptorSets(g_ctx.device, num_buffers, writes, 0, NULL);

  batch_begin(mrb, op);
  VkCommandBuffer cmd = g_ctx.batch_cmd;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
    g_ctx.pipe_layouts[lid], 0, 1, &desc_set, 0, NULL);
  if (push_size > 0) {
    vkCmdPushConstants(cmd, g_ctx.pipe_layouts[lid], VK_SHADER_STAGE_COMPUTE_BIT,
      0, push_size, push_data);
  }
  vkCmdDispatch(cmd, group_x, group_y, group_z);

  /* Make this dispatch's writes visible to whatever is recorded next. One
   * global barrier per dispatch is coarser than tracking buffers one by one,
   * but it costs nothing next to the submit + fence wait it replaces. */
  VkMemoryBarrier barrier = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
  };
  vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0, 1, &barrier, 0, NULL, 0, NULL);

  for (int i = 0; i < num_buffers; i++) bufs[i]->epoch = g_ctx.epoch;
  g_ctx.batch_dispatches++;

  if (g_ctx.eager) gpu_flush(mrb);
}

/* Free what the GC could not: buffers whose owners died while a batch still
 * referenced them. Only called once no work is pending. */
static void bury_graveyard(mrb_state *mrb) {
  for (size_t i = 0; i < g_ctx.graveyard_len; i++) {
    GpuBuffer *b = g_ctx.graveyard[i];
    vkDestroyBuffer(g_ctx.device, b->buffer, NULL);
    vkFreeMemory(g_ctx.device, b->memory, NULL);
    mrb_free(mrb, b);
  }
  g_ctx.graveyard_len = 0;
}

void gpu_flush(mrb_state *mrb) {
  if (!g_ctx.initialized) return;
  VkResult r = VK_SUCCESS;
  const char *step = NULL;

  if (g_ctx.batch_recording) {
    step = "vkEndCommandBuffer";
    r = vkEndCommandBuffer(g_ctx.batch_cmd);
    if (r == VK_SUCCESS) {
      VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &g_ctx.batch_cmd
      };
      step = "vkQueueSubmit";
      r = vkQueueSubmit(g_ctx.queue, 1, &si, g_ctx.batch_fence);
    }
    if (r == VK_SUCCESS) {
      /* Without this check a lost device returns immediately and the caller
       * reads whatever is in the buffer as a valid result. */
      step = "vkWaitForFences";
      r = vkWaitForFences(g_ctx.device, 1, &g_ctx.batch_fence, VK_TRUE, UINT64_MAX);
      if (r == VK_SUCCESS) vkResetFences(g_ctx.device, 1, &g_ctx.batch_fence);
    }
    /* Whatever happened, this command buffer is spent; the next batch_begin
     * resets it. */
    g_ctx.batch_recording = 0;
  }

  /* No work is pending from here on (either it completed or it never left
   * the host), so the sets and the graveyard can go. */
  g_ctx.batch_dispatches = 0;
  if (g_ctx.batch_sets > 0) {
    vkResetDescriptorPool(g_ctx.device, g_ctx.desc_pool, 0);
    g_ctx.batch_sets = 0;
  }
  g_ctx.epoch++;
  bury_graveyard(mrb);
  gpu_pipelines_sweep(mrb);

  if (step && r != VK_SUCCESS) gpu_fail(mrb, r, step, "running the batch");
}

void gpu_sync_buffer(mrb_state *mrb, GpuBuffer *buf) {
  if (g_ctx.batch_recording && buf->epoch == g_ctx.epoch) gpu_flush(mrb);
}

/* ---- Devices ----
 *
 * Which physical device the context runs on is decided once, when the device
 * starts (gpu_init), from the first of these that says something:
 *
 *   1. GPU.device = index or name, called before the first GPU operation;
 *   2. the environment variable MRUBY_GPU_DEVICE, an index or a name, so a
 *      script that makes no choice runs unchanged on another device;
 *   3. the first device that is not a CPU (a software rasterizer such as
 *      llvmpipe), or device 0 when every device is one.
 *
 * An index is the position in vkEnumeratePhysicalDevices' answer, which is
 * also the order of GPU.devices. A name matches a device whose name is equal
 * to it, ignoring case, and otherwise a device whose name contains it; with
 * several matches the lowest index wins. Listing and choosing need only the
 * instance, so neither starts the device.
 *
 * Once the device has started it cannot change: buffers and pipelines of
 * every GPU gem live on it. Choosing the device already in use is accepted,
 * and anything else raises. */

#define GPU_DEVICE_ENV "MRUBY_GPU_DEVICE"

static void instance_ensure(mrb_state *mrb) {
  if (g_ctx.instance_ready) return;
  VkApplicationInfo app_info = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName = "mruby-gpu-runtime",
    .apiVersion = VK_API_VERSION_1_1
  };
  /* Opt into portability enumeration if the loader exposes it (macOS/MoltenVK).
   * On the Pi this extension is absent, so we take the plain path. */
  const char *inst_exts[1];
  uint32_t inst_ext_count = 0;
  VkInstanceCreateFlags inst_flags = 0;
  if (has_instance_ext(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
    inst_exts[inst_ext_count++] = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    inst_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  }
  VkInstanceCreateInfo inst_info = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &app_info,
    .flags = inst_flags,
    .enabledExtensionCount = inst_ext_count,
    .ppEnabledExtensionNames = inst_ext_count ? inst_exts : NULL
  };
  VkResult r = vkCreateInstance(&inst_info, NULL, &g_ctx.instance);
  if (r != VK_SUCCESS) gpu_fail(mrb, r, "vkCreateInstance", "starting Vulkan");
  g_ctx.instance_ready = 1;
}

/* The instance's physical devices, in a malloc'ed array the caller frees. */
static VkPhysicalDevice *physical_devices(mrb_state *mrb, uint32_t *count) {
  instance_ensure(mrb);
  uint32_t dev_count = 0;
  VkResult count_r = vkEnumeratePhysicalDevices(g_ctx.instance, &dev_count, NULL);
  if (count_r != VK_SUCCESS) {
    gpu_fail(mrb, count_r, "vkEnumeratePhysicalDevices", "listing the Vulkan devices");
  }
  if (dev_count == 0) {
    GPU_RAISE(mrb, GPU_RT_ERROR_DEVICE,
      "no Vulkan physical device found (is a Vulkan driver installed?)");
  }
  VkPhysicalDevice *devs = malloc(sizeof(VkPhysicalDevice) * dev_count);
  if (!devs) GPU_RAISE(mrb, GPU_RT_ERROR_OUT_OF_MEMORY, "out of memory enumerating devices");
  VkResult dev_r = vkEnumeratePhysicalDevices(g_ctx.instance, &dev_count, devs);
  /* VK_INCOMPLETE means a device appeared between the two calls; the ones
   * returned are valid. */
  if ((dev_r != VK_SUCCESS && dev_r != VK_INCOMPLETE) || dev_count == 0) {
    free(devs);
    gpu_fail(mrb, dev_r == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : dev_r,
             "vkEnumeratePhysicalDevices", "listing the Vulkan devices");
  }
  *count = dev_count;
  return devs;
}

GpuDeviceEntry *gpu_devices_list(mrb_state *mrb, uint32_t *count) {
  VkPhysicalDevice *devs = physical_devices(mrb, count);
  GpuDeviceEntry *entries = malloc(sizeof(GpuDeviceEntry) * *count);
  if (!entries) {
    free(devs);
    GPU_RAISE(mrb, GPU_RT_ERROR_OUT_OF_MEMORY, "out of memory enumerating devices");
  }
  for (uint32_t i = 0; i < *count; i++) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(devs[i], &p);
    memcpy(entries[i].name, p.deviceName, sizeof(entries[i].name));
    entries[i].name[sizeof(entries[i].name) - 1] = '\0';
    entries[i].type = p.deviceType;
  }
  free(devs);
  return entries;
}

const char *gpu_device_type_name(VkPhysicalDeviceType type) {
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "cpu";
    default:                                     return "other";
  }
}

static char lower(char c) { return (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c; }

/* Whether `hay` contains `needle`, ignoring ASCII case; with `whole`, whether
 * they are equal. */
static int name_matches(const char *hay, const char *needle, int whole) {
  size_t hn = strlen(hay), nn = strlen(needle);
  if (nn == 0 || nn > hn || (whole && nn != hn)) return 0;
  for (size_t at = 0; at + nn <= hn; at++) {
    size_t i = 0;
    while (i < nn && lower(hay[at + i]) == lower(needle[i])) i++;
    if (i == nn) return 1;
    if (whole) break;
  }
  return 0;
}

/* The index of the device `name` means, or -1. */
static int device_by_name(const GpuDeviceEntry *entries, uint32_t count, const char *name) {
  for (int whole = 1; whole >= 0; whole--) {
    for (uint32_t i = 0; i < count; i++) {
      if (name_matches(entries[i].name, name, whole)) return (int)i;
    }
  }
  return -1;
}

/* Free `entries` and raise `klass` with `what` followed by the list of
 * devices, so the message shows what could have been chosen. */
static void raise_with_devices(mrb_state *mrb, struct RClass *klass, const char *what,
                               GpuDeviceEntry *entries, uint32_t count) {
  mrb_value msg = mrb_str_new_cstr(mrb, what);
  mrb_str_cat_cstr(mrb, msg, " The devices are:");
  for (uint32_t i = 0; i < count; i++) {
    char line[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE + 64];
    snprintf(line, sizeof(line), "%s %u: \"%s\" (%s)", i ? "," : "", (unsigned)i,
             entries[i].name, gpu_device_type_name(entries[i].type));
    mrb_str_cat_cstr(mrb, msg, line);
  }
  free(entries);
  mrb_exc_raise(mrb, mrb_exc_new_str(mrb, klass, msg));
}

/* The device used when GPU.device= has said nothing: MRUBY_GPU_DEVICE, or
 * the first device that is not a CPU. Raises, after freeing `entries`, when
 * the variable matches no device. */
static uint32_t default_device(mrb_state *mrb, GpuDeviceEntry *entries, uint32_t count) {
  const char *env = getenv(GPU_DEVICE_ENV);
  if (env && env[0] != '\0') {
    int found = -1;
    if (strspn(env, "0123456789") == strlen(env)) {
      unsigned long n = strtoul(env, NULL, 10);
      if (strlen(env) <= 9 && n < count) found = (int)n;
    } else {
      found = device_by_name(entries, count, env);
    }
    if (found < 0) {
      char what[256];
      snprintf(what, sizeof(what),
        "no Vulkan device matches the environment variable " GPU_DEVICE_ENV "=%.100s.", env);
      raise_with_devices(mrb, gpu_rt_error_class(mrb, GPU_RT_ERROR_DEVICE), what, entries,
                         count);
    }
    return (uint32_t)found;
  }
  /* Prefer a real GPU over a software rasterizer. The Pi exposes both V3D
   * (hardware) and llvmpipe (CPU); pick the first non-CPU device so compute
   * actually runs on the GPU. Falls back to the first device if every device
   * is CPU-type. */
  for (uint32_t i = 0; i < count; i++) {
    if (entries[i].type != VK_PHYSICAL_DEVICE_TYPE_CPU) return i;
  }
  return 0;
}

void gpu_device_choose(mrb_state *mrb, int by_index, mrb_int index, const char *name) {
  uint32_t count = 0;
  GpuDeviceEntry *entries = gpu_devices_list(mrb, &count);
  uint32_t chosen;
  char what[256];

  if (by_index) {
    if (index < 0 || (uint64_t)index >= count) {
      snprintf(what, sizeof(what), "GPU.device = %ld: no Vulkan device has this index.",
               (long)index);
      raise_with_devices(mrb, E_ARGUMENT_ERROR, what, entries, count);
    }
    chosen = (uint32_t)index;
  } else if (name) {
    int found = device_by_name(entries, count, name);
    if (found < 0) {
      snprintf(what, sizeof(what), "GPU.device = \"%.100s\": no Vulkan device has this name.",
               name);
      raise_with_devices(mrb, E_ARGUMENT_ERROR, what, entries, count);
    }
    chosen = (uint32_t)found;
  } else {
    /* nil forgets an earlier choice even when the default then turns out to
     * be an MRUBY_GPU_DEVICE that matches nothing: the start reports that. */
    if (!g_ctx.initialized) g_ctx.choice_set = 0;
    chosen = default_device(mrb, entries, count);
  }

  if (g_ctx.initialized && chosen != g_ctx.device_index) {
    snprintf(what, sizeof(what),
      "GPU.device=: the device has already started on %u: \"%.100s\", and cannot change to "
      "%u. Choose before the first GPU operation, or set " GPU_DEVICE_ENV ".",
      (unsigned)g_ctx.device_index,
      g_ctx.device_index < count ? entries[g_ctx.device_index].name : "?", (unsigned)chosen);
    raise_with_devices(mrb, gpu_rt_error_class(mrb, GPU_RT_ERROR_GENERIC), what, entries,
                       count);
  }
  free(entries);
  g_ctx.choice_set = (by_index || name) ? 1 : 0;
  g_ctx.choice_index = chosen;
}

/* ---- Features ----
 *
 * What gpu_rt_context_has_feature and GPU.info[:features] answer. Two of the
 * four need nothing turned on, and the other two are turned on at device
 * creation whenever the device has them, so that a gem can rely on a "yes":
 *
 *   storage_8bit     VK_KHR_8bit_storage with storageBuffer8BitAccess; the
 *                    extension and the feature are enabled. (A Vulkan 1.2
 *                    device has the feature in core, but the instance asks
 *                    for 1.1, where it is reached through the extension.)
 *   timestamp_query  the compute queue family has timestampValidBits, and
 *                    timestampPeriod is positive. Nothing to enable.
 *   subgroups        VkPhysicalDeviceSubgroupProperties: the compute stage
 *                    with at least the basic operations. Nothing to enable.
 *   memory_budget    VK_EXT_memory_budget; the extension is enabled.
 *
 * Fallback names for SDK headers that predate an extension, as above. */
#define GPU_8BIT_STORAGE_EXT  "VK_KHR_8bit_storage"
#define GPU_MEMORY_BUDGET_EXT "VK_EXT_memory_budget"

/* ---- Init ---- */
void gpu_init(mrb_state *mrb) {
  if (g_ctx.initialized) return;
  /* A half-built context cannot be reused, and retrying would leak everything
   * the failed attempt created -- once per operation, since initialization is
   * lazy. One attempt per process; fix the environment and restart. */
  if (g_ctx.init_failed) {
    GPU_RAISE(mrb, GPU_RT_ERROR_DEVICE,
      "GPU initialization already failed in this process and is not retried");
  }

  /* Instance and physical device. Nothing here leaves anything half-built
   * (the instance is kept and reused), so a failure up to the device's
   * creation can be retried, for example after choosing another device. */
  uint32_t dev_count = 0;
  GpuDeviceEntry *entries = gpu_devices_list(mrb, &dev_count);
  uint32_t dev_index = g_ctx.choice_set ? g_ctx.choice_index
                                        : default_device(mrb, entries, dev_count);
  free(entries);
  VkPhysicalDevice *devs = physical_devices(mrb, &dev_count);
  if (dev_index >= dev_count) {
    free(devs);
    mrb_raisef(mrb, gpu_rt_error_class(mrb, GPU_RT_ERROR_DEVICE),
      "the chosen Vulkan device %d is gone; %d devices are left", (int)dev_index, (int)dev_count);
  }
  g_ctx.physical_device = devs[dev_index];
  g_ctx.device_index = dev_index;
  free(devs);

  /* Remember the dispatch limit, so an oversized array is reported rather
   * than silently truncated (see dispatch_pipeline). */
  VkPhysicalDeviceProperties dev_props;
  vkGetPhysicalDeviceProperties(g_ctx.physical_device, &dev_props);
  g_ctx.max_workgroups = dev_props.limits.maxComputeWorkGroupCount[0];
  g_ctx.device_type = dev_props.deviceType;

  /* Queue Family (compute) */
  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.physical_device, &qf_count, NULL);
  VkQueueFamilyProperties *qf_props = malloc(sizeof(VkQueueFamilyProperties) * qf_count);
  if (!qf_props) GPU_RAISE(mrb, GPU_RT_ERROR_OUT_OF_MEMORY, "out of memory enumerating queue families");
  vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.physical_device, &qf_count, qf_props);
  int have_compute = 0;
  uint32_t timestamp_valid_bits = 0;
  g_ctx.queue_family = 0;
  for (uint32_t i = 0; i < qf_count; i++) {
    if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      g_ctx.queue_family = i;
      timestamp_valid_bits = qf_props[i].timestampValidBits;
      have_compute = 1;
      break;
    }
  }
  free(qf_props);
  if (!have_compute) {
    mrb_raisef(mrb, gpu_rt_error_class(mrb, GPU_RT_ERROR_DEVICE),
      "device '%s' has no compute-capable queue family", dev_props.deviceName);
  }

  /* Features (see "Features" above). The structs chained below need a
   * device that speaks Vulkan 1.1; an older one reports none of them. */
  int features[GPU_RT_FEATURE_COUNT] = {0};
  uint32_t subgroup_size = 0;
  int have_8bit = 0, have_budget = 0;
  features[GPU_RT_FEATURE_TIMESTAMP_QUERY] =
    timestamp_valid_bits != 0 && dev_props.limits.timestampPeriod > 0.0f;
  if (dev_props.apiVersion >= VK_API_VERSION_1_1) {
    VkPhysicalDeviceSubgroupProperties subgroup = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES
    };
    VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &subgroup
    };
    vkGetPhysicalDeviceProperties2(g_ctx.physical_device, &props2);
    if ((subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
        (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) &&
        subgroup.subgroupSize > 0) {
      features[GPU_RT_FEATURE_SUBGROUPS] = 1;
      subgroup_size = subgroup.subgroupSize;
    }
#ifdef VK_KHR_8bit_storage
    if (has_device_ext(g_ctx.physical_device, GPU_8BIT_STORAGE_EXT)) {
      VkPhysicalDevice8BitStorageFeaturesKHR f8 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR
      };
      VkPhysicalDeviceFeatures2 feats2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &f8
      };
      vkGetPhysicalDeviceFeatures2(g_ctx.physical_device, &feats2);
      have_8bit = f8.storageBuffer8BitAccess == VK_TRUE;
    }
#endif
    have_budget = has_device_ext(g_ctx.physical_device, GPU_MEMORY_BUDGET_EXT);
  }
  features[GPU_RT_FEATURE_STORAGE_8BIT] = have_8bit;
  features[GPU_RT_FEATURE_MEMORY_BUDGET] = have_budget;

  /* From here a failure leaves a device or its objects behind. */
  g_ctx.init_failed = 1;   /* cleared only on the success path at the end */

  /* Device + Queue */
  float priority = 1.0f;
  VkDeviceQueueCreateInfo q_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = g_ctx.queue_family,
    .queueCount = 1,
    .pQueuePriorities = &priority
  };
  /* If the device advertises VK_KHR_portability_subset it MUST be enabled
   * (Vulkan spec). Present on MoltenVK, absent on the Pi's V3D. */
  const char *dev_exts[3];
  uint32_t dev_ext_count = 0;
  if (has_device_ext(g_ctx.physical_device, GPU_PORTABILITY_SUBSET_EXT)) {
    dev_exts[dev_ext_count++] = GPU_PORTABILITY_SUBSET_EXT;
  }
  /* Turn on what the device has of the optional features, so that a "yes"
   * from gpu_rt_context_has_feature means a gem may use it. */
  const void *dev_next = NULL;
#ifdef VK_KHR_8bit_storage
  VkPhysicalDevice8BitStorageFeaturesKHR enable_8bit = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR,
    .storageBuffer8BitAccess = VK_TRUE
  };
  if (have_8bit) {
    dev_exts[dev_ext_count++] = GPU_8BIT_STORAGE_EXT;
    dev_next = &enable_8bit;
  }
#endif
  if (have_budget) dev_exts[dev_ext_count++] = GPU_MEMORY_BUDGET_EXT;
  VkDeviceCreateInfo dev_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pNext = dev_next,
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &q_info,
    .enabledExtensionCount = dev_ext_count,
    .ppEnabledExtensionNames = dev_ext_count ? dev_exts : NULL
  };
  VK_CHECK(mrb, vkCreateDevice(g_ctx.physical_device, &dev_info, NULL, &g_ctx.device));
  vkGetDeviceQueue(g_ctx.device, g_ctx.queue_family, 0, &g_ctx.queue);
  memcpy(g_ctx.features, features, sizeof(g_ctx.features));
  g_ctx.timestamp_period = dev_props.limits.timestampPeriod;
  g_ctx.timestamp_valid_bits = timestamp_valid_bits;
  g_ctx.subgroup_size = subgroup_size;

  /* Command Pool */
  VkCommandPoolCreateInfo pool_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    .queueFamilyIndex = g_ctx.queue_family
  };
  VK_CHECK(mrb, vkCreateCommandPool(g_ctx.device, &pool_info, NULL, &g_ctx.cmd_pool));

  /* The one command buffer every dispatch is recorded into, and the fence
   * each flush waits on. Both live as long as the context. */
  VkCommandBufferAllocateInfo cbai = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = g_ctx.cmd_pool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1
  };
  VK_CHECK(mrb, vkAllocateCommandBuffers(g_ctx.device, &cbai, &g_ctx.batch_cmd));
  VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VK_CHECK(mrb, vkCreateFence(g_ctx.device, &fi, NULL, &g_ctx.batch_fence));
  g_ctx.batch_recording = 0;
  g_ctx.batch_dispatches = 0;
  g_ctx.batch_sets = 0;
  g_ctx.epoch = 1;   /* a fresh buffer carries epoch 0, which never matches */

  /* Descriptor Set Layouts: 3, 2 and 1 storage buffers respectively */
  int buf_counts[LAYOUT_COUNT] = {3, 2, 1};
  for (int l = 0; l < LAYOUT_COUNT; l++) {
    VkDescriptorSetLayoutBinding bindings[3];
    for (int i = 0; i < buf_counts[l]; i++) {
      bindings[i] = (VkDescriptorSetLayoutBinding){
        .binding = i,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
      };
    }
    VkDescriptorSetLayoutCreateInfo dl_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = buf_counts[l],
      .pBindings = bindings
    };
    VK_CHECK(mrb, vkCreateDescriptorSetLayout(g_ctx.device, &dl_info, NULL,
                                              &g_ctx.desc_layouts[l]));
  }

  /* Pipeline Layouts (one per descriptor layout, shared push constant range) */
  VkPushConstantRange push_range = {
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    .offset = 0,
    .size = GPU_PUSH_CONSTANT_SIZE
  };
  for (int l = 0; l < LAYOUT_COUNT; l++) {
    VkPipelineLayoutCreateInfo pl_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &g_ctx.desc_layouts[l],
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &push_range
    };
    VK_CHECK(mrb, vkCreatePipelineLayout(g_ctx.device, &pl_info, NULL,
                                         &g_ctx.pipe_layouts[l]));
  }

  /* Descriptor Pool */
  VkDescriptorPoolSize pool_size = {
    .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 3 * GPU_MAX_DESC_SETS
  };
  VkDescriptorPoolCreateInfo dp_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
    .maxSets = GPU_MAX_DESC_SETS,
    .poolSizeCount = 1,
    .pPoolSizes = &pool_size
  };
  VK_CHECK(mrb, vkCreateDescriptorPool(g_ctx.device, &dp_info, NULL, &g_ctx.desc_pool));

  g_ctx.initialized = 1;
  g_ctx.init_failed = 0;
}

void gpu_ensure_initialized(mrb_state *mrb) {
  if (!g_ctx.initialized) gpu_init(mrb);
}

/* ---- Shutdown ----
 *
 * Runs from the gem's final. mruby calls the finals of dependent gems first,
 * so by now they have destroyed the pipelines they created. */
void gpu_shutdown(mrb_state *mrb) {
  /* A choice belongs to the mruby that made it, not to the next one. */
  g_ctx.choice_set = 0;
  if (!g_ctx.initialized) {
    /* GPU.devices or GPU.device= made the instance, and nothing started the
     * device. After a failed start the instance may still carry a device,
     * so it is left alone. */
    if (g_ctx.instance_ready && !g_ctx.init_failed) {
      vkDestroyInstance(g_ctx.instance, NULL);
      g_ctx.instance_ready = 0;
    }
    return;
  }
  /* Whatever is still recorded was never submitted and nobody can observe it
   * now, so it is dropped rather than run. Nothing is in flight: every submit
   * is waited on before gpu_flush returns. */
  g_ctx.batch_recording = 0;
  bury_graveyard(mrb);
  gpu_pipelines_shutdown(mrb);
  free(g_ctx.graveyard);
  g_ctx.graveyard = NULL;
  g_ctx.graveyard_cap = 0;
  vkDestroyFence(g_ctx.device, g_ctx.batch_fence, NULL);
  vkDestroyDescriptorPool(g_ctx.device, g_ctx.desc_pool, NULL);
  for (int l = 0; l < LAYOUT_COUNT; l++) {
    vkDestroyPipelineLayout(g_ctx.device, g_ctx.pipe_layouts[l], NULL);
    vkDestroyDescriptorSetLayout(g_ctx.device, g_ctx.desc_layouts[l], NULL);
  }
  vkDestroyCommandPool(g_ctx.device, g_ctx.cmd_pool, NULL);
  vkDestroyDevice(g_ctx.device, NULL);
  vkDestroyInstance(g_ctx.instance, NULL);
  g_ctx.instance_ready = 0;
  g_ctx.initialized = 0;
}
