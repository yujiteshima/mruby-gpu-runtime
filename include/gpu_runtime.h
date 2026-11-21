/* gpu_runtime.h -- the C API of mruby-gpu-runtime.
 *
 * This is the only header a GPU gem includes. It gives the gem the device,
 * buffers, compute pipelines and the deferred batch that every GPU gem in one
 * mruby shares, behind opaque handles. No Vulkan type appears here, and a gem
 * built on it does not call Vulkan itself.
 *
 * ---- Usage ----
 *
 * A gem that adds `GPU::NArray#double`, with its shader compiled to SPIR-V
 * and embedded as `double_spv` (see README.md, "Writing a gem on the C API"):
 *
 *   #include <gpu_runtime.h>
 *
 *   static gpu_rt_pipeline *double_pipe;
 *
 *   static mrb_value narray_double(mrb_state *mrb, mrb_value self) {
 *     gpu_rt_context *ctx = gpu_rt_context_get(mrb);   // the device, started on first use
 *     gpu_rt_buffer  *in  = gpu_rt_buffer_get(mrb, self);
 *     uint32_t        n   = (uint32_t)(gpu_rt_buffer_size(mrb, in) / sizeof(float));
 *
 *     if (!double_pipe) {
 *       gpu_rt_layout layout = { .storage_buffers = 2, .push_constant_size = sizeof(n) };
 *       double_pipe = gpu_rt_pipeline_create(mrb, ctx, double_spv, sizeof(double_spv),
 *                                            &layout, "double", NULL);
 *     }
 *
 *     gpu_rt_buffer_desc desc = { .size = n * sizeof(float), .usage = GPU_RT_BUFFER_STORAGE };
 *     gpu_rt_buffer *out = gpu_rt_buffer_create(mrb, ctx, &desc);
 *     // Wrap at once: from here the GC frees `out`, even if the dispatch raises.
 *     mrb_value result = gpu_rt_buffer_wrap(mrb, mrb_obj_class(mrb, self), out);
 *
 *     gpu_rt_buffer *bufs[2] = { in, out };            // bindings 0 and 1
 *     gpu_rt_dispatch_desc d = {
 *       .pipeline = double_pipe,
 *       .buffers = bufs, .buffer_count = 2,
 *       .push_constants = &n, .push_constant_size = sizeof(n),
 *       .groups = { (n + 255) / 256 },
 *       .label = "double",
 *     };
 *     gpu_rt_dispatch(mrb, &d);                        // recorded, not run yet
 *     return result;
 *   }
 *
 *   void mrb_mruby_gpu_double_gem_init(mrb_state *mrb) {
 *     gpu_rt_require(mrb, 0, 1);                       // stop here if the runtime is another API
 *     ...
 *   }
 *
 * Nothing runs on the GPU until a result is needed: gpu_rt_buffer_map() waits
 * for the dispatches that touch that buffer, and gpu_rt_sync() for all of them.
 *
 * ---- Conventions ----
 *
 *  - Every name starts with gpu_rt_ (GPU_RT_ for macros and enumerators).
 *  - `mrb` is the first parameter of every function. Then comes the object the
 *    function works on, then its inputs, and outputs come last.
 *  - A function raises a Ruby exception when it fails, so it does not return
 *    in that case, and returns no error code. Whatever the caller allocated
 *    and the GC does not own must be released before the call.
 *    gpu_rt_pipeline_create() can report instead of raising; see there.
 *  - A descriptor struct (gpu_rt_buffer_desc, gpu_rt_layout,
 *    gpu_rt_dispatch_desc) is passed by pointer and read during the call
 *    only. Fill it with designated initializers: a member this header gains
 *    later is then zero, and zero always means today's behaviour.
 *  - A string the runtime returns belongs to the runtime. A string or array
 *    the caller passes in is copied if the runtime keeps it.
 *  - Everything runs on the thread that owns `mrb`.
 */
#ifndef GPU_RUNTIME_H
#define GPU_RUNTIME_H

#include <mruby.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- API version ----
 *
 * The version of this header, not of the gem. The major number changes when
 * existing code stops compiling or changes meaning, the minor number when
 * something is added. While the major number is 0 the API is still settling,
 * and any minor number may break. */
#define GPU_RT_API_VERSION_MAJOR 0
#define GPU_RT_API_VERSION_MINOR 1
#define GPU_RT_MAKE_VERSION(major, minor) (((major) << 16) | (minor))
#define GPU_RT_API_VERSION \
  GPU_RT_MAKE_VERSION(GPU_RT_API_VERSION_MAJOR, GPU_RT_API_VERSION_MINOR)

/* Check that the runtime built into this mruby provides the API a gem was
 * written for. Call it first in the gem's init function, with the numbers the
 * gem was written against (literals, not the macros above):
 *
 *   gpu_rt_require(mrb, 0, 1);
 *
 * It returns when `major` is the runtime's major number and the runtime's
 * minor number is at least `minor` (exactly `minor` while the major number is
 * 0). Otherwise it raises a RuntimeError that names both versions and the
 * calling source file, and prints the same line on stderr, because mruby 3.4
 * drops an exception raised while gems are initialised.
 *
 * It does not start the device. */
#define gpu_rt_require(mrb, major, minor) \
  gpu_rt_require_at((mrb), (major), (minor), __FILE__)
void gpu_rt_require_at(mrb_state *mrb, int major, int minor, const char *file);

/* ---- Handles ----
 *
 * All three are opaque: a gem holds the pointer and never looks inside. */
typedef struct gpu_rt_context  gpu_rt_context;   /* the device, its queue and the batch */
typedef struct gpu_rt_buffer   gpu_rt_buffer;    /* a block of GPU memory */
typedef struct gpu_rt_pipeline gpu_rt_pipeline;  /* a compute shader ready to dispatch */

/* ---- Context ---- */

/* The context of this mruby. The first call starts Vulkan (instance, device,
 * queue), so a program that never touches the GPU never pays for it; later
 * calls only return the handle. Raises if there is no usable device. The
 * handle is valid until the mruby state is closed, and must not be kept
 * across that.
 *
 * This is where the device is chosen, once: what the program said with
 * `GPU.device = index_or_name` before this call, else the environment
 * variable MRUBY_GPU_DEVICE (an index or a name), else the first device that
 * is not a CPU. A gem does not choose; the program does (README.md, "Devices
 * and features"). Raises RuntimeError when MRUBY_GPU_DEVICE matches no
 * device. */
gpu_rt_context *gpu_rt_context_get(mrb_state *mrb);

/* Whether the context has been started. Never starts it and never raises. */
mrb_bool gpu_rt_context_ready(mrb_state *mrb);

/* What kind of device it is. GPU_RT_DEVICE_CPU is a software rasterizer such
 * as llvmpipe or lavapipe. */
typedef enum gpu_rt_device_type {
  GPU_RT_DEVICE_OTHER = 0,
  GPU_RT_DEVICE_INTEGRATED,
  GPU_RT_DEVICE_DISCRETE,
  GPU_RT_DEVICE_VIRTUAL,
  GPU_RT_DEVICE_CPU
} gpu_rt_device_type;

typedef struct gpu_rt_device_info {
  const char *name;            /* the device's own name, e.g. "V3D 7.1.7.0" */
  const char *backend;         /* "Vulkan" */
  uint32_t    api_major;       /* the API version the device reports */
  uint32_t    api_minor;
  uint32_t    api_patch;
  uint32_t    max_workgroups;  /* the largest groups[0] one dispatch may have */
  uint32_t    index;           /* which of GPU.devices this is */
  gpu_rt_device_type type;
} gpu_rt_device_info;

/* Fill `*out` with what GPU.info reports. The strings belong to the runtime
 * and stay valid as long as the context does. */
void gpu_rt_context_info(mrb_state *mrb, gpu_rt_context *ctx, gpu_rt_device_info *out);

/* What a device may or may not be able to do. A gem asks before it relies on
 * one, and takes another path or raises when the answer is no. */
typedef enum gpu_rt_feature {
  GPU_RT_FEATURE_STORAGE_8BIT = 0,  /* 8-bit integers in storage buffers
                                     * (storageBuffer8BitAccess): a shader may declare
                                     * uint8_t / int8_t members in a storage block */
  GPU_RT_FEATURE_TIMESTAMP_QUERY,   /* per-dispatch GPU time: the compute queue
                                     * writes timestamps */
  GPU_RT_FEATURE_SUBGROUPS,         /* subgroup operations in compute shaders, at
                                     * least the basic ones */
  GPU_RT_FEATURE_MEMORY_BUDGET,     /* the device reports how much memory is left
                                     * (VK_EXT_memory_budget) */
  GPU_RT_FEATURE_COUNT
} gpu_rt_feature;

/* True when the device has the feature *and* the runtime has turned it on,
 * so a shader or a gem may use it. The runtime turns on every one of them
 * that the device has, when the device starts. GPU.info[:features] gives the
 * same answers to Ruby, under the keys :storage_8bit, :timestamp_query,
 * :subgroups and :memory_budget. An unknown value is false. */
mrb_bool gpu_rt_context_has_feature(mrb_state *mrb, gpu_rt_context *ctx, gpu_rt_feature feature);

/* ---- Buffers ---- */

/* How a buffer may be used; OR them together. */
#define GPU_RT_BUFFER_STORAGE (1u << 0)   /* bound to a pipeline as a storage buffer */

typedef struct gpu_rt_buffer_desc {
  size_t   size;    /* in bytes; 0 is allowed and makes an empty buffer */
  uint32_t usage;   /* GPU_RT_BUFFER_* bits, at least one */
  /* A dtype and a shape join this struct with runtime#4. Left at zero they
   * will mean what every buffer is today: float32, one dimension. */
} gpu_rt_buffer_desc;

/* Create a buffer. Its contents are undefined until written. The host can
 * always map it. Raises ArgumentError for a bad description and RuntimeError
 * when the device cannot provide the memory.
 *
 * The caller owns the returned buffer until it passes it to
 * gpu_rt_buffer_wrap() or gpu_rt_buffer_release(). Do one of them before
 * anything else that can raise. */
gpu_rt_buffer *gpu_rt_buffer_create(mrb_state *mrb, gpu_rt_context *ctx,
                                    const gpu_rt_buffer_desc *desc);

/* Make `buf` the data of a new Ruby object of class `klass` and return the
 * object. The GC owns the buffer from here and frees it with the object;
 * the caller must not release it. `klass` must be GPU::NArray or one of its
 * subclasses, or a class set up the same way (MRB_SET_INSTANCE_TT(klass,
 * MRB_TT_CDATA)). */
mrb_value gpu_rt_buffer_wrap(mrb_state *mrb, struct RClass *klass, gpu_rt_buffer *buf);

/* Free a buffer that was never wrapped. Does not raise and does not wait: if
 * recorded dispatches still use the buffer, it is freed after they have run. */
void gpu_rt_buffer_release(mrb_state *mrb, gpu_rt_buffer *buf);

/* The buffer behind a Ruby object made by gpu_rt_buffer_wrap(), such as a
 * GPU::NArray passed in from Ruby. The handle is valid as long as the object
 * is alive, so keep the object reachable (an argument or `self` is).
 *
 * gpu_rt_buffer_get() raises TypeError for any other object;
 * gpu_rt_buffer_check() returns NULL instead, for a caller with a message of
 * its own. */
gpu_rt_buffer *gpu_rt_buffer_get(mrb_state *mrb, mrb_value obj);
gpu_rt_buffer *gpu_rt_buffer_check(mrb_state *mrb, mrb_value obj);

/* The size in bytes the buffer was created with. */
size_t gpu_rt_buffer_size(mrb_state *mrb, gpu_rt_buffer *buf);

/* Map the buffer into host memory, for reading and writing gpu_rt_buffer_size()
 * bytes. Recorded dispatches that touch this buffer are run and waited for
 * first, so the bytes are the current ones. Never returns NULL; raises
 * RuntimeError when the mapping fails. Unmap before the next dispatch that
 * uses the buffer, and do not map a buffer twice. The pointer of an empty
 * buffer must not be dereferenced. */
void *gpu_rt_buffer_map(mrb_state *mrb, gpu_rt_buffer *buf);
void  gpu_rt_buffer_unmap(mrb_state *mrb, gpu_rt_buffer *buf);

/* ---- Pipelines ---- */

/* The most bytes of push constants a layout may ask for. */
#define GPU_RT_MAX_PUSH_CONSTANT_SIZE 16
/* The most storage buffers a layout may ask for. */
#define GPU_RT_MAX_STORAGE_BUFFERS 3

/* What a shader binds, described by value. The shader declares its storage
 * buffers as `layout(set = 0, binding = i)` for i from 0, in the order they
 * are passed to gpu_rt_dispatch(), and at most one push-constant block. */
typedef struct gpu_rt_layout {
  uint32_t storage_buffers;      /* 1 .. GPU_RT_MAX_STORAGE_BUFFERS */
  uint32_t push_constant_size;   /* bytes, 0 .. GPU_RT_MAX_PUSH_CONSTANT_SIZE */
  /* More kinds of binding (a uniform buffer, more storage buffers) join this
   * struct with runtime#3. */
} gpu_rt_layout;

/* Why gpu_rt_pipeline_create() failed. */
typedef struct gpu_rt_error {
  const char *operation;   /* the step that failed, e.g. "creating the compute pipeline" */
  const char *reason;      /* in words, e.g. "out of device memory" */
  int32_t     vk_result;   /* the VkResult, or 0 when the failure is not Vulkan's */
} gpu_rt_error;

/* Create a compute pipeline from `spirv_size` bytes of SPIR-V with the entry
 * point "main". The bytes are copied and need not stay around; they need no
 * particular alignment. `label` names the pipeline in error messages and, by
 * default, its dispatches in the profiler; it is copied, and may be NULL.
 *
 * SPIR-V the device does not accept is an expected failure, and gems want to
 * say different things about it, so the caller chooses:
 *
 *  - `error_out` is NULL: raises RuntimeError naming the label, the failed
 *    step and the VkResult.
 *  - `error_out` is not NULL: returns NULL and fills `*error_out`. On success
 *    `*error_out` is zeroed.
 *
 * A layout this runtime cannot provide raises ArgumentError either way.
 *
 * The caller owns the pipeline until gpu_rt_pipeline_release(). */
gpu_rt_pipeline *gpu_rt_pipeline_create(mrb_state *mrb, gpu_rt_context *ctx,
                                        const void *spirv, size_t spirv_size,
                                        const gpu_rt_layout *layout, const char *label,
                                        gpu_rt_error *error_out);

/* Give a pipeline back. Does not raise and does not wait, so it is safe in a
 * GC free function and in a gem's final function: if recorded dispatches
 * still use the pipeline, it is destroyed after they have run. NULL is
 * ignored. */
void gpu_rt_pipeline_release(mrb_state *mrb, gpu_rt_pipeline *pipeline);

/* ---- Dispatch ---- */

typedef struct gpu_rt_dispatch_desc {
  gpu_rt_pipeline *pipeline;
  gpu_rt_buffer  **buffers;            /* in binding order */
  uint32_t         buffer_count;       /* must be the layout's storage_buffers */
  const void      *push_constants;     /* copied during the call; NULL when the size is 0 */
  uint32_t         push_constant_size; /* at most the layout's push_constant_size */
  uint32_t         groups[3];          /* workgroups in x, y, z. A 0 in y or z means 1;
                                        * a 0 in x records a dispatch that does nothing. */
  const char      *label;              /* the operation's name, e.g. "add"; copied.
                                        * NULL means the pipeline's label. */
} gpu_rt_dispatch_desc;

/* Record one dispatch into the batch. It runs, together with everything else
 * recorded, when a result is needed: gpu_rt_buffer_map(), gpu_rt_sync(),
 * gpu_rt_buffer_sync(), or GPU.sync from Ruby. Dispatches run in the order
 * they were recorded, and each one sees what the ones before it wrote. With
 * GPU.sync_mode = :eager it runs before this function returns.
 *
 * The label is for error messages and the profiler (runtime#8). The profiler
 * also wants the Ruby line that asked for the dispatch; the runtime reads it
 * from `mrb` at this call, so call this from the C function that implements
 * the Ruby method.
 *
 * Raises ArgumentError when the description does not fit the pipeline, or
 * when groups[0] is more than the device allows in one dispatch. */
void gpu_rt_dispatch(mrb_state *mrb, const gpu_rt_dispatch_desc *desc);

/* ---- Sync ---- */

/* Run everything recorded and wait for it. Returns at once when nothing is
 * recorded. Raises RuntimeError if the device fails. */
void gpu_rt_sync(mrb_state *mrb, gpu_rt_context *ctx);

/* The same, but only if a recorded dispatch touches `buf`. */
void gpu_rt_buffer_sync(mrb_state *mrb, gpu_rt_buffer *buf);

/* Dispatches recorded and not yet run (GPU.pending). */
uint32_t gpu_rt_pending(mrb_state *mrb, gpu_rt_context *ctx);

#ifdef __cplusplus
}
#endif

#endif
