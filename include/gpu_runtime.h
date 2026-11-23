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
 *  - What the device, the driver or a shader does wrong is raised as GPU::Error
 *    or one of its subclasses; see "Errors" below. A mistake in a call (a NULL
 *    description, a dispatch that does not fit its pipeline) is an
 *    ArgumentError or a TypeError.
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
 * calls only return the handle. Raises GPU::DeviceError if there is no usable
 * device (GPU::OutOfMemoryError or GPU::UnsupportedError when the VkResult
 * says so). The handle is valid until the mruby state is closed, and must not
 * be kept across that.
 *
 * This is where the device is chosen, once: what the program said with
 * `GPU.device = index_or_name` before this call, else the environment
 * variable MRUBY_GPU_DEVICE (an index or a name), else the first device that
 * is not a CPU. A gem does not choose; the program does (README.md, "Devices
 * and features"). Raises GPU::DeviceError when MRUBY_GPU_DEVICE matches no
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

/* What this device allows in one layout (see gpu_rt_layout). The numbers
 * differ between devices; Vulkan guarantees at least 4 storage buffers,
 * 12 uniform buffers, 128 bytes of push constants and 16384 bytes in a
 * uniform buffer. */
typedef struct gpu_rt_limits {
  uint32_t max_storage_buffers;      /* in one layout */
  uint32_t max_uniform_buffers;      /* in one layout */
  uint32_t max_push_constant_size;   /* bytes */
  uint32_t max_uniform_buffer_size;  /* bytes of a buffer bound as a uniform buffer */
} gpu_rt_limits;

/* Fill `*out` with the device's limits. */
void gpu_rt_context_limits(mrb_state *mrb, gpu_rt_context *ctx, gpu_rt_limits *out);

/* ---- Buffers ---- */

/* How a buffer may be used; OR them together. */
#define GPU_RT_BUFFER_STORAGE (1u << 0)   /* bound to a pipeline as a storage buffer */
#define GPU_RT_BUFFER_UNIFORM (1u << 1)   /* bound to a pipeline as a uniform buffer */

/* What one element of a buffer is. The runtime records it and does not
 * convert: the bytes are the host's, in host byte order, and the shader that
 * a gem dispatches must read them the same way.
 *
 *   dtype                  bytes per element (gpu_rt_dtype_size)
 *   GPU_RT_DTYPE_FLOAT32   4
 *   GPU_RT_DTYPE_UINT8     1
 *   GPU_RT_DTYPE_UINT16    2
 *   GPU_RT_DTYPE_INT32     4
 *
 * FLOAT32 is 0, so a description that does not name a dtype is float32. */
typedef enum gpu_rt_dtype {
  GPU_RT_DTYPE_FLOAT32 = 0,
  GPU_RT_DTYPE_UINT8,
  GPU_RT_DTYPE_UINT16,
  GPU_RT_DTYPE_INT32,
  GPU_RT_DTYPE_COUNT
} gpu_rt_dtype;

/* The most dimensions a buffer's shape may have. */
#define GPU_RT_MAX_DIMS 4

typedef struct gpu_rt_buffer_desc {
  size_t   size;    /* in bytes; 0 is allowed and makes an empty buffer */
  uint32_t usage;   /* GPU_RT_BUFFER_* bits, at least one */
  gpu_rt_dtype dtype;                /* 0 is GPU_RT_DTYPE_FLOAT32 */
  uint32_t ndim;                     /* dimensions, 0 .. GPU_RT_MAX_DIMS; 0 means one
                                      * dimension whose extent comes from `size` */
  uint32_t shape[GPU_RT_MAX_DIMS];   /* the extent of each of the `ndim` dimensions,
                                      * outermost first (row-major); 0 is allowed */
} gpu_rt_buffer_desc;

/* How `size` and the shape relate. A buffer always holds a whole number of
 * elements: gpu_rt_buffer_size() == gpu_rt_buffer_count() * gpu_rt_dtype_size().
 *
 *  - `ndim` is 0: the buffer has one dimension of `size` / element size
 *    elements. So { .size = n * sizeof(float), .usage = ... } is n float32,
 *    as it was before buffers had a dtype.
 *  - `ndim` is not 0 and `size` is 0: the size is the product of the extents
 *    times the element size. This is the form to prefer.
 *  - both are given: they must agree.
 *
 * gpu_rt_buffer_create() raises ArgumentError when `size` is not a whole
 * number of elements, when `size` and the shape disagree, when the product
 * does not fit in a size_t, for more than GPU_RT_MAX_DIMS dimensions, and for
 * an unknown dtype. An extent of the description past `ndim` is ignored.
 *
 * The memory behind a buffer is a whole number of 32-bit words, whatever the
 * dtype. A shader for a device without 8-bit or 16-bit storage access
 * (GPU_RT_FEATURE_STORAGE_8BIT) can therefore declare a uint8 or uint16
 * buffer as `uint data[]` and unpack the elements itself. The bytes between
 * gpu_rt_buffer_size() and the next multiple of 4 are undefined, can be
 * written through the mapping, and no element lives there. */

/* Create a buffer. Its contents are undefined until written. The host can
 * always map it. Raises ArgumentError for a bad description and
 * GPU::OutOfMemoryError when the device cannot provide the memory.
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

/* The size in bytes: what the buffer was created with, or what its shape
 * and dtype come to. */
size_t gpu_rt_buffer_size(mrb_state *mrb, gpu_rt_buffer *buf);

/* Map the buffer into host memory, for reading and writing gpu_rt_buffer_size()
 * bytes. Recorded dispatches that touch this buffer are run and waited for
 * first, so the bytes are the current ones. Never returns NULL; raises
 * GPU::Error (GPU::OutOfMemoryError as a rule) when the mapping fails. Unmap before the next dispatch that
 * uses the buffer, and do not map a buffer twice. The pointer of an empty
 * buffer must not be dereferenced. */
void *gpu_rt_buffer_map(mrb_state *mrb, gpu_rt_buffer *buf);
void  gpu_rt_buffer_unmap(mrb_state *mrb, gpu_rt_buffer *buf);

/* ---- Dtype and shape of a buffer ----
 *
 * What the buffer was created with, for any buffer: one the gem created, or
 * one passed in from Ruby and read with gpu_rt_buffer_get() or
 * gpu_rt_buffer_check(). They are fixed for the life of the buffer. None of
 * these functions raises, waits for the batch or starts the device. */

/* The element type. */
gpu_rt_dtype gpu_rt_buffer_dtype(mrb_state *mrb, gpu_rt_buffer *buf);

/* The number of dimensions, 1 .. GPU_RT_MAX_DIMS. */
uint32_t gpu_rt_buffer_ndim(mrb_state *mrb, gpu_rt_buffer *buf);

/* Copy the shape into `shape_out`, which has room for GPU_RT_MAX_DIMS
 * extents: the first gpu_rt_buffer_ndim() are the buffer's, outermost first,
 * and the rest are set to 0. */
void gpu_rt_buffer_shape(mrb_state *mrb, gpu_rt_buffer *buf,
                         uint32_t shape_out[GPU_RT_MAX_DIMS]);

/* The number of elements: the product of the extents. */
size_t gpu_rt_buffer_count(mrb_state *mrb, gpu_rt_buffer *buf);

/* The bytes one element of `dtype` takes, or 0 for a value that is not a
 * dtype. */
size_t gpu_rt_dtype_size(mrb_state *mrb, gpu_rt_dtype dtype);

/* The name of `dtype` for messages and for Ruby ("float32", "uint8",
 * "uint16", "int32"), or "unknown" for a value that is not a dtype. The
 * string belongs to the runtime. */
const char *gpu_rt_dtype_name(mrb_state *mrb, gpu_rt_dtype dtype);

/* ---- Pipelines ---- */

/* A layout within these two is accepted on every device. They were the most
 * a layout could ask for before layouts became general; the most on this
 * device is what gpu_rt_context_limits() reports. */
#define GPU_RT_MAX_PUSH_CONSTANT_SIZE 16
#define GPU_RT_MAX_STORAGE_BUFFERS 3

/* What a shader binds, described by value. All bindings are in set 0:
 *
 *  - the storage buffers at `layout(set = 0, binding = i)` for i from 0, in
 *    the order they are passed to gpu_rt_dispatch();
 *  - then the uniform buffers, at the bindings after the last storage buffer
 *    (binding = storage_buffers + j), in the order of the dispatch's
 *    uniform_buffers;
 *  - and at most one push-constant block, of at most push_constant_size bytes.
 *
 * The runtime does not read the SPIR-V, so the description must be true: a
 * shader that binds more, or declares a larger push-constant block, than its
 * layout says is an error the device may not report.
 *
 * A description beyond gpu_rt_context_limits() raises ArgumentError. */
typedef struct gpu_rt_layout {
  uint32_t storage_buffers;      /* 1 .. max_storage_buffers */
  uint32_t push_constant_size;   /* bytes, a multiple of 4, 0 .. max_push_constant_size */
  uint32_t uniform_buffers;      /* 0 .. max_uniform_buffers; usually 0 or 1 */
} gpu_rt_layout;

/* ---- Errors ----
 *
 * The exception classes of every GPU gem. The runtime defines them, so they
 * exist before any other GPU gem is initialised:
 *
 *   StandardError
 *     GPU::Error                 anything the GPU side does wrong
 *       GPU::DeviceError         no usable device, initialisation failed, device lost
 *       GPU::OutOfMemoryError    host or device memory, or another resource, ran out
 *       GPU::UnsupportedError    the device lacks an extension, a feature or a format
 *       GPU::CompileError        a shader could not be made into a pipeline
 *     ArgumentError
 *       GPU::ShapeError          arrays whose sizes or shapes do not fit the operation
 *
 * GPU::ShapeError is an ArgumentError and not a GPU::Error: it is the
 * caller's mistake, and `rescue ArgumentError` keeps catching it.
 *
 * The runtime chooses the class of a failed Vulkan call from its VkResult
 * (the table is in README.md, "Errors"), and its message names the operation,
 * the Vulkan call and the VkResult. */
typedef enum gpu_rt_error_kind {
  GPU_RT_ERROR_GENERIC = 0,     /* GPU::Error */
  GPU_RT_ERROR_DEVICE,          /* GPU::DeviceError */
  GPU_RT_ERROR_OUT_OF_MEMORY,   /* GPU::OutOfMemoryError */
  GPU_RT_ERROR_UNSUPPORTED,     /* GPU::UnsupportedError */
  GPU_RT_ERROR_COMPILE,         /* GPU::CompileError */
  GPU_RT_ERROR_SHAPE,           /* GPU::ShapeError */
  GPU_RT_ERROR_KIND_COUNT
} gpu_rt_error_kind;

/* The exception class of a kind, for a gem that raises one itself:
 *
 *   mrb_raisef(mrb, gpu_rt_error_class(mrb, GPU_RT_ERROR_SHAPE),
 *              "shape mismatch: %d vs %d", (int)n, (int)m);
 *
 * An unknown kind gives GPU::Error. Does not start the device. */
struct RClass *gpu_rt_error_class(mrb_state *mrb, gpu_rt_error_kind kind);

/* Why gpu_rt_pipeline_create() failed. */
typedef struct gpu_rt_error {
  const char *operation;   /* the step that failed, e.g. "creating the compute pipeline" */
  const char *reason;      /* in words, e.g. "out of device memory" */
  int32_t     vk_result;   /* the VkResult, or 0 when the failure is not Vulkan's */
  gpu_rt_error_kind kind;  /* what the runtime would have raised: pass it to
                            * gpu_rt_error_class(). GPU_RT_ERROR_COMPILE unless
                            * the VkResult says memory, the device or a missing
                            * feature. */
} gpu_rt_error;

/* Create a compute pipeline from `spirv_size` bytes of SPIR-V with the entry
 * point "main". The bytes are copied and need not stay around; they need no
 * particular alignment. `label` names the pipeline in error messages and, by
 * default, its dispatches in the profiler; it is copied, and may be NULL.
 *
 * SPIR-V the device does not accept is an expected failure, and gems want to
 * say different things about it, so the caller chooses:
 *
 *  - `error_out` is NULL: raises GPU::CompileError, or the GPU::Error subclass
 *    the VkResult stands for, naming the label, the failed step and the
 *    VkResult.
 *  - `error_out` is not NULL: returns NULL and fills `*error_out`. On success
 *    `*error_out` is zeroed.
 *
 * A layout this device cannot provide raises ArgumentError either way.
 *
 * Pipelines are cached: a call with the same SPIR-V bytes and the same
 * layout as a pipeline that is still alive does not compile again. It
 * returns a new handle to the same device pipeline, with its own label.
 * Every handle is released once, and the device pipeline goes with the last
 * one, so creating the same pipeline in several places is cheap and safe.
 *
 * The caller owns the handle until gpu_rt_pipeline_release(). */
gpu_rt_pipeline *gpu_rt_pipeline_create(mrb_state *mrb, gpu_rt_context *ctx,
                                        const void *spirv, size_t spirv_size,
                                        const gpu_rt_layout *layout, const char *label,
                                        gpu_rt_error *error_out);

/* Give a pipeline back. Does not raise and does not wait, so it is safe in a
 * GC free function and in a gem's final function: if recorded dispatches
 * still use the pipeline, it is destroyed after they have run. NULL is
 * ignored. */
void gpu_rt_pipeline_release(mrb_state *mrb, gpu_rt_pipeline *pipeline);

/* How many pipelines the device holds for the handles that are alive. Handles
 * that share one cached pipeline count once. For tests and diagnostics. */
uint32_t gpu_rt_pipeline_count(mrb_state *mrb, gpu_rt_context *ctx);

/* ---- Dispatch ---- */

typedef struct gpu_rt_dispatch_desc {
  gpu_rt_pipeline *pipeline;
  gpu_rt_buffer  **buffers;            /* the storage buffers, in binding order */
  uint32_t         buffer_count;       /* must be the layout's storage_buffers */
  const void      *push_constants;     /* copied during the call; NULL when the size is 0 */
  uint32_t         push_constant_size; /* a multiple of 4, at most the layout's push_constant_size */
  uint32_t         groups[3];          /* workgroups in x, y, z. A 0 in y or z means 1;
                                        * a 0 in x records a dispatch that does nothing. */
  const char      *label;              /* the operation's name, e.g. "add"; copied.
                                        * NULL means the pipeline's label. */
  gpu_rt_buffer  **uniform_buffers;    /* in binding order, created with GPU_RT_BUFFER_UNIFORM;
                                        * NULL when the layout has none */
  uint32_t         uniform_buffer_count; /* must be the layout's uniform_buffers */
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
 * A uniform buffer is read when the batch runs, like any other buffer: what
 * it holds then is what the shader sees, and mapping it waits for the
 * dispatches that use it.
 *
 * Raises ArgumentError when the description does not fit the pipeline, or
 * when groups[0] is more than the device allows in one dispatch, and
 * GPU::Error or a subclass when the device fails; the message then starts
 * with the label. */
void gpu_rt_dispatch(mrb_state *mrb, const gpu_rt_dispatch_desc *desc);

/* ---- Sync ---- */

/* Run everything recorded and wait for it. Returns at once when nothing is
 * recorded. Raises GPU::DeviceError if the device is lost, and GPU::Error or
 * another subclass for any other failure of the device. */
void gpu_rt_sync(mrb_state *mrb, gpu_rt_context *ctx);

/* The same, but only if a recorded dispatch touches `buf`. */
void gpu_rt_buffer_sync(mrb_state *mrb, gpu_rt_buffer *buf);

/* Dispatches recorded and not yet run (GPU.pending). */
uint32_t gpu_rt_pending(mrb_state *mrb, gpu_rt_context *ctx);

#ifdef __cplusplus
}
#endif

#endif
