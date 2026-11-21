/* capi_test.c -- Ruby bindings over the gpu_rt_ functions, for
 * test/runtime_test.rb. Everything here uses gpu_runtime.h and nothing else
 * of the runtime, the way a GPU gem does.
 */
#include <gpu_runtime.h>
#include <mruby/array.h>
#include <mruby/class.h>
#include <mruby/hash.h>
#include <mruby/string.h>
#include <string.h>

#include "axpy_spv.h"   /* c = a * s + b: 3 buffers, push { uint n; float s; } */
#include "inc_spv.h"    /* a += 1:        1 buffer,  push { uint n; }          */

static gpu_rt_pipeline *axpy_pipe;
static gpu_rt_pipeline *inc_pipe;

static struct RClass *buffer_class(mrb_state *mrb) {
  return mrb_class_get_under(mrb, mrb_module_get(mrb, "GPURuntimeTest"), "Buffer");
}

static uint32_t float_count(mrb_state *mrb, gpu_rt_buffer *buf) {
  return (uint32_t)(gpu_rt_buffer_size(mrb, buf) / sizeof(float));
}

/* A new wrapped buffer of n floats. */
static mrb_value new_floats(mrb_state *mrb, gpu_rt_context *ctx, uint32_t n, gpu_rt_buffer **out) {
  gpu_rt_buffer_desc desc = { .size = n * sizeof(float), .usage = GPU_RT_BUFFER_STORAGE };
  *out = gpu_rt_buffer_create(mrb, ctx, &desc);
  return gpu_rt_buffer_wrap(mrb, buffer_class(mrb), *out);
}

static gpu_rt_pipeline *make_inc(mrb_state *mrb, gpu_rt_context *ctx) {
  gpu_rt_layout layout = { .storage_buffers = 1, .push_constant_size = sizeof(uint32_t) };
  return gpu_rt_pipeline_create(mrb, ctx, inc_spv, sizeof(inc_spv), &layout, "inc", NULL);
}

static void record_inc(mrb_state *mrb, gpu_rt_pipeline *pipe, gpu_rt_buffer *a) {
  uint32_t n = float_count(mrb, a);
  gpu_rt_dispatch_desc d = {
    .pipeline = pipe,
    .buffers = &a, .buffer_count = 1,
    .push_constants = &n, .push_constant_size = sizeof(n),
    .groups = { (n + 255) / 256 },
  };
  gpu_rt_dispatch(mrb, &d);
}

/* ---- version and context ---- */

static mrb_value t_api_version(mrb_state *mrb, mrb_value self) {
  mrb_value v = mrb_ary_new(mrb);
  mrb_ary_push(mrb, v, mrb_fixnum_value(GPU_RT_API_VERSION_MAJOR));
  mrb_ary_push(mrb, v, mrb_fixnum_value(GPU_RT_API_VERSION_MINOR));
  return v;
}

static mrb_value t_require_api(mrb_state *mrb, mrb_value self) {
  mrb_int major, minor;
  mrb_get_args(mrb, "ii", &major, &minor);
  gpu_rt_require(mrb, (int)major, (int)minor);
  return mrb_true_value();
}

static mrb_value t_ready(mrb_state *mrb, mrb_value self) {
  return mrb_bool_value(gpu_rt_context_ready(mrb));
}

static mrb_value t_device_info(mrb_state *mrb, mrb_value self) {
  gpu_rt_context *ctx = gpu_rt_context_get(mrb);
  gpu_rt_device_info info;
  gpu_rt_context_info(mrb, ctx, &info);
  mrb_value h = mrb_hash_new(mrb);
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_lit(mrb, "name")), mrb_str_new_cstr(mrb, info.name));
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_lit(mrb, "backend")), mrb_str_new_cstr(mrb, info.backend));
  mrb_value api = mrb_ary_new(mrb);
  mrb_ary_push(mrb, api, mrb_fixnum_value((mrb_int)info.api_major));
  mrb_ary_push(mrb, api, mrb_fixnum_value((mrb_int)info.api_minor));
  mrb_ary_push(mrb, api, mrb_fixnum_value((mrb_int)info.api_patch));
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_lit(mrb, "api")), api);
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_lit(mrb, "max_workgroups")),
               mrb_fixnum_value((mrb_int)info.max_workgroups));
  return h;
}

static mrb_value t_features(mrb_state *mrb, mrb_value self) {
  gpu_rt_context *ctx = gpu_rt_context_get(mrb);
  mrb_value v = mrb_ary_new(mrb);
  for (int f = 0; f <= GPU_RT_FEATURE_COUNT; f++) {   /* one past the end on purpose */
    mrb_ary_push(mrb, v, mrb_bool_value(gpu_rt_context_has_feature(mrb, ctx, (gpu_rt_feature)f)));
  }
  return v;
}

static mrb_value t_sync(mrb_state *mrb, mrb_value self) {
  gpu_rt_sync(mrb, gpu_rt_context_get(mrb));
  return mrb_nil_value();
}

static mrb_value t_pending(mrb_state *mrb, mrb_value self) {
  return mrb_fixnum_value((mrb_int)gpu_rt_pending(mrb, gpu_rt_context_get(mrb)));
}

/* A context handle that is not the runtime's. */
static mrb_value t_sync_bad_context(mrb_state *mrb, mrb_value self) {
  gpu_rt_sync(mrb, NULL);
  return mrb_nil_value();
}

/* ---- buffers ---- */

/* Buffer.floats(n) -> Buffer of n floats, contents undefined */
static mrb_value buf_s_floats(mrb_state *mrb, mrb_value self) {
  mrb_int n;
  mrb_get_args(mrb, "i", &n);
  gpu_rt_buffer *buf;
  return new_floats(mrb, gpu_rt_context_get(mrb), (uint32_t)n, &buf);
}

/* Buffer.with_usage(bytes, usage) -> Buffer; for the usage checks */
static mrb_value buf_s_with_usage(mrb_state *mrb, mrb_value self) {
  mrb_int size, usage;
  mrb_get_args(mrb, "ii", &size, &usage);
  gpu_rt_buffer_desc desc = { .size = (size_t)size, .usage = (uint32_t)usage };
  gpu_rt_buffer *buf = gpu_rt_buffer_create(mrb, gpu_rt_context_get(mrb), &desc);
  return gpu_rt_buffer_wrap(mrb, mrb_class_ptr(self), buf);
}

/* Buffer.create_and_release(bytes) -> true; a buffer that is never wrapped */
static mrb_value buf_s_create_and_release(mrb_state *mrb, mrb_value self) {
  mrb_int size;
  mrb_get_args(mrb, "i", &size);
  gpu_rt_buffer_desc desc = { .size = (size_t)size, .usage = GPU_RT_BUFFER_STORAGE };
  gpu_rt_buffer *buf = gpu_rt_buffer_create(mrb, gpu_rt_context_get(mrb), &desc);
  gpu_rt_buffer_release(mrb, buf);
  return mrb_true_value();
}

static mrb_value buf_bytesize(mrb_state *mrb, mrb_value self) {
  return mrb_fixnum_value((mrb_int)gpu_rt_buffer_size(mrb, gpu_rt_buffer_get(mrb, self)));
}

/* buf.write([floats]) -> buf */
static mrb_value buf_write(mrb_state *mrb, mrb_value self) {
  mrb_value ary;
  mrb_get_args(mrb, "A", &ary);
  gpu_rt_buffer *buf = gpu_rt_buffer_get(mrb, self);
  uint32_t n = float_count(mrb, buf);
  if ((uint32_t)RARRAY_LEN(ary) != n) mrb_raise(mrb, E_ARGUMENT_ERROR, "wrong number of values");
  /* Converted before mapping: mrb_as_float can raise, and a raise between
   * map and unmap would leave the buffer mapped. */
  float *tmp = (float *)mrb_malloc(mrb, sizeof(float) * (n ? n : 1));
  for (uint32_t i = 0; i < n; i++) {
    mrb_value v = mrb_ary_ref(mrb, ary, i);
    tmp[i] = mrb_float_p(v) ? (float)mrb_float(v) : (float)mrb_integer(v);
  }
  float *m = (float *)gpu_rt_buffer_map(mrb, buf);
  if (n) memcpy(m, tmp, sizeof(float) * n);
  gpu_rt_buffer_unmap(mrb, buf);
  mrb_free(mrb, tmp);
  return self;
}

/* buf.read -> [floats]; waits for the dispatches that touch buf */
static mrb_value buf_read(mrb_state *mrb, mrb_value self) {
  gpu_rt_buffer *buf = gpu_rt_buffer_get(mrb, self);
  uint32_t n = float_count(mrb, buf);
  float *tmp = (float *)mrb_malloc(mrb, sizeof(float) * (n ? n : 1));
  float *m = (float *)gpu_rt_buffer_map(mrb, buf);
  if (n) memcpy(tmp, m, sizeof(float) * n);
  gpu_rt_buffer_unmap(mrb, buf);
  mrb_value ary = mrb_ary_new_capa(mrb, n);
  int ai = mrb_gc_arena_save(mrb);
  for (uint32_t i = 0; i < n; i++) {
    mrb_ary_push(mrb, ary, mrb_float_value(mrb, tmp[i]));
    mrb_gc_arena_restore(mrb, ai);
  }
  mrb_free(mrb, tmp);
  return ary;
}

static mrb_value buf_sync(mrb_state *mrb, mrb_value self) {
  gpu_rt_buffer_sync(mrb, gpu_rt_buffer_get(mrb, self));
  return self;
}

/* GPURuntimeTest.buffer?(obj) -> gpu_rt_buffer_check found one */
static mrb_value t_is_buffer(mrb_state *mrb, mrb_value self) {
  mrb_value obj;
  mrb_get_args(mrb, "o", &obj);
  return mrb_bool_value(gpu_rt_buffer_check(mrb, obj) != NULL);
}

/* GPURuntimeTest.bytesize_of(obj) -> size through gpu_rt_buffer_get */
static mrb_value t_bytesize_of(mrb_state *mrb, mrb_value self) {
  mrb_value obj;
  mrb_get_args(mrb, "o", &obj);
  return mrb_fixnum_value((mrb_int)gpu_rt_buffer_size(mrb, gpu_rt_buffer_get(mrb, obj)));
}

/* ---- pipelines and dispatch ---- */

/* GPURuntimeTest.axpy(a, b, s, label = nil, count = 3, push = 8) -> c = a * s + b
 * `count` and `push` are what the dispatch claims, for the mismatch checks. */
static mrb_value t_axpy(mrb_state *mrb, mrb_value self) {
  mrb_value a_obj, b_obj;
  mrb_float s;
  const char *label = NULL;
  mrb_int count = 3, push_size = 8;
  mrb_get_args(mrb, "oof|z!ii", &a_obj, &b_obj, &s, &label, &count, &push_size);
  gpu_rt_context *ctx = gpu_rt_context_get(mrb);
  gpu_rt_buffer *a = gpu_rt_buffer_get(mrb, a_obj);
  gpu_rt_buffer *b = mrb_nil_p(b_obj) ? NULL : gpu_rt_buffer_get(mrb, b_obj);

  if (!axpy_pipe) {
    gpu_rt_layout layout = { .storage_buffers = 3, .push_constant_size = 8 };
    axpy_pipe = gpu_rt_pipeline_create(mrb, ctx, axpy_spv, sizeof(axpy_spv), &layout, "axpy", NULL);
  }

  uint32_t n = float_count(mrb, a);
  gpu_rt_buffer *c;
  mrb_value result = new_floats(mrb, ctx, n, &c);
  gpu_rt_buffer *bufs[3] = { a, b, c };
  struct { uint32_t n; float s; } push = { n, (float)s };
  gpu_rt_dispatch_desc d = {
    .pipeline = axpy_pipe,
    .buffers = bufs, .buffer_count = (uint32_t)count,
    .push_constants = &push, .push_constant_size = (uint32_t)push_size,
    .groups = { (n + 255) / 256 },
    .label = label,
  };
  gpu_rt_dispatch(mrb, &d);
  return result;
}

/* GPURuntimeTest.inc(a) -> a, incremented in place */
static mrb_value t_inc(mrb_state *mrb, mrb_value self) {
  mrb_value a_obj;
  mrb_get_args(mrb, "o", &a_obj);
  gpu_rt_context *ctx = gpu_rt_context_get(mrb);
  if (!inc_pipe) inc_pipe = make_inc(mrb, ctx);
  record_inc(mrb, inc_pipe, gpu_rt_buffer_get(mrb, a_obj));
  return a_obj;
}

/* GPURuntimeTest.inc_and_release(a) -> a. A pipeline of its own, released
 * while its dispatch is still only recorded. */
static mrb_value t_inc_and_release(mrb_state *mrb, mrb_value self) {
  mrb_value a_obj;
  mrb_get_args(mrb, "o", &a_obj);
  gpu_rt_context *ctx = gpu_rt_context_get(mrb);
  gpu_rt_buffer *a = gpu_rt_buffer_get(mrb, a_obj);
  gpu_rt_pipeline *pipe = make_inc(mrb, ctx);
  record_inc(mrb, pipe, a);
  gpu_rt_pipeline_release(mrb, pipe);
  return a_obj;
}

/* GPURuntimeTest.inc_groups(a, x) -> a. Asks for x workgroups, for the limit. */
static mrb_value t_inc_groups(mrb_state *mrb, mrb_value self) {
  mrb_value a_obj;
  mrb_int x;
  mrb_get_args(mrb, "oi", &a_obj, &x);
  gpu_rt_context *ctx = gpu_rt_context_get(mrb);
  if (!inc_pipe) inc_pipe = make_inc(mrb, ctx);
  gpu_rt_buffer *a = gpu_rt_buffer_get(mrb, a_obj);
  uint32_t n = float_count(mrb, a);
  gpu_rt_dispatch_desc d = {
    .pipeline = inc_pipe,
    .buffers = &a, .buffer_count = 1,
    .push_constants = &n, .push_constant_size = sizeof(n),
    .groups = { (uint32_t)x },
  };
  gpu_rt_dispatch(mrb, &d);
  return a_obj;
}

/* GPURuntimeTest.create_pipeline(storage_buffers, push_size, spirv_bytes, report)
 *   -> true, or [operation, reason, vk_result] when `report` and it failed.
 * `spirv_bytes` cuts the inc shader short; 0 means all of it. */
static mrb_value t_create_pipeline(mrb_state *mrb, mrb_value self) {
  mrb_int bufs, push, bytes;
  mrb_bool report;
  mrb_get_args(mrb, "iiib", &bufs, &push, &bytes, &report);
  gpu_rt_context *ctx = gpu_rt_context_get(mrb);
  gpu_rt_layout layout = { .storage_buffers = (uint32_t)bufs, .push_constant_size = (uint32_t)push };
  gpu_rt_error err = { "untouched", "untouched", 12345 };
  gpu_rt_pipeline *p = gpu_rt_pipeline_create(mrb, ctx, inc_spv,
                                              bytes ? (size_t)bytes : sizeof(inc_spv),
                                              &layout, "probe", report ? &err : NULL);
  if (p) {
    gpu_rt_pipeline_release(mrb, p);
    if (report && (err.operation || err.reason || err.vk_result)) {
      mrb_raise(mrb, E_RUNTIME_ERROR, "error_out was not zeroed on success");
    }
    return mrb_true_value();
  }
  mrb_value v = mrb_ary_new(mrb);
  mrb_ary_push(mrb, v, mrb_str_new_cstr(mrb, err.operation));
  mrb_ary_push(mrb, v, mrb_str_new_cstr(mrb, err.reason));
  mrb_ary_push(mrb, v, mrb_fixnum_value(err.vk_result));
  return v;
}

void mrb_mruby_gpu_runtime_capi_test_gem_init(mrb_state *mrb) {
  gpu_rt_require(mrb, 0, 1);

  struct RClass *t = mrb_define_module(mrb, "GPURuntimeTest");
  mrb_define_module_function(mrb, t, "api_version", t_api_version, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, t, "require_api", t_require_api, MRB_ARGS_REQ(2));
  mrb_define_module_function(mrb, t, "ready?", t_ready, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, t, "device_info", t_device_info, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, t, "features", t_features, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, t, "sync", t_sync, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, t, "pending", t_pending, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, t, "sync_bad_context", t_sync_bad_context, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, t, "buffer?", t_is_buffer, MRB_ARGS_REQ(1));
  mrb_define_module_function(mrb, t, "bytesize_of", t_bytesize_of, MRB_ARGS_REQ(1));
  mrb_define_module_function(mrb, t, "axpy", t_axpy, MRB_ARGS_ARG(3, 3));
  mrb_define_module_function(mrb, t, "inc", t_inc, MRB_ARGS_REQ(1));
  mrb_define_module_function(mrb, t, "inc_and_release", t_inc_and_release, MRB_ARGS_REQ(1));
  mrb_define_module_function(mrb, t, "inc_groups", t_inc_groups, MRB_ARGS_REQ(2));
  mrb_define_module_function(mrb, t, "create_pipeline", t_create_pipeline, MRB_ARGS_REQ(4));

  struct RClass *buf = mrb_define_class_under(mrb, t, "Buffer", mrb->object_class);
  MRB_SET_INSTANCE_TT(buf, MRB_TT_CDATA);
  mrb_undef_class_method(mrb, buf, "new");
  mrb_define_class_method(mrb, buf, "floats", buf_s_floats, MRB_ARGS_REQ(1));
  mrb_define_class_method(mrb, buf, "with_usage", buf_s_with_usage, MRB_ARGS_REQ(2));
  mrb_define_class_method(mrb, buf, "create_and_release", buf_s_create_and_release, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, buf, "bytesize", buf_bytesize, MRB_ARGS_NONE());
  mrb_define_method(mrb, buf, "write", buf_write, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, buf, "read", buf_read, MRB_ARGS_NONE());
  mrb_define_method(mrb, buf, "sync", buf_sync, MRB_ARGS_NONE());
}

/* Runs before the runtime's final, which tears the device down. */
void mrb_mruby_gpu_runtime_capi_test_gem_final(mrb_state *mrb) {
  gpu_rt_pipeline_release(mrb, axpy_pipe);
  gpu_rt_pipeline_release(mrb, inc_pipe);
  axpy_pipe = inc_pipe = NULL;
}
