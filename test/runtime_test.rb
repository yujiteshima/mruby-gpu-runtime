# mruby-gpu-runtime -- tests of the GPU module on its own (no arrays).
# Run: mruby test/runtime_test.rb
#
# The first half checks the GPU module, which needs this gem only. The second
# half checks the C API (include/gpu_runtime.h): buffers, pipelines, dispatch
# and sync. C functions cannot be called from Ruby, so it runs through the
# small gem in test/capi, and is skipped, with a line saying so, in a build
# that leaves that gem out. List both to run everything:
#
#   conf.gem '/path/to/mruby-gpu-runtime'
#   conf.gem '/path/to/mruby-gpu-runtime/test/capi'

$pass = 0
$fail = 0

def ok(label)
  $pass += 1
  puts "PASS #{label}"
end

def ng(label, detail)
  $fail += 1
  puts "FAIL #{label}: #{detail}"
end

def assert_true(label, cond, detail = "false")
  cond ? ok(label) : ng(label, detail)
end

def assert_raise(label, klass)
  begin
    yield
  rescue => e
    if e.is_a?(klass)
      return ok(label)
    else
      return ng(label, "expected #{klass}, got #{e.class}: #{e.message}")
    end
  end
  ng(label, "expected #{klass}, but nothing was raised")
end

info = GPU.info
puts "device: #{info[:device]} (#{info[:backend]} #{info[:api_version]})"
puts

# ---- device information ----
assert_true("GPU.info is a Hash", info.is_a?(Hash), info.class.to_s)
assert_true("GPU.info[:device] is a non-empty String",
            info[:device].is_a?(String) && !info[:device].empty?, info[:device].inspect)
assert_true("GPU.info[:backend] is Vulkan", info[:backend] == "Vulkan", info[:backend].inspect)
assert_true("GPU.info[:api_version] looks like 1.x.y",
            info[:api_version].is_a?(String) && info[:api_version].start_with?("1."),
            info[:api_version].inspect)
assert_true("GPU.info[:max_workgroups] is positive",
            info[:max_workgroups].is_a?(Integer) && info[:max_workgroups] > 0,
            info[:max_workgroups].inspect)
assert_true("GPU.device_name matches GPU.info[:device]",
            GPU.device_name == info[:device], GPU.device_name.inspect)

# ---- the batch, with nothing recorded ----
assert_true("nothing is pending at start", GPU.pending == 0, "pending = #{GPU.pending}")
assert_true("GPU.sync with nothing pending returns nil", GPU.sync.nil?)
assert_true("GPU.sync leaves nothing pending", GPU.pending == 0, "pending = #{GPU.pending}")

# ---- sync mode ----
assert_true("default sync mode is :deferred", GPU.sync_mode == :deferred, GPU.sync_mode.inspect)
GPU.sync_mode = :eager
assert_true("sync_mode reports :eager", GPU.sync_mode == :eager, GPU.sync_mode.inspect)
GPU.sync_mode = :deferred
assert_true("sync_mode reports :deferred again", GPU.sync_mode == :deferred, GPU.sync_mode.inspect)
assert_raise("unknown sync mode -> ArgumentError", ArgumentError) { GPU.sync_mode = :later }
assert_true("a rejected mode leaves the mode unchanged", GPU.sync_mode == :deferred,
            GPU.sync_mode.inspect)

# ---- the C API, through test/capi ----
if Object.const_defined?(:GPURuntimeTest)
  T = GPURuntimeTest
  B = GPURuntimeTest::Buffer
  puts
  puts "C API #{T.api_version.join('.')}"

  def assert_raise_message(label, klass, *parts)
    begin
      yield
    rescue => e
      return ng(label, "expected #{klass}, got #{e.class}: #{e.message}") unless e.is_a?(klass)
      missing = parts.reject { |p| e.message.include?(p) }
      return ok(label) if missing.empty?
      return ng(label, "message lacks #{missing.inspect}: #{e.message.inspect}")
    end
    ng(label, "expected #{klass}, but nothing was raised")
  end

  # -- version --
  major, minor = T.api_version
  assert_true("gpu_rt_require accepts the version it was built with", T.require_api(major, minor))
  assert_raise_message("gpu_rt_require refuses a newer minor", RuntimeError,
                       "#{major}.#{minor + 1}", "#{major}.#{minor}", "capi_test.c",
                       "Update mruby-gpu-runtime") { T.require_api(major, minor + 1) }
  assert_raise_message("gpu_rt_require refuses another major", RuntimeError,
                       "#{major + 1}.0", "Update mruby-gpu-runtime") { T.require_api(major + 1, 0) }
  if major == 0 && minor > 0
    assert_raise_message("gpu_rt_require refuses an older minor while the major is 0",
                         RuntimeError, "Update the gem") { T.require_api(0, minor - 1) }
  end

  # -- context --
  assert_true("the context is ready once GPU.info has run", T.ready?)
  di = T.device_info
  assert_true("gpu_rt_context_info agrees with GPU.info",
              di[:name] == info[:device] && di[:backend] == info[:backend] &&
              di[:api].join(".") == info[:api_version] &&
              di[:max_workgroups] == info[:max_workgroups], di.inspect)
  assert_true("no feature is reported yet, and an unknown one is false",
              T.features.all? { |f| f == false }, T.features.inspect)
  assert_raise_message("a context handle that is not the runtime's -> ArgumentError",
                       ArgumentError, "gpu_rt_sync", "gpu_rt_context_get") { T.sync_bad_context }

  # -- buffers --
  a = B.floats(5)
  assert_true("a buffer reports the size it was created with", a.bytesize == 20, a.bytesize.to_s)
  a.write([1, 2, 3, 4, 5])
  assert_true("what is written through a mapping is read back", a.read == [1.0, 2.0, 3.0, 4.0, 5.0],
              a.read.inspect)
  assert_true("a buffer of 3 bytes keeps its size", B.with_usage(3, 1).bytesize == 3)
  e = B.floats(0)
  assert_true("an empty buffer has size 0 and reads as []", e.bytesize == 0 && e.read == [],
              "#{e.bytesize} #{e.read.inspect}")
  assert_raise_message("usage 0 -> ArgumentError", ArgumentError, "usage") { B.with_usage(16, 0) }
  assert_raise_message("unknown usage bits -> ArgumentError", ArgumentError, "usage") { B.with_usage(16, 6) }
  assert_true("a buffer can be released without being wrapped", B.create_and_release(64))
  assert_true("gpu_rt_buffer_check finds a wrapped buffer", T.buffer?(a))
  assert_true("gpu_rt_buffer_check returns NULL for other objects",
              !T.buffer?("x") && !T.buffer?(nil) && !T.buffer?(1) && !T.buffer?([a]))
  assert_true("gpu_rt_buffer_get reads a buffer passed in from Ruby", T.bytesize_of(a) == 20)
  assert_raise_message("gpu_rt_buffer_get of a String -> TypeError", TypeError, "String") { T.bytesize_of("x") }

  # -- dispatch into the batch --
  GPU.sync
  b = B.floats(5).write([10, 20, 30, 40, 50])
  c = T.axpy(a, b, 2.0)
  assert_true("a dispatch is recorded, not run", T.pending == 1 && GPU.pending == 1,
              "pending = #{T.pending}")
  assert_true("mapping the result runs the batch", c.read == [12.0, 24.0, 36.0, 48.0, 60.0],
              c.read.inspect)
  assert_true("nothing is pending after the read", T.pending == 0, "pending = #{T.pending}")

  d = T.axpy(c, a, 0.5, "half-plus")      # 7 14 21 28 35
  T.inc(d)
  T.inc(d)
  assert_true("three dispatches wait in one batch", T.pending == 3, "pending = #{T.pending}")
  assert_true("mapping a buffer the batch does not touch leaves it pending",
              b.read == [10.0, 20.0, 30.0, 40.0, 50.0] && T.pending == 3, "pending = #{T.pending}")
  b.sync
  assert_true("gpu_rt_buffer_sync of an untouched buffer leaves it pending", T.pending == 3)
  assert_true("later dispatches see what earlier ones wrote", d.read == [9.0, 16.0, 23.0, 30.0, 37.0],
              d.read.inspect)

  T.inc(d)
  d.sync
  assert_true("gpu_rt_buffer_sync runs the batch that touches the buffer", T.pending == 0)
  T.inc(d)
  T.sync
  assert_true("gpu_rt_sync runs everything", T.pending == 0 && d.read == [11.0, 18.0, 25.0, 32.0, 39.0],
              d.read.inspect)

  big = B.floats(1000).write((0...1000).to_a)
  r = T.inc(T.axpy(big, big, 3.0)).read    # 4 workgroups, the last one partly past n
  assert_true("a dispatch over several workgroups", r[0] == 1.0 && r[255] == 1021.0 &&
              r[256] == 1025.0 && r[999] == 3997.0, [r[0], r[255], r[256], r[999]].inspect)

  z = T.inc(B.floats(0))
  assert_true("a dispatch of 0 workgroups is recorded and does nothing",
              T.pending == 1 && z.read == [])

  GPU.sync_mode = :eager
  f = T.inc(B.floats(2).write([1, 2]))
  assert_true(":eager runs a dispatch before gpu_rt_dispatch returns", T.pending == 0)
  GPU.sync_mode = :deferred
  assert_true(":eager gives the same result", f.read == [2.0, 3.0], f.read.inspect)

  # -- pipelines --
  g = B.floats(3).write([1, 2, 3])
  T.inc_and_release(g)
  assert_true("a pipeline released while its dispatch waits still runs",
              T.pending == 1 && g.read == [2.0, 3.0, 4.0], g.read.inspect)
  T.inc_and_release(g)
  GC.start
  assert_true("and again after a garbage collection", g.read == [3.0, 4.0, 5.0], g.read.inspect)

  assert_true("a layout by value: 1 buffer, 4 bytes of push constants", T.create_pipeline(1, 4, 0, false))
  assert_true("the same, reporting instead of raising, zeroes the error", T.create_pipeline(1, 4, 0, true))
  assert_raise_message("a layout of 0 buffers -> ArgumentError", ArgumentError, "probe", "storage buffers") do
    T.create_pipeline(0, 4, 0, false)
  end
  # What a layout may ask for is the device's limit, so the tests read it.
  lim = T.limits
  assert_true("gpu_rt_context_limits reports at least what Vulkan guarantees",
              lim[:max_storage_buffers] >= 4 && lim[:max_uniform_buffers] >= 12 &&
              lim[:max_push_constant_size] >= 128 && lim[:max_uniform_buffer_size] >= 16384,
              lim.inspect)
  puts "limits: #{lim.inspect}"
  if lim[:max_storage_buffers] < 0xffffffff
    assert_raise_message("more storage buffers than the device has -> ArgumentError, also when reporting",
                         ArgumentError, "storage buffers", lim[:max_storage_buffers].to_s) do
      T.create_pipeline(lim[:max_storage_buffers] + 1, 4, 0, true)
    end
  end
  if lim[:max_push_constant_size] < 0xfffffffc
    assert_raise_message("more push constants than the device has -> ArgumentError", ArgumentError,
                         "push constants", lim[:max_push_constant_size].to_s) do
      T.create_pipeline(1, lim[:max_push_constant_size] + 4, 0, false)
    end
  end
  if lim[:max_uniform_buffers] < 0xffffffff
    assert_raise_message("more uniform buffers than the device has -> ArgumentError", ArgumentError,
                         "uniform buffers", lim[:max_uniform_buffers].to_s) do
      T.create_pipeline(1, 4, 0, false, lim[:max_uniform_buffers] + 1)
    end
  end
  assert_raise_message("push constants that are not whole words -> ArgumentError", ArgumentError,
                       "push constants", "multiple of 4") { T.create_pipeline(1, 6, 0, false) }
  assert_true("layouts the three fixed ones did not have: 4 buffers, 20 bytes, a uniform buffer",
              T.create_pipeline(4, 4, 0, false) && T.create_pipeline(1, 20, 0, false) &&
              T.create_pipeline(1, 4, 0, true, 1))
  assert_true("a layout with no push constants", T.ones(B.floats(300)).read == [1.0] * 300)
  assert_raise_message("SPIR-V that is not whole words -> RuntimeError naming the pipeline",
                       RuntimeError, "probe", "SPIR-V", "VkResult") { T.create_pipeline(1, 4, 13, false) }
  rep = T.create_pipeline(1, 4, 13, true)
  assert_true("the same, reported through error_out", rep.is_a?(Array) && rep[0].include?("SPIR-V") &&
              rep[1].is_a?(String) && rep[2] == 0, rep.inspect)

  # -- dispatches that do not fit the pipeline --
  GPU.sync
  assert_raise_message("too few buffers -> ArgumentError naming the label", ArgumentError,
                       "my-op", "2 buffers", "binds 3") { T.axpy(a, b, 1.0, "my-op", 2) }
  assert_raise_message("a missing buffer -> ArgumentError naming the binding", ArgumentError,
                       "axpy", "binding 1") { T.axpy(a, nil, 1.0) }
  assert_raise_message("more push constants than the layout -> ArgumentError", ArgumentError,
                       "push constants") { T.axpy(a, b, 1.0, nil, 3, 12) }
  assert_raise_message("more workgroups than the device allows -> ArgumentError", ArgumentError,
                       "workgroups") { T.inc_groups(a, info[:max_workgroups] + 1) } if info[:max_workgroups] < 0xffffffff
  assert_true("a refused dispatch records nothing", T.pending == 0, "pending = #{T.pending}")
  assert_true("and leaves its inputs alone", a.read == [1.0, 2.0, 3.0, 4.0, 5.0], a.read.inspect)

  # -- a layout with four storage buffers, a uniform buffer and 32 bytes of push constants --
  GPU.sync
  x = B.floats(5).write([1, 2, 3, 4, 5])
  y = B.floats(5).write([10, 20, 30, 40, 50])
  zz = B.floats(5).write([100, 200, 300, 400, 500])
  w = B.with_usage(16, 2).write([1, 0.5, 0.25, 2])        # GPU_RT_BUFFER_UNIFORM
  # (x + y/2 + zz/4 + 2) * 2 + offsets[i % 4]
  o = T.blend(x, y, zz, w, 2.0, [0, 1000, 2000, 3000])
  assert_true("a dispatch with a uniform buffer is recorded like any other", T.pending == 1,
              "pending = #{T.pending}")
  assert_true("4 storage buffers, a uniform buffer and 32 bytes of push constants",
              o.read == [66.0, 1128.0, 2190.0, 3252.0, 314.0], o.read.inspect)
  o1 = T.blend(x, y, zz, w, 1.0, [0, 0, 0, 0])            # 33 64 95 126 157
  assert_true("the next dispatch is pending", T.pending == 1, "pending = #{T.pending}")
  w.write([0, 0, 1, 0])                                   # mapping a bound uniform buffer runs the batch
  assert_true("writing a uniform buffer waits for the dispatch that reads it", T.pending == 0,
              "pending = #{T.pending}")
  o2 = T.blend(x, y, zz, w, 1.0, [0, 0, 0, 0])            # zz
  assert_true("each dispatch saw the uniform buffer as it was when recorded",
              o1.read == [33.0, 64.0, 95.0, 126.0, 157.0] &&
              o2.read == [100.0, 200.0, 300.0, 400.0, 500.0], "#{o1.read.inspect} #{o2.read.inspect}")
  both = B.with_usage(16, 3).write([1, 1, 1, 0])          # storage and uniform
  assert_true("a buffer with both usages binds as either",
              T.blend(both, both, both, both, 1.0, [0, 0, 0, 0]).read == [3.0, 3.0, 3.0, 0.0])
  GPU.sync
  assert_raise_message("no uniform buffer for a layout that has one -> ArgumentError", ArgumentError,
                       "blend", "0 uniform buffers", "binds 1") { T.blend(x, y, zz, w, 1.0, [0, 0, 0, 0], 0) }
  assert_raise_message("a missing uniform buffer -> ArgumentError naming the binding", ArgumentError,
                       "binding 4") { T.blend(x, y, zz, nil, 1.0, [0, 0, 0, 0]) }
  assert_raise_message("a storage buffer as the uniform buffer -> ArgumentError", ArgumentError,
                       "binding 4", "GPU_RT_BUFFER_UNIFORM") { T.blend(x, y, zz, x, 1.0, [0, 0, 0, 0]) }
  assert_raise_message("a uniform buffer as a storage buffer -> ArgumentError", ArgumentError,
                       "binding 2", "GPU_RT_BUFFER_STORAGE") { T.blend(x, y, w, w, 1.0, [0, 0, 0, 0]) }
  assert_true("a refused dispatch with a uniform buffer records nothing", T.pending == 0)
  w.write([1, 0, 0, 1])                                   # acc + 1
  acc4 = B.floats(4).write([0, 0, 0, 0])
  600.times { acc4 = T.blend(acc4, acc4, acc4, w, 1.0, [0, 0, 0, 0]) }
  assert_true("600 chained dispatches of the wider layout, more than the descriptor pool holds",
              acc4.read == [600.0, 600.0, 600.0, 600.0], acc4.read.inspect)
  assert_true("the three-buffer layout still works next to it",
              T.axpy(x, y, 2.0).read == [12.0, 24.0, 36.0, 48.0, 60.0])

  # -- the pipeline cache --
  P = GPURuntimeTest::Pipeline
  GPU.sync
  T.inc(B.floats(1).write([0])).read       # the test gem's own inc pipeline exists: 1 buffer, 4 bytes
  count = T.pipeline_count
  p1 = P.inc(1, 4, "first")
  p2 = P.inc(1, 4, "second")
  assert_true("the same SPIR-V with the same layout is not compiled again",
              T.pipeline_count == count, "#{T.pipeline_count} pipelines, #{count} before")
  q1 = P.inc(1, 8)
  assert_true("the same SPIR-V with another push-constant size is another pipeline",
              T.pipeline_count == count + 1, "#{T.pipeline_count} pipelines, #{count} before")
  q2 = P.inc(2, 4)
  assert_true("and so it is with another number of buffers",
              T.pipeline_count == count + 2, "#{T.pipeline_count} pipelines, #{count} before")
  q3 = P.inc(1, 8)
  assert_true("each of them is found again", T.pipeline_count == count + 2,
              "#{T.pipeline_count} pipelines, #{count} before")
  assert_raise_message("each handle keeps its own label", ArgumentError, "second") do
    p2.inc(x, 8)
  end
  h = B.floats(3).write([1, 2, 3])
  p1.inc(h)
  p1.release
  p2.inc(h)
  assert_true("releasing one handle leaves the others of the same pipeline working",
              h.read == [3.0, 4.0, 5.0] && T.pipeline_count == count + 2, h.read.inspect)
  q1.release
  assert_true("a cached pipeline stays while a handle is left", T.pipeline_count == count + 2,
              "#{T.pipeline_count} pipelines")
  q3.inc(h)
  assert_true("and still runs", h.read == [4.0, 5.0, 6.0], h.read.inspect)
  q3.release
  assert_true("it goes with its last handle", T.pipeline_count == count + 1,
              "#{T.pipeline_count} pipelines")

  # Released while a recorded dispatch uses it, then asked for again before the batch has run.
  r1 = P.inc(1, 8)
  r1.inc(h)
  r1.release
  assert_true("a pipeline released with its dispatch pending no longer counts",
              T.pending == 1 && T.pipeline_count == count + 1, "#{T.pipeline_count} pipelines")
  r2 = P.inc(1, 8)
  assert_true("asked for again before the batch has run, it is taken back from the cache",
              T.pending == 1 && T.pipeline_count == count + 2, "#{T.pipeline_count} pipelines")
  assert_true("the pending dispatch runs", h.read == [5.0, 6.0, 7.0], h.read.inspect)
  r2.inc(h)
  assert_true("and the pipeline outlives that batch", h.read == [6.0, 7.0, 8.0] &&
              T.pipeline_count == count + 2, h.read.inspect)
  r2.inc(h)
  r2.release
  assert_true("released with a dispatch pending and not asked for again, it runs and is destroyed",
              h.read == [7.0, 8.0, 9.0] && T.pipeline_count == count + 1, h.read.inspect)
  q2.release
  p2.release
  assert_true("with every handle released, the count is back where it was",
              T.pipeline_count == count, "#{T.pipeline_count} pipelines, #{count} before")
  assert_true("the test gem's own handle to the shared pipeline still works",
              T.inc(h).read == [8.0, 9.0, 10.0], h.read.inspect)
  P.inc(1, 12)                             # never released: the GC frees it after the device is gone

  # -- many buffers come and go with a batch pending --
  acc = B.floats(4).write([0, 0, 0, 0])
  one = B.floats(4).write([1, 1, 1, 1])
  600.times { acc = T.axpy(acc, one, 1.0) }   # more dispatches than the descriptor pool holds
  assert_true("600 chained dispatches, their intermediates garbage-collected",
              acc.read == [600.0, 600.0, 600.0, 600.0], acc.read.inspect)
else
  puts
  puts "SKIP the C API tests: this mruby was built without test/capi (see the top of this file)"
end

# ---- summary ----
puts
puts "#{$pass + $fail} tests, #{$pass} passed, #{$fail} failed"
puts($fail > 0 ? "SOME TESTS FAILED" : "ALL TESTS PASSED")
