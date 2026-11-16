# mruby-gpu-runtime -- tests of the GPU module on its own (no arrays).
# Run: mruby test/runtime_test.rb
#
# Buffers and dispatches are exercised through mruby-gpu-narray's tests, which
# build on this gem; this file checks what the runtime offers by itself.

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

# ---- summary ----
puts
puts "#{$pass + $fail} tests, #{$pass} passed, #{$fail} failed"
puts($fail > 0 ? "SOME TESTS FAILED" : "ALL TESTS PASSED")
