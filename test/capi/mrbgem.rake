# The gem behind the C API section of test/runtime_test.rb. It is also the
# smallest gem written on gpu_runtime.h: read src/capi_test.c next to the
# usage example in the header.
MRuby::Gem::Specification.new('mruby-gpu-runtime-capi-test') do |spec|
  spec.license = 'MIT'
  spec.authors = 'Yuji Teshima'
  spec.summary = 'Test bindings for the C API of mruby-gpu-runtime'

  spec.add_dependency 'mruby-gpu-runtime'

  # The header as it will be once the legacy declarations are gone, so this
  # gem cannot lean on them by accident.
  spec.cc.defines << 'GPU_RT_NO_LEGACY'
end
