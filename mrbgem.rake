MRuby::Gem::Specification.new('mruby-gpu-runtime') do |spec|
  spec.license = 'MIT'
  spec.authors = 'Yuji Teshima'
  spec.version = '0.1.0'
  spec.summary = 'Vulkan Compute runtime shared by mruby GPU gems: device, buffers, batched dispatch'

  # Only dependency is the Vulkan loader.
  spec.linker.libraries << 'vulkan'

  # gpu_runtime.h is for the gems built on this one. mruby 4.0 adds a
  # dependency's include/ only to the gems that list it directly, so export it
  # explicitly: mruby-gpu-kernel reaches it through mruby-gpu-narray.
  spec.export_include_paths << File.join(File.dirname(File.expand_path(__FILE__)), 'include')
end
