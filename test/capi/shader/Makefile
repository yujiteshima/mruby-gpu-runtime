# Regenerate the embedded SPIR-V after editing a shader:  make -C test/capi/shader
# The generated headers are committed, so building the test gem needs no
# glslangValidator.
GLSLANG ?= glslangValidator

all: ../src/axpy_spv.h ../src/inc_spv.h

../src/%_spv.h: %.comp
	$(GLSLANG) -V --vn $*_spv $< -o $@

.PHONY: all
