# Settings for x86 on Linux
ifeq ($(TARGET), linux)
  ifeq ($(TARGET_ARCH), x86_64)
    PLATFORM_FLAGS = \
      -DTF_LITE_DISABLE_X86_NEON
    CXXFLAGS += $(PLATFORM_FLAGS)
    CCFLAGS += $(PLATFORM_FLAGS)
  endif
  ifeq ($(TARGET_ARCH), i386)
      PLATFORM_FLAGS = \
      -m32 -DTF_LITE_DISABLE_X86_NEON
    CXXFLAGS += $(PLATFORM_FLAGS)
    CCFLAGS += $(PLATFORM_FLAGS)
  endif
endif
