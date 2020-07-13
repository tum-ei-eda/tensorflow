# Settings for x86 gcc on Windows (MinGW etc)
ifeq ($(TARGET), windows)
  ifeq ($(TARGET_ARCH), x86_64)
    PLATFORM_FLAGS = \
      -DTF_LITE_DISABLE_X86_NEON
    CXXFLAGS += $(PLATFORM_FLAGS)
    CCFLAGS += $(PLATFORM_FLAGS)
    TARGET_TOOLCHAIN_ROOT:=/mingw64/bin/
  endif
  ifeq ($(TARGET_ARCH), i386)
      PLATFORM_FLAGS = \
      -m32 -DTF_LITE_DISABLE_X86_NEON
    CXXFLAGS += $(PLATFORM_FLAGS)
    CCFLAGS += $(PLATFORM_FLAGS)
    # Aaaargh MINGW64 is not by default compiled multilib AND the compiler front-ends
    # need PATH set to function (FFS).   Setting target TARGET_TOOLCHAIN_ROOT 
    # to make it obvious which compiler is being used despite hacking PATH
    PATH:=/mingw32/bin:$(PATH)
    TARGET_TOOLCHAIN_ROOT:=/mingw32/bin/
  endif
endif
