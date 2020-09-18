# Settings for RISCV 32-bit MCU toolchain target Variants



#
# Common settings (derived from pulpine via TUM / ETISSS examples
#

ifneq ($(subst ifx_riscv32,,$(TARGET)),$(TARGET))


  LD := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${LD_TOOL}
  OBJDUMP := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${OBJDUMP_TOOL}
  OBJCOPY := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${OBJCOPY_TOOL}
  READELF := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${READELF_TOOL}


  ifeq ($(TARGET), ifx_riscv32_fpga)
    IFX_RV_TARGET_ARCH ?= rv32im
  else
    IFX_RV_TARGET_ARCH ?= rv32imac
  endif
  IFX_RV_TARGET_ABI ?= ilp32
  TARGET_ARCH := $(IFX_RV_TARGET_ARCH)
  TARGET_ABI := $(IFX_RV_TARGET_ABI)

  # Fetch common settings if not already set...
ifndef IFX_TOOLSPREFIX
  $(error IFX_TOOLSPREFIX not set - called without aiml_deployment/GET_PROJECT_SETTINGS.sh settings? Currentmy unsupported!) 
endif
  
  TARGET_TOOLCHAIN_ROOT:=$(IFX_TOOLSPREFIX)/$(IFX_RV_TOOLCHAIN_ID)-$(IFX_RV_TOOLCHAIN_VERSION)/bin/
  SWERV_ISS_HOME := $(IFX_TOOLSPREFIX)/swerviss-$(IFX_SWERVISS_VERSION)

CC_WARNINGS := -Wsign-compare -Wdouble-promotion \
               -Wshadow -Wunused-variable -Wmissing-field-initializers \
               -Wno-error=unused-function
               
  TARGET_TOOLCHAIN_PREFIX := $(IFX_RV_TOOLCHAIN_TRIPLET)-
ifneq ($(BUILD_TYPE), release)
  BUILD_TYPE_FLAGS := \
    -DTF_LITE_MCU_DEBUG_LOG
  ifeq ($(BUILD_TYPE), debug)
     PLATFORM_OPT_FLAGS := -O1
  endif
endif

  PLATFORM_FLAGS = \
    -march=$(TARGET_ARCH) \
    -mabi=$(TARGET_ABI) \
    -DIFX_RISCV \
    -DIFX_RISCV_$(patsubst ifx_riscv%,,$(TARGET)) \
    -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS \
    -DTF_LITE_USE_GLOBAL_MAX \
    -DTF_LITE_USE_GLOBAL_MIN \
    $(BUILD_TYPE_FLAGS) \
    -fno-exceptions \
    -fno-unwind-tables \
    -ffunction-sections \
    -fdata-sections \
    -funsigned-char \
    -Wvla \
    -Wall \
    -Wextra \
    -Wno-unused-parameter \
    -Wno-missing-field-initializers \
    -Wno-write-strings \
    -Wno-sign-compare \
    -fno-delete-null-pointer-checks \
    -fomit-frame-pointer \
    $(PLATFORM_OPT_FLAGS)


  BUILD_TYPE := micro

  LDFLAGS +=  -Wl,--gc-sections
endif

ifeq ($(TARGET), ifx_riscv32_install_only)
  IFX_RISCV32_RECOGNIZED:=1
endif

#
# ----------------------------------------------------------------------
# Settings for RISCV 32-bit MCU toolchain semi-hosted ELF (SWERV simulator).
# -----------------------------------------------------------------------
#

ifeq ($(TARGET), ifx_riscv32_mcu)
  
  IFX_RISCV32_RECOGNIZED=1

  TEST_SCRIPT := $(MAKEFILE_DIR)/gen/test_swerv_iss_binary.sh
  THIRD_PARTY_TARGETS += $(TEST_SCRIPT)
  
  CXXFLAGS += $(PLATFORM_FLAGS) 
  CCFLAGS += -std=gnu99 $(PLATFORM_FLAGS)
  
  
endif

#
# ----------------------------------------------------------------------
# Settings for RISCV 32-bit MCU on FPGA.
# -----------------------------------------------------------------------
#



ifeq ($(TARGET), ifx_riscv32_fpga)

  IFX_RISCV32_RECOGNIZED=1

  ADDITIONAL_PLATFORM_FLAGS = \
    -mcmodel=medany \
    -mexplicit-relocs \
    -fno-builtin-printf \
    -Wsign-compare \
    -Wdouble-promotion \
    -Wshadow \
    -Wno-unused-variable \
    -Wunused-function \
    -fno-threadsafe-statics \
    -fno-use-cxa-atexit

  LDFLAGS +=  -nostartfiles 
  LDFLAGS +=	-T $(INSTALLED_TFLITE_MICRO)/tools/make/targets/ifx_riscv/ifx_rv32_link.ld \
	   $(INSTALLED_TFLITE_MICRO)/tools/make/targets/ifx_riscv/ifx_rv32_startup.c 


  THIRD_PARTY_TARGETS += $(TEST_SCRIPT)
  
  CXXFLAGS += $(PLATFORM_FLAGS) $(ADDITIONAL_PLATFORM_FLAGS)

  CCFLAGS += $(PLATFORM_FLAGS)  $(ADDITIONAL_PLATFORM_FLAGS)
  
$(BINDIR)%.bin: $(BINDIR)%.elf
	@mkdir -p $(dir $@)
	$(OBJDUMP)  -S --line-numbers --full-contents -D -EL  "$<" > "$(BINDIR)$*.disasm"
	$(OBJCOPY) "$<" "$@" -O binary
	$(READELF) -a -I "$<" > "$(BINDIR)$
  
endif


#
# ----------------------------------------------------------------------
# Settings for RISCV 32-bit MCU toolchain semi-hosted ELF (SWERV simulator).
# -----------------------------------------------------------------------
#

ifeq ($(TARGET), ifx_riscv32_ovp)
  
  IFX_RISCV32_RECOGNIZED=1

  TEST_SCRIPT := $(TESTING_DIR)/test_ovp_iss_binary.sh
  
  
  CXXFLAGS += $(PLATFORM_FLAGS) 

  CCFLAGS += -std=gnu99 $(PLATFORM_FLAGS)
  
  
endif

#
# ----------------------------------------------------------------------
# Settings for RISCV 32-bit MCU toolchain idealized pulpino-like bare-metal
# -----------------------------------------------------------------------
#

ifeq ($(TARGET), ifx_riscv32_pulpino)

  IFX_RISCV32_RECOGNIZED=1
  PULPINO_CRT ?= $(ETISS_HOME)/crt_cmake
  LDFLAGS += -nostartfiles -T $(PULPINO_CRT)/pulpino_tumeda/link.ld -Wl,--no-relax

  ROM_SIZE_KB ?= 4096
  RAM_SIZE_KB ?= 1024
  STACK_SIZE_KB ?= 128

  ROM_SIZE := $(shell expr $(ROM_SIZE_KB) '*' 1024)
  RAM_SIZE := $(shell expr $(RAM_SIZE_KB) '*' 1024)
  STACK_SIZE := $(shell expr $(STACK_SIZE_KB) '*' 1024)
  LDFLAGS += -Xlinker --defsym=PULPINO_RAM_START="$(ROM_SIZE)"
  LDFLAGS += -Xlinker --defsym=PULPINO_RAM_SIZE="$(RAM_SIZE)"
  LDFLAGS += -Xlinker --defsym=PULPINO_STACK_SIZE="$(STACK_SIZE)"


  PLATFORM_FLAGS += \
    -fno-builtin \
    -ffreestanding \
    -fno-builtin-printf 
    -fno-delete-null-pointer-checks \
    -fno-threadsafe-statics \
    -fomit-frame-pointer \
    -fno-use-cxa-atexit

  INCLUDES += -I$(PULPINO_CRT)/pulpino/sw/libs/sys_lib/inc -I$(PULPINO_CRT)/pulpino/sw/libs/string_lib/inc


  # paths relative to TF-lite(u) installation ...
  SYS_LIB_CC_FILES := uart.c int.c exceptions.c i2c.c gpio.c spi.c timer.c

  PLATFORM_CC_SRCS := \
   $(PULPINO_CRT)/pulpino_tumeda/crt0.riscv.S \
   $(PULPINO_CRT)/pulpino_tumeda/syscalls.c \
   $(foreach f,$(SYS_LIB_CC_FILES),$(PULPINO_CRT)/pulpino/sw/libs/sys_lib/src/$(f))

$(BINDIR)%.bin: $(BINDIR)%
	@mkdir -p $(dir $@)
	$(OBJCOPY) $< $@ -O binary
	$(READELF) -a $< > "$*.txt"
	dd bs=1024 count=$(ROM_SIZE_KB) if=$@ of="$*.rom"
	dd bs=1024 skip=$(ROM_SIZE_KB) if=$@ of="$*.ram"
	$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)size $<

  TEST_SCRIPT :=  $(MAKEFILE_DIR)/gen/test_etiss_binary.sh
  TEST_BIN_TARGET_SUFFIX := .bin
endif

ifneq ($(subst ifx_riscv32,,$(TARGET)),$(TARGET))
ifeq ($(IFX_RISCV32_RECOGNIZED),)
error("Unorecognised ifx_riscv32* target $(TARGET))
endif
endif
