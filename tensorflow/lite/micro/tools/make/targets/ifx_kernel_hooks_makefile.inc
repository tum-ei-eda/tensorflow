#
#
#  Set CXXFLAGS based on tags for kernel hooks/precompilation
#
#


ifneq ($(filter recorded_model,$(TAGS)),)
    $(info Using recorded model-specific kernel-variant selection)
    CXXFLAGS += -DTF_LITE_MICRO_USE_RECORDED_KERNEL_VARIANTS
endif

ifneq ($(filter record_model,$(TAGS)),)
    $(info Recording models-specific prepare-phase kernel variant selection)
    CXXFLAGS += -DTF_LITE_MICRO_RECORD_STATIC_KERNEL_VARIANT
endif

ifneq ($(filter autodump,$(TAGS)),)
    CXXFLAGS += -DTF_LITE_MICRO_AUTO_DUMP_POINTER_TABLES
endif
