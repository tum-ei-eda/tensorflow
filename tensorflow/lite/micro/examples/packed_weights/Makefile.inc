PACKED_WEIGHTS_TEST_SRCS := \
tensorflow/lite/micro/examples/packed_weights/packed_weights_test.cc \
tensorflow/lite/micro/examples/packed_weights/hello_world_2x4in8.cc

PACKED_WEIGHTS_TEST_HDRS := \
tensorflow/lite/micro/examples/packed_weights/hello_world_2x4in8.h \
tensorflow/lite/micro/examples/packed_weights/hello_world_2x4in8refdata.h

include $(wildcard tensorflow/lite/micro/examples/packed_weights/*/Makefile.inc)

# Tests loading and running the sine model.
$(eval $(call microlite_test,packed_weights_test,\
$(PACKED_WEIGHTS_TEST_SRCS),$(PACKED_WEIGHTS_TEST_HDRS)))




