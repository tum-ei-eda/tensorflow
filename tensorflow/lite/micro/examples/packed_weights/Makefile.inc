HELLO_WORLD_TEST_SRCS := \
tensorflow/lite/micro/examples/packed_weights/packed_weights_test.cc \
tensorflow/lite/micro/examples/packed_weights/hello_world_4bit_q.cc

HELLO_WORLD_TEST_HDRS := \
tensorflow/lite/micro/examples/packed_weights/hello_world_4bit_q_data.h

OUTPUT_HANDLER_TEST_SRCS := \
tensorflow/lite/micro/examples/packed_weights/output_handler_test.cc \
tensorflow/lite/micro/examples/packed_weights/output_handler.cc

OUTPUT_HANDLER_TEST_HDRS := \
tensorflow/lite/micro/examples/packed_weights/output_handler.h \
tensorflow/lite/micro/examples/packed_weights/constants.h



include $(wildcard tensorflow/lite/micro/examples/packed_weights/*/Makefile.inc)

# Tests loading and running the sine model.
$(eval $(call microlite_test,packed_weights_test,\
$(HELLO_WORLD_TEST_SRCS),$(HELLO_WORLD_TEST_HDRS)))




