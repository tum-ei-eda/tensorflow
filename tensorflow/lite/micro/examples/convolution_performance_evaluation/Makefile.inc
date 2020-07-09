CONV_PERFORMANCE_TEST_SRC := \
tensorflow/lite/micro/examples/convolution_performance_evaluation/conv_performance_test.cc

DCONV_PERFORMANCE_TEST_SRC := \
tensorflow/lite/micro/examples/convolution_performance_evaluation/depthwise_conv_performance_test.cc

#Find any platform - specific rules for the examples.
include $(wildcard tensorflow/lite/micro/examples/convolution_performance_evaluation/*/Makefile.inc)

# Tests loading and running the conv performance test.
$(eval $(call microlite_test,conv_performance_test,\
$(CONV_PERFORMANCE_TEST_SRC)))

# Tests loading and running the depthwise conv performance test.
$(eval $(call microlite_test,depthwise_conv_performance_test,\
$(DCONV_PERFORMANCE_TEST_SRC)))
