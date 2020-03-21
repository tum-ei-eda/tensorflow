#!/bin/bash
# Some minor hacks to suppress breakages in bazel scripts
# that poison VS-code bazel target discovery
cp third_party/toolchains/cpus/arm/cc_config.bzl.tpl third_party/toolchains/cpus/arm/cc_config.bzl
patch --strip=1  < VS_code_bazel_query.patch
