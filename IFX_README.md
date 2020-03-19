
## General

### Relationship to upstream tensorflow

Like all repos with external upstreams an ifx prefix is used to ensure branch-name separation from the upstream.   Ifx/develop is our current development mainline with (one day) ifx/master for stable mainline.
Currently ifx/develop is a branch off the upstream r2.1 release branch .

### Working with this repo / building

* Build is possible under: Windows WSL(Ubuntu 18.04),  R&D Cluster RH7 hosts or  MSYS2 under Windows.   See parent aiml_deployment repo README.md for setup details.
  
* I (Andrew) currently use Eclipse CDT for both the toco translator (bazel build) and  TF-lite(u) (GNU make).  Visual Studio Code works also passably well too but needs workarounds to avoid crashing its build target detection on latent bugs in the TF bazel scripts.  Patches for this are in VS_code_bazel_query.patch.
  
* Before attempting to build the TF-liter Translator toco you probably need to trigger a tensorflow build (which downloads a shit-load of upstream dependencies from the wider Internet Ecosystem and insantiates a bunch of templates possibly used in sub-builds) before working actively in the repo.  This is not necessary for TF-lite(u).

### Known Issues and Bugs

* Bazel requires a filesystem with correct / sane behaviour.  

    * Build *does not work* on NTFS volumes accessed through a WSL(1) mount (at least for our crap-tastic virus-scanner enriched iSCvX clients).

    * You will need to shift bazelroot (see docs) onto a /tmp or similar on the R&D cluster.


## Tensorflow lite (micro)

* The developer docs and source  for TF-lite(micro) are found in
[tensorflow/lite/experimental/micro](https://bitbucket.vih.infineon.com/projects/AIML_DEPLOYMENT/repos/tensorflow/browse/tensorflow/lite/experimental/micro?at=refs%2Fheads%2Fifx%2Fdevelop)

* Following the google/tensorflow philosophy  EVERYTHING is intended to be driven from the top-level project root directory.   
Despite the GNU make build for stand-alone (cross)compilation being separate from the bazel used for tensorflow build proper you cannot currently disentangle TF-lite micro from the tensorflow repository as it shares some headers with TF-lite proper.  

  * AFAIK the parallel bazel BUILD stuff here is used to build a host-compiled TF-lite micro interpreter to allow testing from within the tensorflow python environment (e.g. “read your own writing” tests and cross-comparison of running compiled tflite(micro) flat-buffers against source/reference tensorflow models.

* A Eclipse CDT (2019_12) project I (andrew) use to work on toco and TF-lite(u) with some example debug launch configurations for WSL Linux and MSYS2 setups under windows can be found in the root directory of the fork repo.

* A shim makefile for building tflite(u), running tests etc is at the top-level dir of the tensorflow project (saves typing a lengthy path to the actual Makefile).

### Extensions / additions in the IFX fork

For usage in aiml_deployment setting the original google Makefiles have been extended. 

* Additional targets for IFX RISCV  (more will be  addded!)
  *  ifx_riscv32_mcu (build for semi-hosted execution on SWERV-ISS simulator or HW platform)
  *  ifx_riscv32_pulpino (build for pulpino-like minimal C++ runtime bare-metal environment - a matching config of ETISS simulator is available, SWERV-ISS setup is TODO)
  *  You can specify a host-native (x86_64) build as default or TARGET=hostnative
* Debug build support.   Set BUILD make variable to DEBUG.  Default is RELEASE.
*  An `install` target to copy into place an installed copy in location derived from the parent aiml_deployment  working copy's `SETTINGS_AND_VERSIONS.sh`.  
  * The Makefile in the installed copy's tools/make can be used to compile external examples see `egs_and_tests/tflite` for examples.
  * Pathnames for generated files have been made  dependent on target architecture and build type.


Some example build command-lines...
* To compile for IFX generic RISCV using aiml_deployment gcc:  `make TARGET=ifx_riscv32_mcu` ...
* TO compile debug build (no optimization, debug symbols):  `make  BUILD=DEBUG` ...
* Some handy targets:
   * hello_world example:   `make hello_world`
   * hello_world unit tests:  `make test_hello_world`
   * Full test suite:  `make test`
   * Full test suite on SWRV-ISS RISCV32 simulator:  `make  TARGET=ifx_riscv32_mcu test`

### Debugging with gdb / SWERV-IIS / additions in the IFX fork

Start your executable on SWERV-ISS simulator with gdb server activated:

`/opt/swerviss-`*version*`/bin/whisper --gdb --gdb-tcp-port` *port_num* *path_to_elf*


Start gdb with remote target localhost:*port_num** - Eclipse CDT works well as a GUI front-end to avoid gdb command-line self-flagellation.  See CDT project in repo root for examples of CDT launch configurations for some sample TF-lite(u) unit test executables.

Current swerv-iss *version* is 1.4.59.0, a reasonable tcp port number (*port_num*) is 8080.


