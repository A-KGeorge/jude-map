#pragma once
// ---------------------------------------------------------------------------
// platform_tf.h
//
// Resolves the libtensorflow include and library paths at build time via
// binding.gyp variables, and at load time for runtime linking.
//
// Environment variable LIBTENSORFLOW_PATH overrides the default search.
// Default search paths:
//   Windows : C:\libtensorflow
//   Linux   : /usr/local
//   macOS   : /usr/local  (or Homebrew prefix)
// ---------------------------------------------------------------------------

// The actual header — binding.gyp adds the correct include_dir so this
// include resolves regardless of install location.
#include "tensorflow/c/c_api.h"

// Some TensorFlow C distributions do not define TF_MAJOR_VERSION in c_api.h.
// Only enforce compile-time version checks when version macros are available.
#if defined(TF_MAJOR_VERSION)
#if TF_MAJOR_VERSION < 2
#error "jude-tf requires TensorFlow C library 2.x or later"
#endif
#endif