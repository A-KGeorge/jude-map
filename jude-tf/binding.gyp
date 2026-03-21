{
  "targets": [
    {
      "target_name": "jude-tf",
      "sources": [
        "src/native/tf_session.cc",
        "src/native/platform_tf.h",
        "src/native/proto_parser.h",        
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "<!@(echo $LIBTENSORFLOW_PATH || echo /usr/local || true)/include"
      ],
      "dependencies": [
          "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "libraries": [
        "-ltensorflow"
      ],
      "library_dirs": [
        "<!@(echo $LIBTENSORFLOW_PATH || echo /usr/local || true)/lib"
      ],
      "sources!": [
      ],
      "sources+": [
      ],
      "defines": [
        "NAPI_VERSION=8",
        "NAPI_DISABLE_CPP_EXCEPTIONS"
      ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "cflags": [ "-O3", "-ffast-math" ],
      "cflags_cc": [ "-std=c++17", "-O3", "-ffast-math" ],      
      "msvs_settings": {
        "VCCLCompilerTool": {
          "ExceptionHandling": 1,
          "AdditionalOptions": [ "/std:c++17", "/O2", "/fp:fast", "/arch:AVX2" ],
          "Optimization": 3,
          "FavorSizeOrSpeed": 1,
          "InlineFunctionExpansion": 2
        }
      },
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "CLANG_CXX_LIBRARY": "libc++",
        "MACOSX_DEPLOYMENT_TARGET": "10.15",
        "OTHER_CPLUSPLUSFLAGS": [ "-std=c++17", "-stdlib=libc++", "-O3", "-ffast-math"],
        "GCC_OPTIMIZATION_LEVEL": "3"
      },
      "conditions": [
        # Condition for Windows
        ["OS=='win'", {
          "defines": [ "_HAS_EXCEPTIONS=1", "__AVX2__=1", "__AVX__=1", "__SSE3__=1" ],
          "include_dirs": [
            "C:/libtensorflow/include"
          ],
          "library_dirs": [
            "C:/libtensorflow/lib"
          ]
        }],
        # Condition for x64 architecture (Linux/macOS)
        ['target_arch=="x64"', {
          "cflags+": [ "-msse3", "-mavx", "-mavx2" ],
          "cflags_cc+": [ "-msse3", "-mavx", "-mavx2" ],
          'xcode_settings': {
            'OTHER_CPLUSPLUSFLAGS+': [ '-msse3', '-mavx', '-mavx2' ]
          }
        }],
        # Condition for ia32 architecture (Linux/macOS - if you support 32-bit x86)
        ['target_arch=="ia32"', {
           "cflags+": [ "-msse3", "-mavx", "-mavx2" ], # Or adjust based on 32-bit support
           "cflags_cc+": [ "-msse3", "-mavx", "-mavx2" ],
          'xcode_settings': {
            'OTHER_CPLUSPLUSFLAGS+': [ '-msse3', '-mavx', '-mavx2' ]
          }
        }],
        # Condition for arm64 architecture (Android, iOS, M1/M2 Macs, etc.)
        ['target_arch=="arm64"', {
          # ARMv8-a baseline: NEON + FP support (compatible with all ARMv8 CPUs)
          "cflags+": [ "-march=armv8-a+fp+simd" ],
          "cflags_cc+": [ "-march=armv8-a+fp+simd" ],
          'xcode_settings': {
            'OTHER_CPLUSPLUSFLAGS+': [ '-march=armv8-a+fp+simd' ]
          }
          # Optional: Upgrade to ARMv8.2-a for newer CPUs (Tensor G4, Apple M2+, Graviton 3+)
          # Enables FP16 arithmetic and additional optimizations
          # Uncomment the lines below to enable ARMv8.2-a:
          # "cflags+": [ "-march=armv8.2-a+fp16" ],
          # "cflags_cc+": [ "-march=armv8.2-a+fp16" ],
          # 'xcode_settings': {
          #   'OTHER_CPLUSPLUSFLAGS+': [ '-march=armv8.2-a+fp16' ]
          # }
        }],
        # Condition for 32-bit ARM (older Android devices)
        ['target_arch=="arm"', {
          "cflags+": [ "-mfpu=neon", "-mfloat-abi=hard" ],  # 32-bit ARM NEON
          "cflags_cc+": [ "-mfpu=neon", "-mfloat-abi=hard" ],
        }]
      ]
    }
  ]
}