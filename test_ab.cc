#include <napi.h>
#include <iostream>

Napi::Value Check(const Napi::CallbackInfo& info) {
    if (info[0].IsArrayBuffer()) {
        std::cout << "IS AB" << std::endl;
        return info.Env().Undefined();
    }
    if (info[0].IsTypedArray()) {
        std::cout << "IS TA" << std::endl;
        return info.Env().Undefined();
    }
    std::cout << "NO" << std::endl;
    return info.Env().Undefined();
}
Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("check", Napi::Function::New(env, Check));
    return exports;
}
NODE_API_MODULE(node_addon, Init)
