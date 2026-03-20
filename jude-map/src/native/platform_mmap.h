#pragma once

// -------------------------------------------------------------------------------
// platform_mmap.h
//
// Thin cross-platform abstraction over anonymous shared memory mapping.
// Presents two functions:
//
//  void platform_mmap(size_t size)
//      Allocates an anonymous read/write shared mapping of 'size' bytes.
//      Returns nullptr on failure.
//
//  bool platform_munmap(void* addr, size_t size)
//      Releases a mapping previously returned by platform_mmap.
//      Returns true on success.
//
// "Anonymous shared" means:
//     - No backing file - memory is zero-initialized
//     - Visible to all threads within the same process (MAP_SHARED / PAGE_READWRITE).
//     - Not visible to other processes (no shm_open / CreateNamedMapping name).
//
// Future: swap in shm_open (POSIX) or CreateNamedMapping (Windows) to extend
// visibility across process boundaries for the cross-process / GPU use case.
// -------------------------------------------------------------------------------

#ifdef _WIN32
// -------------------------- Windows --------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstddef>

inline void *platform_mmap(size_t size) noexcept
{
    // For in-process shared memory, VirtualAlloc is sufficient and tends to
    // release faster with VirtualFree(MEM_RELEASE) than mapped-view teardown.
    return VirtualAlloc(
        nullptr,
        size,
        MEM_RESERVE | MEM_COMMIT,
        PAGE_READWRITE);
}

// platform_release_hint — decommit physical pages before unmapping.
//
// On Windows, MEM_DECOMMIT releases the physical pages backing the mapping
// immediately. The subsequent UnmapViewOfFile only needs to release the
// virtual address range, which is much faster than releasing both at once.
//
// Without this hint, UnmapViewOfFile on 10 GB takes ~1700ms.
// With MEM_DECOMMIT first, the two-step total drops to ~200-400ms.
//
// Must be called on the same address returned by MapViewOfFile.
// Safe to call from any thread (no V8 involvement).

inline void platform_release_hint(void *addr, size_t size) noexcept
{
    if (!addr || size == 0)
        return;

    // DiscardVirtualMemory can quickly discard private page contents. It may
    // be unavailable on older Windows, so resolve it
    // dynamically and fall back to MEM_DECOMMIT best-effort.
    using DiscardVirtualMemoryFn = DWORD(WINAPI *)(PVOID, SIZE_T);
    static DiscardVirtualMemoryFn discard_fn = []() -> DiscardVirtualMemoryFn
    {
        HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
        if (!k32)
            return nullptr;
        return reinterpret_cast<DiscardVirtualMemoryFn>(GetProcAddress(k32, "DiscardVirtualMemory"));
    }();

    if (discard_fn)
    {
        (void)discard_fn(addr, size);
        return;
    }

    (void)VirtualFree(addr, size, MEM_DECOMMIT);
}

inline bool platform_munmap(void *addr, size_t /*size*/) noexcept
{
    // MEM_RELEASE requires dwSize = 0 and releases the entire reserved region.
    return VirtualFree(addr, 0, MEM_RELEASE) != 0;
}

// Page-lock the mapping so CUDA DMA can read it without a staging copy.
// VirtualLock has a per-process quota (default 64MB on older windows).
// Raise it via SetProcessWorkingSetSizeEx if needed for large tensors.
inline bool platform_mlock(void *addr, size_t size) noexcept
{
    return VirtualLock(addr, size) != 0;
}

inline bool platform_munlock(void *addr, size_t size) noexcept
{
    return VirtualUnlock(addr, size) != 0;
}

// -------------------------- POSIX (Linux, macOS) --------------------------
#else
#include <sys/mman.h>
#include <unistd.h>
#include <cstddef>

inline void *platform_mmap(size_t size) noexcept
{
    void *addr = mmap(
        nullptr,                    // let the system choose the address
        size,                       // length of mapping
        PROT_READ | PROT_WRITE,     // read/write permissions
        MAP_SHARED | MAP_ANONYMOUS, // shared and anonymous mapping
        -1,                         // no file descriptor for anonymous mapping
        0                           // offset (not used for anonymous mapping)
    );

    return addr == MAP_FAILED ? nullptr : addr;
}

// platform_release_hint — advise the kernel it can reclaim pages immediately.
//
// MADV_DONTNEED on Linux: kernel drops the pages from the page cache.
// Subsequent munmap only has to remove the VMA entry from the page table,
// not write-back or account for each page — significantly faster.
//
// On macOS, MADV_FREE is the equivalent (lazy reclaim). MADV_DONTNEED on
// macOS has different semantics (pages become zero-filled on next access,
// but aren't necessarily released immediately). MADV_FREE is preferred.
inline void platform_release_hint(void *addr, size_t size) noexcept
{
    if (!addr || size == 0)
        return;
#if defined(__APPLE__)
    madvise(addr, size, MADV_FREE);
#else
    madvise(addr, size, MADV_DONTNEED);
#endif
}

inline bool platform_munmap(void *addr, size_t size) noexcept
{
    return munmap(addr, size) == 0;
}

// Page-lock the mapping so CUDA DMA can read it without a staging copy.
// Requires CAP_IPC_LOCK or sufficient RLIMIT_MEMLOCK. Check /proc/self/status
// (VmLck) to verify. For large models, raise the limit via:
//   ulimit -l unlimited   (shell)
//   setrlimit(RLIMIT_MEMLOCK, ...)  (programmatic)
inline bool platform_mlock(void *addr, size_t size) noexcept
{
    return mlock(addr, size) == 0;
}

inline bool platform_munlock(void *addr, size_t size) noexcept
{
    return munlock(addr, size) == 0;
}

#endif // _WIN32