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
    // CreateFIleMapping with INVALID_HANDLE_VALUE  creates an anonymous
    // page-file-backed mapping. PAGE_READWRITE gives read+write access.
    HANDLE hMap = CreateFileMappingA(
        INVALID_HANDLE_VALUE,                          // anonymous - backed by page file
        nullptr,                                       // default security attributes
        PAGE_READWRITE,                                // read/write access
        static_cast<DWORD>((size >> 32) & 0xFFFFFFFF), // max size (high 32 bits)
        static_cast<DWORD>(size & 0xFFFFFFFF),         // max size (low 32 bits)
        nullptr                                        // unnamed - not accessible by other processes
    );

    if (hMap == nullptr || hMap == INVALID_HANDLE_VALUE)
    {
        return nullptr;
    }

    void *addr = MapViewOfFile(
        hMap,
        FILE_MAP_ALL_ACCESS, // read/write access
        0,                   // file offset high
        0,                   // file offset low
        size                 // number of bytes to map
    );

    // The mapping stays alive as long as MapViewOfFile views exist.
    // Closing the handle here is safe — the view holds its own reference.
    CloseHandle(hMap);

    return addr; // nullptr on failure
}

inline bool platform_munmap(void *addr, size_t /*size*/) noexcept
{
    // UnmapViewOfFile does not need the size on Windows.
    return UnmapViewOfFile(addr) != 0;
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