#pragma once
#include "segment.h"
#include <atomic>
#include <cstring>
#include <thread>
#include <cstdint>

// ---------------------------------------------------------------------------
// Lock-free ring buffer segment for streaming tensor pipelines.
//
// Design:
//   - Fixed-capacity circular buffer of tensor slots.
//   - Each slot has an atomic state: EMPTY → WRITING → READY → READING → EMPTY
//   - CAS state transitions — no mutex, no seqlock, no retry on data.
//   - Backpressure: spin briefly → yield → return false (non-blocking push/pop)
//     Callers decide whether to block (re-call), drop, or propagate back-pressure.
//
// Capacity must be a power of 2 (enforced in rcu_ring_init).
//
// Memory layout in the SAB / mmap region:
//   [RingHeader: 128 bytes]
//   [RingSlot[0..capacity-1]: capacity * sizeof(RingSlot)]
//   [slot_data[0..capacity-1]: capacity * max_bytes_per_slot]
//
// Total allocation: RING_HEADER_SIZE + capacity*(sizeof(RingSlot) + max_bytes_per_slot)
//
// Typical use (inference pipeline):
//   Producer (main thread): ring_push(input_tensor)  — returns false if full
//   Consumer (worker):      ring_pop(output_tensor)  — returns false if empty
// ---------------------------------------------------------------------------

// Slot states — stored as uint32_t for atomic CAS.
static constexpr uint32_t RING_SLOT_EMPTY = 0;
static constexpr uint32_t RING_SLOT_WRITING = 1;
static constexpr uint32_t RING_SLOT_READY = 2;
static constexpr uint32_t RING_SLOT_READING = 3;

// Spin iteration thresholds for the hybrid backpressure strategy.
// Tune based on hardware — these defaults suit a ~10µs inference budget.
static constexpr uint32_t RING_SPIN_TIGHT = 32;  // tight spin (no yield)
static constexpr uint32_t RING_SPIN_YIELD = 256; // yield phase
// After RING_SPIN_YIELD iterations the caller returns false (non-blocking).

// ---------------------------------------------------------------------------
// RingSlot: per-slot header (state + tensor metadata).
// Aligned to 64 bytes so state changes don't false-share with adjacent slots.
// ---------------------------------------------------------------------------
struct alignas(64) RingSlot
{
    std::atomic<uint32_t> state{RING_SLOT_EMPTY};
    uint8_t _pad[60 - sizeof(TensorMeta) % 64]; // compact padding
    TensorMeta meta;

    RingSlot() : state(RING_SLOT_EMPTY) {}
};

// Ensure slot header is reasonably sized (not a hard constraint).
static_assert(sizeof(RingSlot) >= 64, "RingSlot must be at least one cache line");

// ---------------------------------------------------------------------------
// RingHeader: the fixed header at offset 0 in the ring segment.
//
//  [0..63]   capacity, max_bytes_per_slot, write_pos, read_pos + pad
//  [64..127] reserved / future fields
// ---------------------------------------------------------------------------
struct alignas(64) RingHeader
{
    uint32_t capacity;           // number of slots (must be power of 2)
    uint32_t max_bytes_per_slot; // max byte payload per slot
    uint8_t _pad0[56];           // pad to 64 bytes

    std::atomic<uint32_t> write_pos{0}; // producer position (monotonic)
    std::atomic<uint32_t> read_pos{0};  // consumer position (monotonic)
    uint8_t _pad1[56];                  // pad to 64 bytes
};

static_assert(sizeof(RingHeader) == 128,
              "RingHeader must be 128 bytes");
static_assert(offsetof(RingHeader, write_pos) == 64,
              "write_pos / read_pos must be on a separate cache line from capacity");

static constexpr size_t RING_HEADER_SIZE = sizeof(RingHeader); // 128 bytes

// ---------------------------------------------------------------------------
// Layout accessors — all arithmetic from the base pointer.
// ---------------------------------------------------------------------------

inline RingHeader *ring_header(void *base)
{
    return reinterpret_cast<RingHeader *>(base);
}

inline RingSlot *ring_slots(void *base)
{
    return reinterpret_cast<RingSlot *>(
        reinterpret_cast<uint8_t *>(base) + RING_HEADER_SIZE);
}

inline uint8_t *ring_slot_data(void *base, uint32_t slot_idx, size_t max_bytes_per_slot)
{
    const auto *hdr = reinterpret_cast<const RingHeader *>(base);
    // Slot data array begins after the slot headers.
    const size_t slots_region = hdr->capacity * sizeof(RingSlot);
    return reinterpret_cast<uint8_t *>(base) + RING_HEADER_SIZE + slots_region + static_cast<size_t>(slot_idx) * max_bytes_per_slot;
}

// Total SAB / mmap allocation size for a ring segment.
inline size_t ring_segment_size(uint32_t capacity, size_t max_bytes_per_slot)
{
    return RING_HEADER_SIZE + capacity * sizeof(RingSlot) + capacity * max_bytes_per_slot;
}

// Returns true if n is a power of 2 and > 0.
inline bool ring_is_power_of_two(uint32_t n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

// Placement-new initialise the ring header + slot states.
// capacity must be a power of 2.
inline bool ring_init(void *base, uint32_t capacity, uint32_t max_bytes_per_slot)
{
    if (!ring_is_power_of_two(capacity))
        return false;

    auto *hdr = new (base) RingHeader();
    hdr->capacity = capacity;
    hdr->max_bytes_per_slot = max_bytes_per_slot;
    hdr->write_pos.store(0, std::memory_order_relaxed);
    hdr->read_pos.store(0, std::memory_order_relaxed);

    auto *slots = ring_slots(base);
    for (uint32_t i = 0; i < capacity; ++i)
        new (&slots[i]) RingSlot();

    return true;
}

// ---------------------------------------------------------------------------
// ring_push — non-blocking producer.
//
// Returns true if the slot was claimed and data written.
// Returns false if the ring is full after RING_SPIN_YIELD iterations
// (backpressure: caller should throttle or drop).
//
// Algorithm:
//   1. Read write_pos and compute slot_idx = write_pos % capacity.
//   2. CAS slot state: EMPTY → WRITING. If fails, spin.
//   3. Write metadata and data.
//   4. Advance write_pos (monotonic, wraps naturally at UINT32_MAX).
//   5. Release slot: WRITING → READY.
// ---------------------------------------------------------------------------
inline bool ring_push(void *base,
                      const uint8_t *src,
                      size_t src_size,
                      const TensorMeta &meta)
{
    auto *hdr = ring_header(base);
    auto *slots = ring_slots(base);
    const uint32_t cap = hdr->capacity;
    const uint32_t mask = cap - 1;
    const size_t mbs = hdr->max_bytes_per_slot;

    if (src_size > mbs)
        return false; // caller error — don't assert in hot path

    for (uint32_t attempt = 0; attempt < RING_SPIN_YIELD; ++attempt)
    {
        const uint32_t pos = hdr->write_pos.load(std::memory_order_relaxed);
        const uint32_t slot_idx = pos & mask;
        RingSlot &slot = slots[slot_idx];

        uint32_t expected = RING_SLOT_EMPTY;
        if (slot.state.compare_exchange_weak(
                expected, RING_SLOT_WRITING,
                std::memory_order_acquire,
                std::memory_order_relaxed))
        {
            // We own this slot.
            std::memcpy(&slot.meta, &meta, sizeof(TensorMeta));
            std::memcpy(ring_slot_data(base, slot_idx, mbs), src, src_size);

            // Advance producer position BEFORE marking ready so pop() can't
            // claim the same slot from a different producer concurrently.
            hdr->write_pos.fetch_add(1, std::memory_order_relaxed);

            // Release: make data visible, transition to READY.
            slot.state.store(RING_SLOT_READY, std::memory_order_release);
            return true;
        }

        // Slot not EMPTY: ring is full or another producer is writing.
        if (attempt < RING_SPIN_TIGHT)
            ; // tight spin — avoids yield overhead for brief waits
        else
            std::this_thread::yield();
    }

    return false; // backpressure: ring full
}

// ---------------------------------------------------------------------------
// ring_pop — non-blocking consumer.
//
// Returns true if a READY slot was claimed and data read.
// Returns false if the ring is empty after RING_SPIN_YIELD iterations.
//
// If copy=true:  copies tensor bytes into `out` (safe, no lifetime concern).
// If copy=false: fills `data_ptr_out` with a pointer into the ring slot's
//                data region. The pointer is valid until the slot transitions
//                back to EMPTY (i.e. until the next ring_pop_release() call
//                with the same slot_idx_out). Use for zero-copy paths where
//                the caller releases the slot explicitly after consuming.
// ---------------------------------------------------------------------------
inline bool ring_pop(void *base,
                     TensorMeta &meta_out,
                     uint8_t *out, // dest for copy mode
                     bool copy,
                     // Zero-copy out params (used when copy=false):
                     uint32_t *slot_idx_out = nullptr,
                     uint8_t **data_ptr_out = nullptr)
{
    auto *hdr = ring_header(base);
    auto *slots = ring_slots(base);
    const uint32_t cap = hdr->capacity;
    const uint32_t mask = cap - 1;
    const size_t mbs = hdr->max_bytes_per_slot;

    for (uint32_t attempt = 0; attempt < RING_SPIN_YIELD; ++attempt)
    {
        const uint32_t pos = hdr->read_pos.load(std::memory_order_relaxed);
        const uint32_t slot_idx = pos & mask;
        RingSlot &slot = slots[slot_idx];

        uint32_t expected = RING_SLOT_READY;
        if (slot.state.compare_exchange_weak(
                expected, RING_SLOT_READING,
                std::memory_order_acquire,
                std::memory_order_relaxed))
        {
            // We own this slot for reading.
            std::memcpy(&meta_out, &slot.meta, sizeof(TensorMeta));

            uint8_t *src = ring_slot_data(base, slot_idx, mbs);

            if (copy)
            {
                std::memcpy(out, src, meta_out.byte_length);
                hdr->read_pos.fetch_add(1, std::memory_order_relaxed);
                slot.state.store(RING_SLOT_EMPTY, std::memory_order_release);
            }
            else
            {
                // Zero-copy: caller must call ring_pop_release() when done.
                if (slot_idx_out)
                    *slot_idx_out = slot_idx;
                if (data_ptr_out)
                    *data_ptr_out = src;
                hdr->read_pos.fetch_add(1, std::memory_order_relaxed);
                // Slot stays in READING state until ring_pop_release().
            }

            return true;
        }

        // Slot not READY: ring is empty or another consumer beat us.
        if (attempt < RING_SPIN_TIGHT)
            ;
        else
            std::this_thread::yield();
    }

    return false; // ring empty
}

// ---------------------------------------------------------------------------
// ring_pop_release — release a zero-copy slot back to EMPTY.
// Must be called after ring_pop with copy=false once the caller is done
// accessing the data pointer.
// ---------------------------------------------------------------------------
inline void ring_pop_release(void *base, uint32_t slot_idx)
{
    auto *slots = ring_slots(base);
    slots[slot_idx].state.store(RING_SLOT_EMPTY, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// ring_size — approximate number of READY items in the ring.
// Not exact under concurrency — useful for monitoring / backpressure hints.
// ---------------------------------------------------------------------------
inline uint32_t ring_size(const void *base)
{
    const auto *hdr = reinterpret_cast<const RingHeader *>(base);
    const uint32_t w = hdr->write_pos.load(std::memory_order_relaxed);
    const uint32_t r = hdr->read_pos.load(std::memory_order_relaxed);
    return w - r; // wraps correctly for uint32_t arithmetic
}