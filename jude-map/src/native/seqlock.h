// Seqlock: writer-priorty, reader-optimistic, zero kernel involvment.
//
// Writer: write_begin() -> mutate data -> write_end()
// Reader: read_begin() -> read data -> read_end() -> if read_retry(seq): retry
//
// The counter is odd while a write is in progress.
// Isolated to its own 64-byte cache line so writer updates to the counter
// do not invalidate the reader's cache of the tensor metadata block.
#include <atomic>

struct alignas(64) Seqlock
{
    std::atomic<uint64_t> sequence{0}; // 64-bit counter, incremented by the writer, initialized to 0.

    // Called by writer before touching shared data.
    inline void write_begin() noexcept
    {
        // Odd value signals write-in-progress to readers.
        sequence.fetch_add(1, std::memory_order_release);
    }

    // Called by writer after all shared data is commited.
    inline void write_end() noexcept
    {
        sequence.fetch_add(1, std::memory_order_release); // Increment to even value, signaling write completion.
    }

    // Called by reader before sampling shared data.
    // Spins until no write is in progress, returns stable sequqnece value.
    inline uint64_t read_begin() const noexcept
    {
        uint64_t seq;
        do
        {
            seq = sequence.load(std::memory_order_acquire);
        } while (seq & 1u); // spin while odd
        return seq;
    }

    // Called by reader after sampling. Returns true if data was torn.
    inline bool read_retry(uint64_t seq) const noexcept
    {
        std::atomic_thread_fence(std::memory_order_acquire);    // ensure visibility of sampled data before re-checking sequence
        return sequence.load(std::memory_order_relaxed) != seq; // if sequence changed, data may be torn, retry
    }
};

static_assert(sizeof(Seqlock) == 64, "Seqlock must be exactly one cache line to avoid false sharing");