%%writefile profile_allocator.py
import torch
import gc
from memory_allocator import MemoryAllocator

def run_profile():
    # Warmup
    dummy = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("Entire Allocator Lifecycle")

    torch.cuda.nvtx.range_push("1. Initialize Memory Pool")
    allocator = MemoryAllocator(2048, 2048, torch.float16, 8, 128, 4)
    torch.cuda.synchronize() 
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("2. Alloc 500 blocks")
    indices_1 = allocator.alloc(500)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("3. Alloc Strip")
    indices_strip = allocator.alloc_strip(need_block=10, block_size=8)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("4. Free first 500 blocks")
    allocator.free(indices_1)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("5. Alloc Contiguous")
    indices_contig, start, end = allocator.alloc_contiguous(200)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("6. Delete Pool and GC")
    allocator.delete_all_pool()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop() 

if __name__ == "__main__":
    run_profile()