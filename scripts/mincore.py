import ctypes, mmap, os, glob
libc = ctypes.CDLL("libc.so.6", use_errno=True)
M = os.path.expanduser("~/.cache/huggingface/hub/models--<org>--Qwen3.8-Flash-Next-NVFP4/snapshots/<revision>")
PAGE = os.sysconf("SC_PAGE_SIZE")
tot_res = tot_pages = 0
files = sorted(glob.glob(M + "/model-plefp8-*.safetensors")) + sorted(glob.glob(M + "/model-000*.safetensors"))[:2] + sorted(glob.glob(M + "/layer-00000-experts-0000-0127.safetensors"))
for f in files:
    f = os.path.realpath(f); size = os.path.getsize(f)
    fd = os.open(f, os.O_RDONLY)
    mm = mmap.mmap(fd, size, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    npages = (size + PAGE - 1) // PAGE
    vec = (ctypes.c_ubyte * npages)()
    buf = (ctypes.c_char * size).from_buffer(mm)
    r = libc.mincore(ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(size), vec)
    if r != 0: print("mincore failed", os.strerror(ctypes.get_errno()), f); continue
    res = sum(1 for i in range(npages) if vec[i] & 1)
    print(f"{os.path.basename(f):<45} {res*PAGE/1e9:6.2f} GB resident of {size/1e9:6.2f} GB  ({100*res/npages:5.1f}%)")
    if "plefp8" in f: tot_res += res; tot_pages += npages
    del buf; mm.close(); os.close(fd)
print(f"PLE TOTAL: {tot_res*PAGE/1e9:.2f} GB resident of {tot_pages*PAGE/1e9:.2f} GB ({100*tot_res/max(1,tot_pages):.1f}%)  page={PAGE}")
