from magicalponies import ffi, lib  # pyright: ignore[reportMissingImports]

def add_b(a, b):
    sum = ffi.new("float*")
    sum = lib.add_bbnos(a, b)
    return sum