from __future__ import annotations

from csv import reader
from multiprocessing import shared_memory
from os import name
from typing import TypeAlias

import numpy as np
from scipy.fftpack import dst


__all__ = ["SharedBuffer"]

RingView: TypeAlias = tuple[memoryview, memoryview | None, int, bool]


class SharedBuffer(shared_memory.SharedMemory):
    """
    Applicant template.

    Replace every method body with your own implementation while preserving the
    public API used by the official tests.

    The intended contract is:
    - one writer and one or more readers
    - shared state visible across processes
    - bounded storage with reusable space after readers advance
    - reads and writes report how many bytes are actually available
    """

    _NO_READER = -1

    def __init__(
        self,
        name: str,
        create: bool,
        size: int,
        num_readers: int,
        reader: int,
        cache_align: bool = False,
        cache_size: int = 64,
    ):
        """
        Open or create the shared buffer.

        Expected behavior:
        - validate constructor arguments
        - allocate or attach to shared memory
        - initialize any shared metadata needed to track writer and reader state
        - set up local views/fields used by the rest of the methods

        Parameters:
        - `name`: shared memory block name
        - `create`: `True` for the creator/owner, `False` to attach to an existing block
        - `size`: logical payload capacity in bytes
        - `num_readers`: number of reader slots to support
        - `reader`: reader index for this instance, or `_NO_READER` for the writer instance
        - `cache_align` / `cache_size`: optional metadata-layout knobs; you may ignore
          them internally as long as validation and behavior remain correct
        """
        #validation
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError("name must be a non-empty string")

        if size <= 0:
            raise ValueError("size must be positive")

        if num_readers <= 0:
            raise ValueError("num_readers must be positive")

        if reader < -1 or reader >= num_readers:
            raise ValueError("reader must be -1 or a valid reader index")

        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if cache_align and (cache_size & (cache_size - 1)) != 0:
            raise ValueError("cache_size must be a power of two when cache_align is True")
        #metadata
        dtype = np.dtype([
            ("write_pos",     np.uint64),
            ("reader_pos",    np.uint64, num_readers),
            ("reader_active", np.uint8,  num_readers),
])
        super().__init__(name=name, create=create, size=dtype.itemsize + size)
        self._meta = np.ndarray(1, dtype=dtype, buffer=self.buf)
        self._buf = self.buf[dtype.itemsize:]

        if create:
            self._meta["write_pos"][0] = 0
            self._meta["reader_pos"][0][:] = 0
            self._meta["reader_active"][0][:] = 0

       
        self._size = size
        self._num_readers = num_readers
        self._reader = reader
    @property
    def buffer_size(self) -> int:
        return self._size
    @property
    def num_readers(self) -> int:
        return self._num_readers

    def close(self) -> None:
        """
        Release local views and close this process's handle to the shared memory.

        This should not destroy the buffer for other attached processes.
        """
        try:
            super().close()
        except Exception:
            pass

    def __enter__(self) -> "SharedBuffer":
        """
        Enter the context manager.

        Reader instances are expected to mark themselves active while inside the
        context. Writer-only instances can simply return `self`.
        """
        if self._reader != self._NO_READER:
            self.set_reader_active(True)
    
        return self

    def __exit__(self, *_):
        """
        Exit the context manager.

        Reader instances are expected to mark themselves inactive on exit, then
        close local resources.
        """
        if self._reader != self._NO_READER:
            self.set_reader_active(False)
    
        self.close()

    def calculate_pressure(self) -> int:
        """
        Return current writer pressure as an integer percentage.

        Pressure is based on how much of the bounded storage is currently in use
        relative to the slowest active reader.
        """
        write_pos = self.get_write_pos()
        active = self._meta["reader_active"][0]
        reader_positions = self._meta["reader_pos"][0]
        active_positions = reader_positions[active.astype(bool)]
        if len(active_positions) == 0:
            return 0
        min_pos = int(active_positions.min())
        return int((write_pos - min_pos) / self._size * 100)

    def int_to_pos(self, value: int) -> int:
        """
        Convert an absolute position counter into a position inside the bounded payload area.

        If your design does not use modulo arithmetic internally, you may still
        keep this helper as the mapping from logical positions to buffer offsets.
        """
        return value % self._size

    def update_reader_pos(self, new_reader_pos: int) -> None:
        """
        Store this reader's absolute read position in shared state.

        This must fail clearly when called on a writer-only instance.
        """
        if self._reader == self._NO_READER:
            raise RuntimeError("cannot update reader pos on a writer instance")
        self._meta["reader_pos"][0][self._reader] = new_reader_pos

    def set_reader_active(self, active: bool) -> None:
        """
        Mark this reader as active or inactive in shared state.

        Active readers apply backpressure. Inactive readers should not reduce
        writer capacity.
        """
        if self._reader == self._NO_READER:
            raise RuntimeError("cannot set reader active on a writer instance")
        self._meta["reader_active"][0][self._reader] = active

    def is_reader_active(self) -> bool:
        """
        Return whether this reader is currently marked active.

        This must fail clearly when called on a writer-only instance.
        """
        if self._reader == self._NO_READER:
            raise RuntimeError("cannot check reader active on a writer instance")
        return bool(self._meta["reader_active"][0][self._reader])

    def update_write_pos(self, new_writer_pos: int) -> None:
        """
        Store the writer's absolute write position in shared state.

        The write position is what makes newly written bytes visible to readers.
        """
        self._meta["write_pos"][0] = new_writer_pos

    def inc_writer_pos(self, inc_amount: int) -> None:
        """
        Advance the writer's absolute position by `inc_amount` bytes.

        This is how a writer publishes bytes after copying them into the buffer.
        """
        self._meta["write_pos"][0] += inc_amount

    def inc_reader_pos(self, inc_amount: int) -> None:
        """
        Advance this reader's absolute position by `inc_amount` bytes.

        This is how a reader consumes bytes after reading them.
        """
        if self._reader == self._NO_READER:
            raise RuntimeError("cannot increment reader pos on a writer instance")
        self._meta["reader_pos"][0][self._reader] += inc_amount

    def get_write_pos(self) -> int:
        """
        Return the current absolute writer position.

        Readers can use this to resynchronize or compute how much data is available.
        """
        return int(self._meta["write_pos"][0])
        
    def compute_max_amount_writable(self, force_rescan: bool = False) -> int:
        """
        Return how many bytes the writer can safely expose right now.

        This should take active readers into account. `force_rescan=True` is used
        by the tests to ensure externally updated reader positions are observed.
        """
        write_pos = self.get_write_pos()
    
       
        active = self._meta["reader_active"][0]
        reader_positions = self._meta["reader_pos"][0]
        
        
        active_positions = reader_positions[active.astype(bool)]
        
       
        if len(active_positions) == 0:
            return self._size
        
        
        min_pos = int(active_positions.min())
        
        return max(0, self._size - (write_pos - min_pos))

    def jump_to_writer(self) -> None:
        """
        Move this reader directly to the current writer position.

        Use this when a reader has fallen too far behind and old unread data is
        no longer retained.
        """
        self.update_reader_pos(self.get_write_pos())

    def expose_writer_mem_view(self, size: int) -> RingView:
        """
        Return a writable view tuple for up to `size` bytes.

        The return shape is:
        - `mv1`: first writable view
        - `mv2`: optional second writable view if the exposed region is split
        - `actual_size`: how many bytes are actually writable right now
        - `split`: whether the writable region is split across two views

        If less than `size` bytes are currently writable, clamp to the amount
        available rather than raising.
        """
       
       
        actual_size = min(size, self.compute_max_amount_writable())
        start = self.int_to_pos(self.get_write_pos())
        if start + actual_size <= self._size:
            return self._buf[start:start + actual_size], None, actual_size, False
        else:
            first_chunk = self._size - start
            mv1 = self._buf[start:]
            mv2 = self._buf[:actual_size - first_chunk]
            return mv1, mv2, actual_size, True

    def expose_reader_mem_view(self, size: int) -> RingView:
        """
        Return a readable view tuple for up to `size` bytes.

        The shape matches `expose_writer_mem_view()`. If less than `size` bytes
        are currently readable, clamp to the amount available rather than raising.
        """
        
        if self._reader == self._NO_READER:
            raise RuntimeError("cannot call this on a writer instance")
        write_pos = self.get_write_pos()
        reader_pos = int(self._meta["reader_pos"][0][self._reader])
        
        # add this check
        if write_pos - reader_pos > self._size:
            self.jump_to_writer()
            return memoryview(self._buf)[0:0], None, 0, False
        
        available = write_pos - reader_pos
        actual_size = min(size, available)
        start = self.int_to_pos(reader_pos)
        if start + actual_size <= self._size:
            return memoryview(self._buf)[start:start + actual_size], None, actual_size, False
        else:
            first_chunk = self._size - start
            mv1 = memoryview(self._buf)[start:]
            mv2 = memoryview(self._buf)[:actual_size - first_chunk]
            return mv1, mv2, actual_size, True


    def simple_write(self, writer_mem_view: RingView, src: object) -> None:
        """
        Copy bytes from `src` into the exposed writer view(s).

        If `src` is larger than the destination region, copy only the prefix that fits.
        This helper should not publish data by itself; publishing happens when the
        writer position is advanced.
        """
        mv1, mv2, actual_size, split = writer_mem_view
        src_bytes = bytes(memoryview(src).cast('B'))
        if not split:
            mv1[:actual_size] = src_bytes[:actual_size]
        else:
            first_chunk = len(mv1)
            mv1[:] = src_bytes[:first_chunk]
            mv2[:] = src_bytes[first_chunk:actual_size]
 

    def simple_read(self, reader_mem_view: RingView, dst: object) -> None:
        """
        Copy bytes from the exposed reader view(s) into `dst`.

        If `dst` is smaller than the readable region, copy only the prefix that fits.
        This helper should not consume data by itself; consumption happens when the
        reader position is advanced.
        """
        mv1, mv2, actual_size, split = reader_mem_view
        dst_bytes = memoryview(dst).cast('B')
        copy_size = min(actual_size, len(dst_bytes))
        if not split:
            dst_bytes[:copy_size] = bytes(mv1[:copy_size])
        else:
            first_chunk = min(len(mv1), copy_size)
            dst_bytes[:first_chunk] = bytes(mv1[:first_chunk])
            remaining = copy_size - first_chunk
            if remaining > 0:
                dst_bytes[first_chunk:copy_size] = bytes(mv2[:remaining])

    def write_array(self, arr: np.ndarray) -> int:
        """
        Write a NumPy array's raw bytes into the shared buffer.

        Return the number of bytes written. If the full array does not fit, the
        contract used by the tests expects this method to return `0`.
        """
        nbytes = arr.nbytes
        
        
        if nbytes > self.compute_max_amount_writable():
            return 0
        
        
        view = self.expose_writer_mem_view(nbytes)
        self.simple_write(view, arr)
        
        
        self.inc_writer_pos(nbytes)
        return nbytes

    def read_array(self, nbytes: int, dtype: np.dtype) -> np.ndarray:
        """
        Read `nbytes` from the shared buffer and interpret them as `dtype`.

        Return a NumPy array view/copy of the requested bytes when enough data is
        available. If there are not enough readable bytes, return an empty array
        with the requested dtype.
        """
        write_pos = self.get_write_pos()
        reader_pos = int(self._meta["reader_pos"][0][self._reader])
        available = write_pos - reader_pos
        if available < nbytes:
            return np.array([], dtype=dtype)
        view = self.expose_reader_mem_view(nbytes)
        _, _, actual_size, _ = view
        buf = bytearray(actual_size)
        self.simple_read(view, buf)
        self.inc_reader_pos(actual_size)
        return np.frombuffer(buf, dtype=dtype)