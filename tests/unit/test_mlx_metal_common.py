import numpy as np

from tests.helpers.util import requires_mlx


@requires_mlx
def test_chunked_scan_trims_to_exact_length():
    """The padded-concat optimization must not change the output contract.

    ``chunked_scan`` concatenates the kernels' *padded* chunks so the allocation
    lands on a multiple-of-chunk_size grid rather than at ``n_samples`` (fewer
    MLX buffer size classes). The single trim at the end is what keeps the
    result exact, and callers reshape to ``n_samples`` immediately afterwards --
    so an off-by-one here surfaces as a reshape error, not as wrong data.
    """
    import mlx.core as mx

    from ezmsg.sigproc.util.mlx_metal_common import chunked_scan

    def launch(x_chunk, state, cs, valid_length):
        # A stand-in kernel with the real ones' contract: output is the FULL
        # chunk width, state carries forward.
        assert x_chunk.shape[-1] == cs
        return x_chunk * 2.0, state + 1

    n_channels = 3
    for chunk_sizes in ((32,), (32, 128), (16, 64, 256)):
        for n_samples in (1, 15, 16, 17, 31, 32, 33, 100, 128, 129, 257, 1000):
            x = mx.array(np.arange(n_channels * n_samples, dtype=np.float32).reshape(n_channels, n_samples))
            y, state = chunked_scan(x, n_samples, chunk_sizes, mx.array(0), launch)
            assert y.shape == (n_channels, n_samples), f"chunk_sizes={chunk_sizes} n_samples={n_samples} gave {y.shape}"
            # Padding must never leak into the result.
            np.testing.assert_allclose(np.asarray(y), np.asarray(x) * 2.0)


@requires_mlx
def test_chunked_scan_pads_only_the_final_chunk():
    """Trimming once is sound only while padding is confined to the last chunk.

    Padding an interior chunk would put padding *between* two runs of valid
    samples and the single trim would return it as data. The invariant holds
    structurally -- padding needs ``remaining < chunk_size``, which makes that
    chunk the tail -- so this pins the property across size sets and lengths
    rather than trying to construct a violation (there isn't one to construct;
    the assertion in ``chunked_scan`` guards future edits to the selection rule).
    """
    import mlx.core as mx

    from ezmsg.sigproc.util.mlx_metal_common import chunked_scan

    for chunk_sizes in ((32,), (32, 128), (16, 64, 256), (128, 32)):
        for n_samples in (1, 15, 31, 32, 33, 100, 128, 129, 257, 1000):
            seen = []

            def launch(x_chunk, state, cs, valid_length, _seen=seen):
                _seen.append((int(np.asarray(valid_length)[0]), cs))
                return x_chunk, state

            chunked_scan(mx.zeros((2, n_samples)), n_samples, chunk_sizes, mx.array(0), launch)
            padded = [i for i, (valid, cs) in enumerate(seen) if valid < cs]
            assert padded in ([], [len(seen) - 1]), (
                f"chunk_sizes={chunk_sizes} n_samples={n_samples}: padded chunks at {padded} of {len(seen)}"
            )


@requires_mlx
def test_chunked_scan_allocates_fewer_size_classes_than_per_chunk_trim():
    """The point of the change, measured against the implementation it replaced.

    Compares like with like: identical inputs, held live so their own size
    classes are not what is being counted, so the delta is the scan's
    intermediates alone. A MiB figure would be machine-dependent; the ratio is
    the property worth pinning.
    """
    import mlx.core as mx

    from ezmsg.sigproc.util.mlx_metal_common import chunked_scan

    chunk_sizes = (32, 1024)

    def launch(x_chunk, state, cs, valid_length):
        return x_chunk * 2.0, state

    def per_chunk_trim(x_flat, n_samples, sizes, state, launch_fn):
        """The pre-change behavior: trim every chunk, concatenate at n_samples."""
        y_chunks, start = [], 0
        while start < n_samples:
            remaining = n_samples - start
            cs = next((s for s in sizes if s >= remaining), sizes[-1])
            valid = min(remaining, cs)
            x_chunk = x_flat[:, start : start + valid]
            if valid < cs:
                x_chunk = mx.pad(x_chunk, [(0, 0), (0, cs - valid)])
            y, state = launch_fn(x_chunk, state, cs, mx.array([valid], dtype=mx.uint32))
            y_chunks.append(y[:, :valid])
            start += valid
        y = y_chunks[0] if len(y_chunks) == 1 else mx.concatenate(y_chunks, axis=-1)
        return y, state

    lengths = [300 * k for k in range(1, 21)] * 3
    inputs = [mx.zeros((256, n)) for n in lengths]  # held live throughout
    mx.eval(*inputs)

    def cache_for(scan):
        mx.clear_cache()
        for x, n in zip(inputs, lengths):
            y, _ = scan(x, n, chunk_sizes, mx.array(0), launch)
            mx.eval(y)
        used = mx.get_cache_memory()
        mx.clear_cache()
        return used

    before = cache_for(per_chunk_trim)
    after = cache_for(chunked_scan)
    assert after < before, f"padded concat cached more, not less: {after} vs {before} bytes"
