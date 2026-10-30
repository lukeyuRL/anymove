"""Pure gradient accumulation scheduling for small VRAM training."""


def optimizer_step_count(n_microbatches, grad_accum):
    if type(n_microbatches) is not int or n_microbatches < 1:
        raise ValueError("n_microbatches must be >= 1")
    if type(grad_accum) is not int or grad_accum < 1:
        raise ValueError("grad_accum must be >= 1")
    return (n_microbatches + grad_accum - 1) // grad_accum


def accumulation_group_size(index, n_microbatches, grad_accum):
    """Loss divisor for each microbatch including the shorter last group."""
    optimizer_step_count(n_microbatches, grad_accum)
    if type(index) is not int or not 0 <= index < n_microbatches:
        raise ValueError("microbatch index out of range")
    start = (index // grad_accum) * grad_accum
    return min(grad_accum, n_microbatches - start)
