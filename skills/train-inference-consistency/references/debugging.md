# First-difference debugging

1. Verify the actual loaded code/binary, checkpoint version, input tokens, masks, positions, topology, score definition and runtime routes.
2. Compare logically aligned final logprobs as raw bytes, preserving the raw failing sample.
3. Capture corresponding layer boundaries. Use the model's declared topological execution order to locate the first difference; lexicographic layer-name order is not execution order.
4. Replay identical operator inputs. Check dtype, quantization, block geometry, intra/inter-rank reduction, FMA/rounding, and KV/router metadata before changing a kernel.
5. Compare fixed-order and independent high-precision references separately. If both tested implementations agree but accuracy fails, repair the shared numerical problem.
6. Preserve a regression fixture, rerun affected gates, and report remaining model-level tests.

`compare` reports the first mismatch in sorted logical-key order, not the earliest numerical cause. Use it as a pointer; map nodes back to the model execution graph.

For communication, verify both endpoints agree on shape, dtype, byte count, offsets, logical ordering and sequence number. Preserve payload bytes; do not insert an unordered in-transit floating-point reduction. Exercise uneven loads, zero counts and out-of-order arrivals when those cases are in scope.
