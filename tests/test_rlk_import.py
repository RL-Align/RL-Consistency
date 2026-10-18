"""Synthetic sidecars exercise the pinned wire format, never model certification."""
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rl_consistency.cli import main
from rl_consistency.common import ContractError, read_json
from rl_consistency.rlk_import import SIDECAR_SCHEMA, inspect_sidecars, response_positions

HAS_TORCH = importlib.util.find_spec('torch') is not None


class LayoutTests(unittest.TestCase):
    def test_cp_partition_and_causal_shift_at_boundaries(self):
        # Independent ownership oracle: assign each response target's preceding
        # logit to a physical block, then map the block to its zigzag owner.
        for cp in (1, 2, 4, 8):
            for total in (3, 7, 16, 31):
                for prompt in range(1, total):
                    width = (total + 2*cp - 1) // (2*cp)
                    for rank in range(cp):
                        expected = []
                        for pos in range(total - prompt):
                            block = (prompt + pos - 1) // width
                            owner = min(block, 2*cp - 1 - block) if cp > 1 else 0
                            if owner == rank: expected.append(pos)
                        self.assertEqual(response_positions(total, total-prompt, cp, rank), expected)


@unittest.skipUnless(HAS_TORCH, 'optional torch CPU integration suite')
class SidecarTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'sidecars'; self.source.mkdir()

    def sidecars(self, tp=2, cp=2, world=4, calls=1, dtype=None):
        t = self.torch
        for call in range(calls):
            for rank in range(world):
                indices = response_positions(9, 7, cp, (rank//tp) % cp)
                full = t.tensor([-1., -2., -3., -4., -5., -6., -7.], dtype=dtype or t.float32)
                values = full[indices]
                payload = {'schema_version': SIDECAR_SCHEMA, 'rank': rank, 'call_index': call,
                           'tensor_parallel_size': tp, 'context_parallel_size': cp,
                           'train_log_probs': [values.clone()], 'rollout_log_probs': [values.clone()],
                           'loss_masks': [t.tensor([1, 0, 1, 1, 1, 0, 1])],
                           'total_lengths': [9], 'response_lengths': [7]}
                t.save(payload, self.source/f'r{rank}-c{call}.pt')

    def mutate(self, name, callback):
        path = self.source/name
        payload = self.torch.load(path, weights_only=True)
        callback(payload)
        self.torch.save(payload, path)

    def inspect(self, **kwargs):
        return inspect_sidecars(self.source, tp=kwargs.get('tp', 2), cp=kwargs.get('cp', 2),
                                world_size=kwargs.get('world', 4), layout=kwargs.get('layout', 'vime-zigzag'))

    def test_real_pt_roundtrip_cp_tp_dp_and_no_certification(self):
        self.sidecars(world=8, calls=2)
        result = self.inspect(world=8)
        self.assertTrue(result['comparison']['passed'])
        self.assertEqual(result['comparison']['elements'], 20)
        self.assertEqual(result['sample_count'], 4)
        self.assertEqual([r['position'] for r in result['training_records'][:5]], [2, 4, 5, 6, 8])
        self.assertFalse(result['gate_eligible']); self.assertFalse(result['certified'])
        self.assertEqual(len(result['source_artifacts']), 16)

    def test_dtypes_and_signed_zero_mismatch_are_preserved(self):
        for dtype in (self.torch.float16, self.torch.bfloat16, self.torch.float32, self.torch.float64):
            self.sidecars(tp=1, cp=1, world=1, dtype=dtype)
            self.mutate('r0-c0.pt', lambda p: p['train_log_probs'][0].__setitem__(0, 0.))
            self.mutate('r0-c0.pt', lambda p: p['rollout_log_probs'][0].__setitem__(0, -0.))
            result = self.inspect(tp=1, cp=1, world=1)
            self.assertFalse(result['comparison']['passed'])
            self.assertEqual(result['comparison']['mismatch_elements'], 1)
            self.assertEqual(result['comparison']['max_abs'], 0.)

    def test_missing_rank_call_and_duplicate_fail(self):
        self.sidecars(calls=2)
        (self.source/'r1-c1.pt').unlink()
        with self.assertRaisesRegex(ContractError, 'missing rank'): self.inspect()
        self.sidecars(calls=2)
        (self.source/'duplicate.pt').write_bytes((self.source/'r0-c0.pt').read_bytes())
        with self.assertRaisesRegex(ContractError, 'duplicate rank'): self.inspect()

    def test_allgather_and_bad_topology_are_rejected(self):
        with self.assertRaisesRegex(ContractError, 'allgather'): self.inspect(layout='allgather')
        with self.assertRaisesRegex(ContractError, 'divisible'): self.inspect(world=3)

    def test_tp_replica_difference_fails_even_when_numeric_equal(self):
        self.sidecars()
        self.mutate('r0-c0.pt', lambda p: p['train_log_probs'][0].__setitem__(0, 0.))
        self.mutate('r1-c0.pt', lambda p: p['train_log_probs'][0].__setitem__(0, -0.))
        with self.assertRaisesRegex(ContractError, 'TP replicas'): self.inspect()

    def test_cp_mask_mismatch_and_bad_shard_length_fail(self):
        self.sidecars()
        for rank in (2, 3):
            self.mutate(f'r{rank}-c0.pt', lambda p: p['loss_masks'][0].__setitem__(0, 0))
        with self.assertRaisesRegex(ContractError, 'across CP'): self.inspect()
        self.sidecars()
        self.mutate('r0-c0.pt', lambda p: p['train_log_probs'].__setitem__(0, self.torch.ones(3)))
        with self.assertRaisesRegex(ContractError, 'shard length'): self.inspect()

    def test_nonfinite_active_values_and_nonbinary_mask_rejected(self):
        self.sidecars(tp=1, cp=1, world=1)
        self.mutate('r0-c0.pt', lambda p: p['train_log_probs'][0].__setitem__(0, float('nan')))
        with self.assertRaisesRegex(ContractError, 'non-finite'): self.inspect(tp=1, cp=1, world=1)
        self.sidecars(tp=1, cp=1, world=1)
        self.mutate('r0-c0.pt', lambda p: p['loss_masks'][0].__setitem__(0, 2))
        with self.assertRaisesRegex(ContractError, 'binary'): self.inspect(tp=1, cp=1, world=1)

    def test_safe_loader_no_unsafe_fallback(self):
        self.sidecars()
        with patch.object(self.torch, 'load', side_effect=RuntimeError('untrusted object')) as load:
            with self.assertRaisesRegex(ContractError, 'weights_only=True'): self.inspect()
        self.assertEqual(load.call_count, 1)
        self.assertEqual(load.call_args.kwargs, {'map_location': 'cpu', 'weights_only': True})

    def test_symlink_escape_rejected(self):
        self.sidecars()
        outside = self.root/'outside.pt'; outside.write_bytes((self.source/'r0-c0.pt').read_bytes())
        (self.source/'r0-c0.pt').unlink(); (self.source/'r0-c0.pt').symlink_to(outside)
        with self.assertRaisesRegex(ContractError, 'escapes'): self.inspect()

    def test_cli_mismatch_publishes_raw_diagnostic_and_returns_one(self):
        self.sidecars(tp=1, cp=1, world=1)
        self.mutate('r0-c0.pt', lambda p: p['train_log_probs'][0].__setitem__(0, -2.))
        output = self.root/'report.json'
        args = ['import-rlk', '--source', str(self.source), '--output', str(output),
                '--tp', '1', '--cp', '1', '--world-size', '1', '--layout', 'vime-zigzag']
        with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(main(args), 1)
        report = read_json(output)
        self.assertIn('training_records', report)
        self.assertFalse(report['gate_eligible'])
        from rl_consistency.compare import validate_trace
        with self.assertRaisesRegex(ContractError, 'unsupported trace schema'): validate_trace(report)
        with contextlib.redirect_stderr(io.StringIO()): self.assertEqual(main(args), 2)
