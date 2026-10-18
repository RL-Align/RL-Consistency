import copy
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path
from rl_consistency.capture import TraceWriter, token_records, worker_manifest
from rl_consistency.common import ContractError, file_hash, read_json
from rl_consistency.compare import compare_traces

HAS_TORCH = importlib.util.find_spec('torch') is not None


def writer(keys=None, **changes):
    manifest = worker_manifest(role='training', rank=0, world_size=1,
                               module_names=['rl_consistency.capture'], mapped_libraries=False)
    args = dict(producer='training', weight_version='test-only', weights_sha256='1'*64,
                inputs_sha256='2'*64, semantics_sha256='3'*64, implementation_manifest=manifest,
                independent=True, expected_keys=keys or [('s', 4, 'logp')])
    args.update(changes)
    return TraceWriter(**args)


def row(position=4, value=-1.):
    return {'sample': 's', 'position': position, 'node': 'logp', 'shape': [],
            'hex': struct.pack('<f', value).hex()}


class CaptureTests(unittest.TestCase):
    def test_incomplete_duplicate_extra_and_mixed_dtype_fail(self):
        with tempfile.TemporaryDirectory() as d:
            w = writer([('s', 4, 'logp'), ('s', 5, 'logp')])
            w.add_records('float32', [row()])
            for rows in ([row()], [row(6)]):
                with self.assertRaises(ContractError): w.add_records('float32', rows)
            with self.assertRaises(ContractError): w.finish(Path(d)/'trace.json')
            self.assertEqual(list(Path(d).iterdir()), [])
            half = row(5); half['hex'] = '00bc'
            with self.assertRaises(ContractError): w.add_records('float16', [half])
            w.add_records('float32', [row(5)])
            w.finish(Path(d)/'trace.json')
            self.assertEqual(len(read_json(Path(d)/'trace.json')['records']), 2)
            with self.assertRaises(ContractError): w.add_records('float32', [row()])

    def test_nonfinite_and_reuse_rejected(self):
        for n in (float('inf'), float('nan')):
            with self.assertRaises(ContractError): writer().add_records('float32', [row(value=n)])
        with self.assertRaises(ContractError): writer(independent=False)
        with self.assertRaises(ContractError): writer(weights_sha256='unknown')

    def test_published_output_cannot_be_overwritten_and_input_is_copied(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'trace.json'
            w = writer(); records = [row()]; w.add_records('float32', records)
            records[0]['hex'] = '00000000'
            w.finish(path); before = file_hash(path)
            w = writer(); w.add_records('float32', [row(value=2)])
            with self.assertRaises(FileExistsError): w.finish(path)
            self.assertEqual(file_hash(path), before)
            self.assertEqual(list(Path(d).iterdir()), [path])
            self.assertEqual(read_json(path)['records'][0], row())

    def test_manifest_captures_loaded_file_without_importing(self):
        m = worker_manifest(role='rollout', rank=0, world_size=1,
                            module_names=['rl_consistency.capture'], mapped_libraries=False)
        artifact = m['loaded_artifacts'][0]
        self.assertEqual(artifact['sha256'], file_hash(artifact['path']))
        with self.assertRaises(ContractError):
            worker_manifest(role='rollout', rank=0, world_size=1, module_names=['does.not.exist'])

    @unittest.skipUnless(Path('/proc/self/maps').exists(), 'Linux mapped libraries')
    def test_manifest_collects_mapped_libraries(self):
        m = worker_manifest(role='training', rank=0, world_size=1, module_names=['rl_consistency.capture'])
        self.assertTrue(any(x['kind'] == 'mapped_library' for x in m['loaded_artifacts']))

    def test_pid_and_rank_do_not_make_an_independent_implementation(self):
        a = writer()
        manifest = copy.deepcopy(a.metadata['capture']['worker_manifest'])
        manifest.update(pid=9999, rank=1, world_size=2)
        for artifact in manifest['loaded_artifacts']: artifact['rank'] = 1
        b = writer(implementation_manifest=manifest)
        self.assertEqual(a.metadata['implementation_sha256'], b.metadata['implementation_sha256'])


@unittest.skipUnless(HAS_TORCH, 'optional torch CPU integration suite')
class TensorCaptureTests(unittest.TestCase):
    def test_dtypes_noncontiguous_mask_and_signed_zero(self):
        import torch
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            tensor = torch.tensor([[0., 99.], [-0., 99.], [float('nan'), 99.]], dtype=dtype)[:, 0]
            name, records = token_records('s', [4, 7, 9], 'logp', tensor,
                                         active_mask=torch.tensor([True, True, False]))
            self.assertEqual([r['position'] for r in records], [4, 7])
            self.assertNotEqual(records[0]['hex'], records[1]['hex'])
            self.assertEqual(name, str(dtype).removeprefix('torch.'))
        with self.assertRaises(ContractError): token_records('s', [0, 1, 2], 'logp', tensor)
        with self.assertRaises(ContractError):
            token_records('s', [0, 1, 2], 'logp', tensor, active_mask=torch.ones(3))

    def test_capture_preserves_autograd_and_trace_compares(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            tensor = torch.tensor([-1., -2.], requires_grad=True)
            keys = [('s', 4, 'logp'), ('s', 5, 'logp')]
            a = writer(keys); a.add_tokens('s', [4, 5], 'logp', tensor); a.finish(Path(d)/'a.json')
            tensor.square().sum().backward()
            self.assertEqual(tensor.grad.tolist(), [-2., -4.])
            b = writer(keys); b.add_tokens('s', [5, 4], 'logp', tensor.flip(0)); b.finish(Path(d)/'b.json')
            self.assertTrue(compare_traces(read_json(Path(d)/'a.json'), read_json(Path(d)/'b.json'))['passed'])
