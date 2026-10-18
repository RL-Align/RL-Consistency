import copy
import unittest
from rl_consistency.common import ContractError, digest
from rl_consistency.gates import evaluate, DEPENDENCIES
from helpers import recipe, fixture, benchmark


class GateTests(unittest.TestCase):
    def setUp(self): self.r = recipe()
    def run_gate(self, gate, mutate=None):
        e, a = fixture(self.r, gate)
        if mutate: mutate(e, a)
        return evaluate(gate,e,self.r,a.__getitem__,benchmark(self.r))

    def test_all_gate_shapes(self):
        for gate in DEPENDENCIES:
            with self.subTest(gate=gate): self.assertTrue(self.run_gate(gate)['passed'])

    def test_no_synthetic_acceptance(self):
        with self.assertRaises(ContractError):
            self.run_gate('C0', lambda e,a: e.update(execution='synthetic'))

    def test_missing_required_case_fails(self):
        with self.assertRaises(ContractError):
            self.run_gate('C3', lambda e,a: e['pairs'].pop())

    def test_no_self_comparison(self):
        with self.assertRaises(ContractError):
            self.run_gate('C6', lambda e,a: e['pairs'][0].update(right=e['pairs'][0]['left']))

    def test_bitwise_mismatch_is_failed_verdict(self):
        result = self.run_gate('C6', lambda e,a: a['train-rollout-right.json']['records'][0].update(hex='00000000'))
        self.assertFalse(result['passed'])

    def test_no_fallback(self):
        with self.assertRaises(ContractError):
            self.run_gate('C5', lambda e,a: a['bindings.json'][0].update(fallback=True))

    def test_missing_binding_fails(self):
        with self.assertRaises(ContractError):
            self.run_gate('C5', lambda e,a: a['bindings.json'].pop())

    def test_same_implementation_reference_fails(self):
        with self.assertRaises(ContractError):
            self.run_gate('C7', lambda e,a: a['gradient-accuracy-right.json'].update(implementation_sha256='a'*64))

    def test_two_updates_required(self):
        with self.assertRaises(ContractError):
            self.run_gate('C8', lambda e,a: a['updates.json'].pop())

    def test_changed_weights_required(self):
        with self.assertRaises(ContractError):
            self.run_gate('C8', lambda e,a: a['updates.json'][0].update(after_sha256='1'*64))

    def test_resume_state_and_first_update_checked(self):
        for key, value in [('resumed_first_step',False),('resumed_after_update_sha256','f'*64)]:
            with self.subTest(key=key), self.assertRaises(ContractError):
                self.run_gate('C9', lambda e,a: a['checkpoint.json'].update({key:value}))
        with self.assertRaises(ContractError):
            self.run_gate('C9', lambda e,a: a['checkpoint.json']['restored_state'].update(rng='f'*64))

    def test_full_model_cannot_be_shrunk(self):
        with self.assertRaises(ContractError):
            self.run_gate('C9', lambda e,a: a['full.json'].update(layers=4))

    def test_reduced_model_preserves_dimensions(self):
        def modify(e,a):
            a['coverage.json']['reduced_layers'] = copy.deepcopy(a['coverage.json']['reduced_layers'])
            a['coverage.json']['reduced_layers'][0]['dimensions']['hidden'] = 128
        with self.assertRaises(ContractError): self.run_gate('C2',modify)

    def test_performance_requires_equal_work(self):
        with self.assertRaises(ContractError):
            self.run_gate('C10',lambda e,a: a['benchmark.json'].update(total_output_tokens=1))

    def test_resume_trace_must_match_restored_parameters(self):
        with self.assertRaises(ContractError):
            self.run_gate('C9', lambda e,a: a['checkpoint.json']['restored_state'].update(model='d'*64))

    def test_performance_requires_equal_sampling(self):
        with self.assertRaises(ContractError):
            self.run_gate('C10', lambda e,a: a['benchmark.json'].update(sampling={}))
