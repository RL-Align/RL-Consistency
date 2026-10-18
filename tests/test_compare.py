import copy
import struct
import unittest
from rl_consistency.common import ContractError
from rl_consistency.compare import compare_traces
from helpers import recipe, trace


class ComparisonTests(unittest.TestCase):
    def setUp(self): self.r = recipe()

    def test_supported_dtypes(self):
        for dtype in ('float16', 'bfloat16', 'float32', 'float64'):
            with self.subTest(dtype=dtype):
                self.assertTrue(compare_traces(trace(self.r, dtype=dtype), trace(self.r, dtype=dtype))['passed'])

    def test_signed_zero_is_bitwise_mismatch(self):
        a, b = trace(self.r, value=0.0), trace(self.r, value=-0.0)
        result = compare_traces(a, b)
        self.assertFalse(result['passed']); self.assertEqual(result['max_abs'], 0.0)
        self.assertTrue(compare_traces(a, b, accuracy=True)['passed'])

    def test_nonfinite_fails_even_if_identical(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ContractError): compare_traces(trace(self.r,value=value), trace(self.r,value=value))

    def test_empty_and_duplicate_fail(self):
        for records in ([], trace(self.r)['records'] * 2):
            t = trace(self.r); t['records'] = records
            with self.assertRaises(ContractError): compare_traces(t, t)

    def test_shape_and_dtype_fail(self):
        a, b = trace(self.r), trace(self.r)
        b['records'][0]['shape'] = [2]
        with self.assertRaises(ContractError): compare_traces(a,b)
        with self.assertRaises(ContractError): compare_traces(a,trace(self.r,dtype='float64'))

    def test_logical_order_not_physical_order(self):
        a = trace(self.r)
        row = copy.deepcopy(a['records'][0]); row['sample'] = 't'
        a['records'].append(row)
        b = copy.deepcopy(a); b['records'].reverse()
        self.assertTrue(compare_traces(a,b)['passed'])
        b['records'].pop()
        with self.assertRaises(ContractError): compare_traces(a,b)

    def test_stale_weights_and_reuse_fail(self):
        for key, value in [('weight_version','1'),('weights_sha256','f'*64),('independent',False)]:
            b = trace(self.r); b[key] = value
            with self.assertRaises(ContractError): compare_traces(trace(self.r),b)

    def test_independent_accuracy_accepts_higher_precision(self):
        result = compare_traces(trace(self.r, value=1.0), trace(self.r, value=1.00000001,dtype='float64'),
                                accuracy=True, atol=1e-6)
        self.assertTrue(result['passed'])
        self.assertFalse(compare_traces(trace(self.r, value=1.0),trace(self.r,value=2.0),accuracy=True,atol=1e-6)['passed'])
