import json
import tempfile
import unittest
from pathlib import Path
from rl_consistency.common import ContractError, read_json, write_json
from rl_consistency.workflow import initialize, record_gate, status, review
from rl_consistency.report import render
from helpers import recipe, dump_fixture


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name); self.root = self.base/'run'; self.r = recipe()
        initialize(self.root,self.r)

    def gate(self,gate):
        return record_gate(self.root,gate,dump_fixture(self.base/gate,self.r,gate))

    def test_missing_dependency_fails(self):
        with self.assertRaises(ContractError): self.gate('C6')

    def test_repeated_prerequisite_invalidates_descendants(self):
        self.gate('C0'); self.gate('C2'); self.gate('C3')
        self.gate('C2')
        self.assertEqual(status(self.root)['gates']['C3']['status'],'invalidated')

    def test_snapshot_mutation_invalidates(self):
        result = self.gate('C0')
        path = self.root/'attempts'/result['id']/'evidence.json'
        path.write_text('{}')
        self.assertEqual(status(self.root)['gates']['C0']['status'],'invalidated')

    def test_original_mutation_does_not_modify_snapshot(self):
        self.gate('C0'); (self.base/'C0'/'runtime.json').write_text('{}')
        self.assertEqual(status(self.root)['gates']['C0']['status'],'passed')

    def test_recipe_change_is_rejected(self):
        r = read_json(self.root/'recipe.json'); r['accuracy']['atol'] = 1
        write_json(self.root/'recipe.json',r)
        with self.assertRaises(ContractError): status(self.root)

    def test_failed_retry_invalidates(self):
        self.gate('C0'); self.gate('C2')
        path = dump_fixture(self.base/'bad', self.r,'C0')
        value = read_json(path); value['execution'] = 'synthetic'; write_json(path,value)
        self.assertFalse(record_gate(self.root,'C0',path)['passed'])
        self.assertEqual(status(self.root)['gates']['C2']['status'],'invalidated')

    def test_no_path_escape(self):
        path = dump_fixture(self.base/'escape',self.r,'C0')
        value = read_json(path); value['runtime'] = '../outside.json'; write_json(path,value)
        result = record_gate(self.root,'C0',path)
        self.assertFalse(result['passed']); self.assertIn('escapes',result['error'])

    def test_review_prerequisites(self):
        with self.assertRaises(ContractError): review(self.root,'release','test','synthetic test decision')
        self.gate('C0'); review(self.root,'precision','test','synthetic test decision')
        self.assertTrue(status(self.root)['reviews'][0]['valid'])
        self.gate('C0')
        self.assertFalse(status(self.root)['reviews'][0]['valid'])

    def test_entire_synthetic_workflow_and_report(self):
        self.gate('C0'); self.gate('C1')
        review(self.root,'baseline','test','synthetic fixture; not a real approval')
        review(self.root,'precision','test','synthetic fixture; not a real approval')
        for i in range(2,11): self.assertTrue(self.gate(f'C{i}')['passed'])
        self.assertFalse(status(self.root)['release_eligible'])
        review(self.root,'release','test','synthetic fixture; not a real approval')
        self.assertTrue(status(self.root)['release_eligible'])
        out = self.base/'report.html'; render(self.root,out)
        self.assertIn('RL-Consistency',out.read_text())

    def test_nonexistent_run_is_not_created_by_read(self):
        with self.assertRaises(FileNotFoundError): status(self.base/'missing')
        self.assertFalse((self.base/'missing').exists())
