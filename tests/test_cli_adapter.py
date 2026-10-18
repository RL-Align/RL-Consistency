import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rl_consistency.adapters import plan
from rl_consistency.cli import main
from rl_consistency.common import ContractError, read_json
from rl_consistency.contracts import validate_recipe
from helpers import recipe


class CLITests(unittest.TestCase):
    def test_list_and_schema(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out): self.assertEqual(main(['list']),0)
        self.assertEqual(len(json.loads(out.getvalue())),6)
        self.assertTrue(all(not x['certified'] for x in json.loads(out.getvalue())))
        validate_recipe(recipe(),bound=True)

    def test_invalid_parameters_fail(self):
        for value in (True,float('nan'),0):
            r=recipe(); r['inputs']['max_context_tokens']=value
            with self.assertRaises(ContractError): validate_recipe(r)

    def test_cuda_rocm_plans_use_real_common_interface(self):
        for backend in ('cuda','rocm'):
            value = plan(recipe(backend),'/repo with spaces','/profile with spaces.json','/venv/bin/python')
            self.assertIn('run',value['argv']); self.assertNotIn('verify',value['argv'])
            self.assertIn('--profile',value['argv']); self.assertEqual(value['argv'][0],'/venv/bin/python')
            self.assertFalse(value['certified'])

    def test_plan_is_read_only(self):
        with patch('subprocess.run',side_effect=AssertionError('must not execute')):
            plan(recipe(),'/repo','/profile.json','/python')

    def test_rocm_unvalidated_configs_rejected(self):
        r=recipe('rocm'); r['target']['rollout']['cp']=2
        with self.assertRaises(ContractError): plan(r,'/repo','/profile','/python')
        r=recipe('rocm'); r['sampling']['temperature']=0
        with self.assertRaises(ContractError): plan(r,'/repo','/profile','/python')

    def test_model_and_version_extension_requires_adapter(self):
        for section,key,value in [('model','id','some/new-model'),('rl_kernel','commit','a'*40)]:
            r=recipe(); r[section][key]=value
            with self.assertRaises(ContractError): plan(r,'/repo','/profile','/python')

    def test_strict_json_rejects_ambiguous_data(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.json'
            for data in ('{"x":1,"x":2}', '{"x":NaN}'):
                p.write_text(data)
                with self.assertRaises(ContractError): read_json(p)

    def test_cli_error_exit(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['check','--recipe','/missing.json']),2)
