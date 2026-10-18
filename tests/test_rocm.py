"""Runner integration tests use synthetic files and never launch GPU processes."""
import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rl_consistency import rocm
from rl_consistency.cli import main
from rl_consistency.common import ContractError, read_json, write_json
from rl_consistency.rlk_import import SIDECAR_SCHEMA, response_positions

HAS_TORCH = importlib.util.find_spec('torch') is not None


class RunnerFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = self.root/'machine.json'
        template = Path(__file__).resolve().parents[1]/'configs/rocm-mi300x.example.json'
        self.config = read_json(template)
        write_json(self.profile,self.config)


class PlanTests(RunnerFixture):
    def test_plan_is_readonly_and_freezes_only_diagnostic_changes(self):
        self.config['defaults'].update(rollouts=200, lr=5e-7, **{'weight-decay':0.1,'global-batch-size':128,'rollout-batch-size':16})
        write_json(self.profile,self.config)
        with patch('subprocess.run',side_effect=AssertionError('plan must not run')):
            result = rocm.plan(self.profile,self.root/'result')
        defaults = result['resolved_profile']['defaults']
        self.assertEqual((defaults['steps'], defaults['lr'], defaults['weight-decay']),(1,0.,0.))
        self.assertEqual(defaults['global-batch-size'],128)
        self.assertEqual(defaults['max-response-len'],7168)
        self.assertEqual(defaults['temperature'],1.0)
        self.assertEqual(result['expected_samples'],128)
        self.assertFalse((self.root/'result').exists())

    def test_profile_errors_fail_before_execution(self):
        for key,value in [('global-batch-size',128),('top-k',8),('cp',1),('temperature',0),('unknown',1)]:
            config = json.loads(json.dumps(self.config)); config['defaults'][key]=value
            write_json(self.profile,config)
            with self.assertRaises(ContractError): rocm.plan(self.profile,self.root/'result')

    def test_preflight_failure_never_launches_and_retains_report(self):
        with patch.object(rocm,'preflight',side_effect=ContractError('wrong GPU')), \
             patch('subprocess.run',side_effect=AssertionError('must not launch')):
            result = rocm.run(self.profile,self.root/'result')
        self.assertEqual(result['verdict'],'inconclusive')
        self.assertIn('wrong GPU',result['errors'])
        self.assertTrue((self.root/'result'/'report.html').is_file())

    def test_output_cannot_be_reused(self):
        output = self.root/'result'; output.mkdir()
        with self.assertRaises(FileExistsError): rocm.run(self.profile,output)

    def test_environment_cannot_override_frozen_profile(self):
        execution = rocm.plan(self.profile,self.root/'result')
        with patch.dict('os.environ',{'RLK_REPRO_PROFILE':'/other','RLK_REPRO_MODEL_ROOT':'/other'}):
            env = rocm._environment(execution)
        self.assertNotIn('RLK_REPRO_PROFILE',env); self.assertNotIn('RLK_REPRO_MODEL_ROOT',env)


@unittest.skipUnless(HAS_TORCH,'optional PyTorch CPU suite')
class CollectTests(RunnerFixture):
    def fixture(self, *, mismatch=False, launcher_code=0):
        import torch
        root = self.root/'result'; root.mkdir()
        execution = rocm.plan(self.profile,root)
        write_json(root/'plan.json',execution)
        write_json(root/'profile.json',execution['resolved_profile'])
        write_json(root/'preflight.json',{'synthetic_test':True})
        write_json(root/'launcher-result.json',{'exit_code':launcher_code})
        (root/'launcher.log').write_text('synthetic test only\n')
        arm = root/'upstream/arms/r-r'; (arm/'mismatch_sidecars').mkdir(parents=True)
        launch = {'case_id':'R/R','framework_consistency':{'use_rollout_logprobs':False,'mismatch_metrics_recompute':True},
                  'frozen_input_fingerprint':'a'*64, 'environment':{'RLK_ABLATION_LR':'0.0','RLK_ABLATION_WEIGHT_DECAY':'0.0'}}
        write_json(arm/'launch.json',launch)
        summary = {'passed':not mismatch,'case_id':'R/R','num_rollout':1,'frozen_sources_match':True,
                   'launcher_returncode':launcher_code,'expected_sample_count':8}
        write_json(arm/'validation.json',summary)
        write_json(root/'upstream/single-arm-summary.json',summary)
        for name in ('before','after'): write_json(root/f'upstream/frozen-inputs.{name}.json',{'fingerprint':'a'*64})
        for rank in range(8):
            indices = response_positions(8,4,2,rank//4)
            values = torch.tensor([-1.,-2.,-3.,-4.])[indices]
            train = values.clone()
            if mismatch and indices: train[0] -= .125
            payload = {'schema_version':SIDECAR_SCHEMA,'rank':rank,'call_index':0,
                'tensor_parallel_size':4,'context_parallel_size':2,
                'train_log_probs':[train]*8,'rollout_log_probs':[values]*8,'loss_masks':[torch.ones(4)]*8,
                'total_lengths':[8]*8,'response_lengths':[4]*8}
            torch.save(payload,arm/f'mismatch_sidecars/rank{rank}.pt')
        return root

    def test_completed_raw_comparison_and_report(self):
        root = self.fixture()
        result = rocm.collect(root)
        self.assertEqual(result['verdict'],'pass')
        self.assertEqual(result['comparison']['elements'],32)
        self.assertFalse(result['certified'])
        self.assertIn('native old-logprob path',result['not_verified'])
        self.assertIn('not run',(root/'report.html').read_text())
        self.assertEqual(rocm.collect(root)['verdict'],'pass')

    def test_mismatch_is_failure_even_if_launcher_failed(self):
        result = rocm.collect(self.fixture(mismatch=True,launcher_code=1))
        self.assertEqual(result['verdict'],'fail')
        self.assertGreater(result['comparison']['mismatch_elements'],0)

    def test_launcher_failure_with_equal_values_is_inconclusive(self):
        self.assertEqual(rocm.collect(self.fixture(launcher_code=1))['verdict'],'inconclusive')

    def test_missing_rank_or_samples_not_success(self):
        root = self.fixture()
        (root/'upstream/arms/r-r/mismatch_sidecars/rank7.pt').unlink()
        self.assertEqual(rocm.collect(root)['verdict'],'inconclusive')

    def test_missing_retained_artifact_cannot_leave_a_pass_verdict(self):
        root=self.fixture(); (root/'preflight.json').unlink()
        self.assertEqual(rocm.collect(root)['verdict'],'inconclusive')

    def test_run_invokes_real_plan_then_collects_without_mocking_comparison(self):
        fixture=self.fixture(); output=self.root/'launched'
        def fake_process(argv, **kwargs):
            self.assertIn('run',argv)
            self.assertEqual(read_json(output/'profile.json')['defaults']['lr'],0.0)
            shutil.copytree(fixture/'upstream',output/'upstream')
            return subprocess.CompletedProcess(argv,0)
        with patch.object(rocm,'preflight',return_value={'synthetic_test':True}), \
             patch('subprocess.run',side_effect=fake_process), contextlib.redirect_stderr(io.StringIO()):
            result=rocm.run(self.profile,output)
        self.assertEqual(result['verdict'],'pass')
        self.assertEqual(result['comparison']['elements'],32)

    def test_recollection_does_not_trust_edited_diagnostic(self):
        root = self.fixture(); rocm.collect(root)
        value=read_json(root/'diagnostic.json'); value['comparison']['elements']=0
        write_json(root/'diagnostic.json',value)
        self.assertEqual(rocm.collect(root)['verdict'],'inconclusive')

    def test_nonzero_lr_and_rollout_reuse_rejected(self):
        root = self.fixture()
        path=root/'upstream/arms/r-r/launch.json'; launch=read_json(path)
        launch['environment']['RLK_ABLATION_LR']='1e-6'; write_json(path,launch)
        self.assertEqual(rocm.collect(root)['verdict'],'inconclusive')
        launch['environment']['RLK_ABLATION_LR']='0'
        launch['framework_consistency']['use_rollout_logprobs']=True; write_json(path,launch)
        self.assertEqual(rocm.collect(root)['verdict'],'inconclusive')

    def test_report_cli_exit_codes(self):
        root=self.fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['rocm-report','--run-dir',str(root)]),0)
