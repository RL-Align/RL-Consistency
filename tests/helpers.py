"""Synthetic fixtures only. These do not attest to any GPU/model execution."""
import copy
import json
import struct
from pathlib import Path
from rl_consistency.common import digest, write_json


def recipe(backend='cuda'):
    path = Path(__file__).resolve().parents[1] / 'src/rl_consistency/recipes' / f'qwen3-8b-vime-{backend}.json'
    r = json.loads(path.read_text())
    r['model'].update(revision='unit-test-only', weights_sha256='1'*64, master_weights_sha256='1'*64,
                      tokenizer_sha256='2'*64, config_sha256='3'*64)
    r['inputs']['fixture_sha256'] = '4'*64
    r['environment'].update(lock_sha256='5'*64, profile_sha256='6'*64, python_version='3.12.0')
    r['environment']['sources'] = {k: '7'*40 for k in ('vime', 'vllm', 'megatron')}
    r['acceptance']['training_step_ratio_max'] = 2.0
    return r


def runtime(r):
    return {'clean': True, 'rl_kernel_commit': r['rl_kernel']['commit'],
            **copy.deepcopy(r['environment']), **copy.deepcopy(r['target']),
            'model': copy.deepcopy(r['model']), 'fixture_sha256': r['inputs']['fixture_sha256'],
            'loaded_artifacts': [{'worker': x, 'path': f'/synthetic/{x}.so', 'sha256': '8'*64}
                                 for x in ('training', 'rollout')]}


def trace(r, producer='rollout', value=-1.0, dtype='float32', weight=None, version='0'):
    raw = struct.pack('<f', value)
    if dtype == 'bfloat16': raw = raw[2:]
    elif dtype == 'float16': raw = struct.pack('<e', value)
    elif dtype == 'float64': raw = struct.pack('<d', value)
    return {'schema': 'rl-consistency.trace.v1', 'producer': producer, 'independent': True,
            'weight_version': version, 'weights_sha256': weight or r['model']['weights_sha256'],
            'inputs_sha256': r['inputs']['fixture_sha256'], 'semantics_sha256': digest(r['comparison']),
            'implementation_sha256': ('9' if producer == 'reference' else 'a')*64,
            'dtype': dtype, 'records': [{'sample': 's', 'position': 0, 'node': 'logp', 'shape': [], 'hex': raw.hex()}]}


def benchmark(r, mode='native', seconds=1.0):
    return {'mode': mode, 'fixture_sha256': r['inputs']['fixture_sha256'],
            'model_sha256': r['model']['weights_sha256'], 'target': r['target'], 'settings': r['benchmark'], 'sampling': r['sampling'],
            'execution_sha256': 'a'*64, 'warmup_steps': 1, 'step_seconds': [seconds]*3,
            'total_output_tokens': 128, 'peak_memory_bytes': 1000}


def evidence(r, gate):
    return {'schema': 'rl-consistency.evidence.v2', 'gate': gate, 'recipe_sha256': digest(r),
            'environment_sha256': digest(r['environment']), 'execution': 'real'}


def fixture(r, gate):
    e, artifacts = evidence(r, gate), {}
    if gate == 'C0':
        e['runtime'] = 'runtime.json'; artifacts['runtime.json'] = runtime(r)
        e['inventory'] = 'inventory.json'
        artifacts['inventory.json'] = {'route': r['target']['training']['engine'], 'modules': [
            {'operator': op, 'inference_implementation': 'synthetic', 'training_math': 'same formula',
             'quantization': 'none', 'discrete_choices': 'none'} for op in r['operators']]}
    elif gate in ('C1', 'C10'):
        e['benchmark'] = 'benchmark.json'
        artifacts['benchmark.json'] = benchmark(r, 'native' if gate == 'C1' else 'consistency')
    elif gate == 'C2':
        layer = {'operators': ['attention', 'ffn'], 'dimensions': {'hidden': 4096}, 'precision': 'bf16'}
        e['coverage'] = 'coverage.json'
        artifacts['coverage.json'] = {'source_config_sha256': r['model']['config_sha256'],
            'source_layers': [layer]*36, 'selected_layers': [0, 1, 2, 3],
            'reduced_layers': [layer]*4, 'boundary_operators': r['operators']}
    elif gate == 'C4':
        e['profile'] = 'profile.json'
        artifacts['profile.json'] = {'raw_profile_sha256':'a'*64, 'target':r['target'],
            'per_token_python_loop':False, 'unbounded_workspace':False, 'graph_capture_replay':True,
            'peak_memory_bytes':1024, 'operator_shapes':[{'operator':op, 'shape':[8,16], 'calls':1} for op in r['operators']]}
    elif gate == 'C6':
        e['backward'] = 'backward.json'
        artifacts['backward.json'] = {'modules':[{'operator':op, 'strategy':'reuse', 'rationale':'same formula',
            'saved_from_forward':['x'], 'gradient_contract_sha256':'a'*64} for op in r['operators']],
            'timing':{'target':r['target'], 'native_step_seconds':1.0, 'strict_step_seconds':1.5}}
    if gate == 'C5':
        e.update(runtime='runtime.json', bindings='bindings.json')
        artifacts['runtime.json'] = runtime(r)
        artifacts['bindings.json'] = [{'side': side, 'operator': op, 'observable': True,
           'fallback': False, 'implementation_sha256': 'a'*64, 'contract_sha256': 'b'*64}
           for side in ('training', 'rollout') for op in r['operators']]
    if gate in ('C3','C5','C7','C8','C9'):
        e['pairs'] = []
        for i, case in enumerate(r['coverage'][gate]):
            left, right = trace(r), trace(r, 'training')
            if gate == 'C7':
                left, right = trace(r, 'training'), trace(r, 'reference')
                right['reference_kind'] = 'independent_mathematical'
            if gate in ('C5','C9') and case in ('old-logprob','actor-forward'):
                right['execution_path'] = {'kind':case, 'model_mode':'train', 'forward_executed':True,
                    'calls':1, 'grad_enabled':case=='actor-forward', 'output_requires_grad':case=='actor-forward',
                    'callsite_sha256':'a'*64}
            if gate in ('C8','C9') and case.startswith('theta-'):
                step = int(case[-1])
                for t in (left, right):
                    t['weight_version'] = str(step)
                    t['weights_sha256'] = str(step+1)*64
            l, rr = f'{case}-left.json', f'{case}-right.json'
            e['pairs'].append({'case': case, 'left': l, 'right': rr})
            artifacts[l], artifacts[rr] = left, right
        if gate in ('C8','C9'):
            e['updates'] = 'updates.json'
            artifacts['updates.json'] = [{'version': str(i+1), 'before_sha256': str(i+1)*64,
                'after_sha256': str(i+2)*64, 'optimizer_state_sha256': 'c'*64,
                'before_master_sha256':str(i+1)*64, 'after_master_sha256':str(i+2)*64,
                'master_update_norm':0.1, 'gradient_norm':1.0, 'finite_loss':True,
                'gradient_coverage_complete':True, 'optimizer_step':i+1,
                'optimizer_step_executed': True} for i in range(2)]
            e['loop'] = 'loop.json'
            artifacts['loop.json'] = {'normal_exit':True, 'executed_updates':2, 'framework_entrypoint':'synthetic-test'}
        if gate == 'C9':
            e.update(full_model='full.json', checkpoint='checkpoint.json')
            artifacts['full.json'] = {'config_sha256': r['model']['config_sha256'], 'layers': 36,
                'context_tokens': 7168, 'target': r['target']}
            state = {k: 'd'*64 for k in ('model','optimizer','scheduler','rng','data_cursor')}
            state['model'] = r['model']['weights_sha256']
            artifacts['checkpoint.json'] = {'uninterrupted_state': state, 'restored_state': copy.deepcopy(state),
                'resumed_first_step': True, 'uninterrupted_trace': 'control.json', 'resumed_trace': 'resumed.json',
                'uninterrupted_after_update_sha256': 'e'*64, 'resumed_after_update_sha256': 'e'*64}
            artifacts['control.json'] = trace(r)
            artifacts['resumed.json'] = trace(r)
    return e, artifacts


def dump_fixture(directory, r, gate):
    directory.mkdir(parents=True, exist_ok=True)
    e, artifacts = fixture(r, gate)
    for name, data in artifacts.items(): write_json(directory / name, data)
    path = directory / 'evidence.json'; write_json(path, e)
    return path
