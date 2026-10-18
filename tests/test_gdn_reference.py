"""Analytical examples and numerical-gradient checks for the FP64 reference only."""
import unittest
from rl_consistency.common import ContractError
from rl_consistency.models.gdn_reference import gated_delta_reference
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"optional CPU PyTorch not installed")
class GDNReferenceTests(unittest.TestCase):
    def scalar_inputs(self):
        q=torch.ones(2,1,1,dtype=torch.float64)
        v=torch.tensor([2.,4.],dtype=torch.float64).reshape(2,1,1)
        g=torch.full((2,1),0.5,dtype=torch.float64).log()
        beta=torch.full((2,1),0.5,dtype=torch.float64)
        return q,q.clone(),v,g,beta

    def test_hand_computed_scalar_decay_and_delta_update(self):
        out,state=gated_delta_reference(*self.scalar_inputs(),cu_seqlens=[0,2],normalize_qk=False)
        torch.testing.assert_close(out.flatten(),torch.tensor([1.,2.25],dtype=torch.float64))
        self.assertEqual(state.item(),2.25)

    def test_packed_sequences_reset_state(self):
        out,states=gated_delta_reference(*self.scalar_inputs(),cu_seqlens=[0,1,2],normalize_qk=False)
        self.assertEqual(out.flatten().tolist(),[1.,2.])
        self.assertEqual(states.flatten().tolist(),[1.,2.])

    def test_grouped_heads_and_resumed_state_match_continuous_recurrence(self):
        torch.manual_seed(0)
        tensors=[torch.randn(4,1,2),torch.randn(4,1,2),torch.randn(4,3,2),
                 -torch.rand(4,3),torch.rand(4,3)]
        full,final=gated_delta_reference(*tensors,cu_seqlens=[0,4])
        first,state=gated_delta_reference(*(t[:2] for t in tensors),cu_seqlens=[0,2])
        snapshot=state.clone()
        last,resumed=gated_delta_reference(*(t[2:] for t in tensors),cu_seqlens=[0,2],initial_state=state)
        self.assertTrue(torch.equal(full,torch.cat([first,last])))
        self.assertTrue(torch.equal(final,resumed))
        self.assertTrue(torch.equal(state,snapshot))
        self.assertEqual(full.dtype,torch.float64)

    def test_all_input_gradients_against_finite_differences(self):
        torch.manual_seed(2)
        args=[torch.randn(2,1,2,dtype=torch.float64)*.2,torch.randn(2,1,2,dtype=torch.float64)*.2,
              torch.randn(2,2,2,dtype=torch.float64)*.2, -torch.rand(2,2,dtype=torch.float64)-.2,
              torch.rand(2,2,dtype=torch.float64)*.5+.2]
        initial=torch.randn(1,2,2,2,dtype=torch.float64)*.1
        args=[x.requires_grad_() for x in [*args,initial]]
        self.assertTrue(torch.autograd.gradcheck(lambda *x:gated_delta_reference(*x[:5],cu_seqlens=[0,2],initial_state=x[5]),
                                                tuple(args),eps=1e-6,atol=1e-5,rtol=1e-3))

    def test_invalid_boundaries_nonfinite_and_unstable_gates_rejected(self):
        for boundaries in ([1,2],[0,1,1,2],[0,3],[0,True,2]):
            with self.assertRaises(ContractError): gated_delta_reference(*self.scalar_inputs(),cu_seqlens=boundaries)
        for index,value in ((0,float("nan")),(3,0.1),(4,1.1)):
            args=list(self.scalar_inputs()); args[index][0]=value
            with self.assertRaises(ContractError): gated_delta_reference(*args,cu_seqlens=[0,2])
