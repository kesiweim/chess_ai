"""FP32 traced inference; no quantization, no weight updates.
Must pass the benchmark before use in play. CPU only, one instance per worker.
"""
import torch
from neural_fast import FastEvaluator

class TracedEvaluator(FastEvaluator):
    def __init__(self,model):
        super().__init__(model)
        with torch.inference_mode():
            example=(torch.zeros(1,19,8,8),torch.zeros(1))
            traced=torch.jit.trace(model.eval(),example,check_trace=True)
            # Avoid optional numerical graph rewrites. Test exact outputs anyway.
            self.model=torch.jit.freeze(traced.eval(),optimize_numerics=False)
            for _ in range(20):self.model(*example)
