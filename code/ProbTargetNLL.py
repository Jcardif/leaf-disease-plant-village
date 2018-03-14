"""
Create a negtive likelihood loss function
where targets are probabilities

```python
Mynll = NLLProbTarget()
output = Mynll(target, logsoftmax)
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn as nn

# Inherit from Function
class NLLProbTargetFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, target, logsoftmax):
        ctx.save_for_backward(target, logsoftmax)
        output = -torch.mul(target,logsoftmax).sum(dim=1)
        output = output.mean(dim=0)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        target, logsoftmax = ctx.saved_variables
        length = len(target)
        grad_target = grad_logsoftmax =  None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_target = -(((logsoftmax/length).t() * grad_output).t())
        if ctx.needs_input_grad[1]:
            grad_logsoftmax = -(((target/length).t() * grad_output).t())
        return grad_target, grad_logsoftmax


class NLLProbTarget(nn.Module):
    def forward(self, target, logsoftmax):
        # See the autograd section for explanation of what happens here.
        return NLLProbTargetFunction.apply(target, logsoftmax)
