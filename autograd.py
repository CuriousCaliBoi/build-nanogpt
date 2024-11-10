import torch

x = torch.randn(5, requires_grad=True)
y = torch.randn(5, requires_grad=True)

# Each operation creates an AutogradFunction
z = x * y  # MulBackward
print(z.grad_fn)  # Shows the operation that created this tensor

w = z.sum()  # SumBackward
print(w.grad_fn)  

w.backward()  # Computes d(w)/dx and d(w)/dy
print(x.grad)  # Contains gradient of w with respect to x