"""
Commenting this file out for now.

For some reason it screws up `test_training` by causing a device error
"""

# import pytest
# import torch


# @pytest.fixture(autouse=True)
# def set_default_device(request):
#     device = request.param if hasattr(request, "param") else None

#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     torch.set_default_device(device)

#     # if device == "cuda":
#     #     torch.set_default_tensor_type("torch.cuda.FloatTensor")
#     # else:
#     #     torch.set_default_tensor_type("torch.FloatTensor")

#     return device
