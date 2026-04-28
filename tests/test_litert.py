import torch
import torch.nn as nn
import litert_torch as lrt

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

model = TestModel().eval()
dummy_input = torch.randn(1, 4)

# IMPORTANT: sample_args must be a tuple
edge_model = lrt.convert(model, (dummy_input,))

# export result
edge_model.export("test.tflite")

print("Conversion successful")