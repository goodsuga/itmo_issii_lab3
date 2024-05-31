# ---------------- Теперь оборачиваем в Bento ---------------------
# ---------------- Now wrap model into Bento ----------------------
from __future__ import annotations
import bentoml
from bentoml.io import NumpyNdarray
import numpy as np
import onnxruntime

m = bentoml.onnx.get("bento_cpu_model:latest").to_runner()

svc = bentoml.Service("model", runners=[m])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_series):
    return m.run.run(input_series)