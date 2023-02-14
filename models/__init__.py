from models.palette_tensoRF import PaletteTensorVM
from models.tensoRF import TensorCP, TensorVMSplit

MODEL_ZOO = {a.__name__: a for a in (TensorCP, TensorVMSplit, PaletteTensorVM)}
