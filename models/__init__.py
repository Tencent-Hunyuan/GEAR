from .llamagen import *
from .vq_model import VQ_models
from .lfq_model import LFQ_models
from .glfq_model import GLFQ_models
from .ibq_model import IBQ_models
from .generate import generate

Models = {}
Models.update(LlamaGen_models)

# Unified tokenizer registry: anything callers used to look up via
# ``VQ_models`` should now use ``Tokenizers`` so adding new tokenizer
# families (LFQ, IBQ, ...) is a one-line registry merge rather
# than a flurry of import edits.
Tokenizers = {}
Tokenizers.update(VQ_models)
Tokenizers.update(LFQ_models)
Tokenizers.update(GLFQ_models)
Tokenizers.update(IBQ_models)
