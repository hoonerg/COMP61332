import gdown

output = "GoogleNews-vectors-negative300.bin.gz"
id = "11MmDxqhR3DJsbcyJrfEqF0XPlfuuiGkh"
gdown.download(id=id, output=output)

output = "results/checkpoints/svm_best_model.pkl"
id = "1ZdTVBXUFPS5UUXiHwSH0u2MbWWAaCyFY"
gdown.download(id=id, output=output)

output = "label_encoder.pkl"
id = "1V9V3qvhu88Fc2t5uYsE6s80TuIL6ATJI"
gdown.download(id=id, output=output)

output = "vocab.pkl"
id = "1YOrQquwlXr46LlIO-RzhvY_ZrjYB71qM"
gdown.download(id=id, output=output)
