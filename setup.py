import gdown

output = "GoogleNews-vectors-negative300.bin.gz"
id = "11MmDxqhR3DJsbcyJrfEqF0XPlfuuiGkh"
gdown.download(id=id, output=output)

output = "results/checkpoints/svm_best_model.pkl"
id = "1ZdTVBXUFPS5UUXiHwSH0u2MbWWAaCyFY"
gdown.download(id=id, output=output)