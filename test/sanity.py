import glob, torch

# adjust pattern if your .so lives elsewhere
so = glob.glob("../build/*.so")[0]
torch.ops.load_library(so)

print("Loaded:", so)
print("Has ooverlap_class:", hasattr(torch.classes, "ooverlap_class"))
print("BaselineImpl:", torch.classes.ooverlap_class.BaselineImpl)
print("Has op generate_nccl_id:", hasattr(torch.ops, "ooverlap_op"))
print("generate_nccl_id:", torch.ops.ooverlap_op.generate_nccl_id)
