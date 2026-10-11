
import runpy

print("\n========== Training Linear SVM ==========\n")
runpy.run_path("src/linear_svm.py", run_name="__main__")

print("\n========== Training RBF SVM ==========\n")
runpy.run_path("src/rbf_svm.py", run_name="__main__")

print("\n========== SVM Training Complete ==========")
