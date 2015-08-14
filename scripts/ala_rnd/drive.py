import pickle
import sys

if __name__ == "__main__":
    _, in_fn, out_fn = sys.argv
    with open(in_fn, 'rb') as f:
        run = pickle.load(f)
    results = run.main_loop()
    with open(out_fn, 'wb') as f:
        pickle.dump(results, f)
