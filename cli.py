import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ikine', '--inverse_kine', action='store_true')
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--model', type=str)

args = parser.parse_args()

print(args.inverse_kine)

ik_method = args.method
print(ik_method)

ann_model = args.model
if ann_model is None:
    print(ann_model)
else:
    raise Exception("error")
