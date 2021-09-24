# USAGE
# python coremlconverter.py --model pokedex.model --labelbin lb.pickle

# 导入一些重要的包
from keras.models import load_model
import coremltools
import argparse
import pickle

# 加载命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
args = vars(ap.parse_args())

# 加载种类标签
print("[INFO] loading class labels from label binarizer")
lb = pickle.loads(open(args["labelbin"], "rb").read())
class_labels = lb.classes_.tolist()
print("[INFO] class labels: {}".format(class_labels))

# 加载已经训练好的模型
print("[INFO] loading model...")
model = load_model(args["model"])

# 将模型转化为Core ML的形式
print("[INFO] converting model")
coreml_model = coremltools.converters.keras.convert(model,
	input_names="image",
	image_input_names="image",
	image_scale=1/255.0,
	class_labels=class_labels,
	is_bgr=True)

# 保存模型
output = args["model"].rsplit(".", 1)[0] + ".mlmodel"
print("[INFO] saving model as {}".format(output))
coreml_model.save(output)