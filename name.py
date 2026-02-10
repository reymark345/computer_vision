from roboflow import Roboflow
rf = Roboflow(api_key="nf2dXJZ2CTWfVcFNuMqD")
project = rf.workspace("test-mango").project("mango-detection-iobne")
version = project.version(1)
dataset = version.download("coco-segmentation")
                