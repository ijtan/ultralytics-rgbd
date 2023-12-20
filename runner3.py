# import torch
# torch.cuda.set_device(0)  # Set to your desired GPU number
from ultralytics import YOLO


# build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8-rgb.yaml")
    # Use the model
    res = 1024
    batch_size = 8
    model.train(
        data="AmphoraBGDepthNoCc.yaml",
        epochs=300,
        imgsz=res,
        resume=False,
        batch=batch_size,  # train 300 epochs
        name="BGDepth-NoAugments-NoCC-1024_batch8__",
        augment=True,
    )
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format
