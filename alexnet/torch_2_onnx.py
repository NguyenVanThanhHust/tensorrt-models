import numpy as np
import torch
import torchvision

INPUT_NAME = "input"
OUTPUT_NAME = "output"

if __name__ == "__main__":
    model = torchvision.models.alexnet(weights="AlexNet_Weights.DEFAULT")
    model.eval()

    batch_size = 1
    # Input to the model
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=False)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "alexnet.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=[INPUT_NAME],  # the model's input names
        output_names=[OUTPUT_NAME],  # the model's output names
        dynamic_axes={
            INPUT_NAME: {0: "batch_size"},  # variable length axes
            OUTPUT_NAME: {0: "batch_size"},
        },
    )

    import onnx

    onnx_model = onnx.load("alexnet.onnx")
    onnx.checker.check_model(onnx_model)

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(
        "alexnet.onnx",
        providers=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
