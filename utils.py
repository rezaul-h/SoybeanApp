import torch
import timm
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def load_model(model_path, num_classes):
    model = timm.create_model("maxvit_tiny_rw_224", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_with_cam(model, image_path, cam_path, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image)
    input_tensor = img_tensor.unsqueeze(0)

    rgb_img = ((img_tensor * 0.5) + 0.5).permute(1, 2, 0).numpy()

    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        class_idx = probs.argmax().item()

    # ✅ Updated target layer for GradCAM (final conv layer)
    target_layers = [model.stages[3].blocks[-1].conv.conv2_kxk]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])[0]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    Image.fromarray(cam_image).save(cam_path)

    predicted_class = class_names[class_idx]
    probabilities_percent = [round(p.item() * 100, 1) for p in probs]

    return predicted_class, probabilities_percent
