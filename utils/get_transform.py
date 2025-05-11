from torchvision import transforms

def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    return transform