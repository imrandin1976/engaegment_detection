# utils/model_loader.py
import torch
import torchvision.models as models

def load_model(model_path, num_classes=2):
    # Load the base SqueezeNet1.1 architecture
    model = models.squeezenet1_1(pretrained=False)
    
    # Replace the classifier for binary classification
    model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_classes

    # Load saved weights (handle both state_dict and full model)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # If saved with DataParallel or as {"model": ...}
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
    
    # Clean DataParallel key prefixes if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    return model
