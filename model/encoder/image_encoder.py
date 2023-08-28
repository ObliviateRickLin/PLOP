import torch
import torch.nn as nn
import torchvision.models as models

class VGG16Encoder(nn.Module):
    def __init__(self):
        super(VGG16Encoder, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        self.classifier = vgg16.classifier

    def forward(self, x):
        outputs = []
        count_conv2d = 0  # Counter to keep track of the number of Conv2D layers in each block
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                count_conv2d += 1
                
                # Check if this Conv2d layer is the last in its block (i.e., right before a pooling layer)
                if count_conv2d in [2, 4, 7, 10, 13]:
                    outputs.append(x)
        
        # Flatten the output to feed it into the classifier
        x = x.view(x.size(0), -1)
        classifier_output = self.classifier(x)

        # Add the classifier output to the list of outputs
        outputs.append(classifier_output)
                          
        return outputs

if __name__ == "＿＿main__":
    model = VGG16Encoder()
    input_tensor = torch.randn(1, 3, 224, 224)

    output_list = model(input_tensor)
    output_shapes = [out.shape for out in output_list]

    print("Output shapes:", output_shapes)
