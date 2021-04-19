import torch
from torch import nn
from torch.nn import functional as F

class BlockPart(nn.Module):
    def __init__(self, part_number):
        super().__init__()
        self.part_number = part_number
        self.forward = self.no_relationships

    def no_relationships(self, x):
        out = self.layers(x)
        return out

    def has_relationships(self, x):
        self.out = self.layers(x)
        return self.out

    def store_results(self):
        self.forward = self.has_relationships

class VGGBlockPart(BlockPart):
    def __init__(self, part_number, in_channels, out_channels, kernel, stride, padding):
        super().__init__(part_number)
        self.layers = nn.Sequential(
            *[
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel,stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
        )

class VGGEncoderBlock(nn.Module):
    def __init__(self, layer_data, kernel=3, stride=1, padding=1, block_part=None):
        super().__init__()
        layer_count, in_channels, out_channels, _ = layer_data
        if block_part is None:
            block_part = VGGBlockPart

        layers = []
        layers.append(block_part(0, in_channels=in_channels,out_channels=out_channels,kernel=kernel,stride=stride, padding=padding))
        for layer in range(1, layer_count):
            layers.append(block_part(layer, in_channels=out_channels,out_channels=out_channels,kernel=kernel,stride=stride, padding=padding))

        self.layers = layers
        self.layers_executor = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers_executor(x)
        return out


class VGGDecoderBlock(nn.Module):
    def __init__(self, block_index, layer_data, kernel=3, stride=1, padding=1, block_part=None):
        super().__init__()
        layer_count, _, out_channels, deeper_channels = layer_data
        if block_part is None:
            block_part = VGGBlockPart

        layers = []
        layers.append(block_part(0, in_channels=out_channels * (block_index + 1) + deeper_channels,out_channels=out_channels,kernel=kernel,stride=stride, padding=padding))
        for layer in range(1, layer_count):
            layers.append(block_part(layer, in_channels=out_channels,out_channels=out_channels,kernel=kernel,stride=stride, padding=padding))
        
        self.layers = layers
        self.layers_executor = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers_executor(x)
        return out



class SamplerPair:
    def down(self, x):
        return self.downsampler(x)
    def up(self, x):
        return self.upsampler(x)


class BasicSamplerPair(SamplerPair):
    def __init__(self):
        self.downsampl_fun = nn.MaxPool2d(2)
    def downsampler(self, x):
        self.shape = x.shape[2:]
        return self.downsampl_fun(x)
    def upsampler(self, x):
        return F.interpolate(x, self.shape, mode="bilinear", align_corners=False)


class BaseBlock(nn.Module):
    def __init__(self):
        super().__init__()
    def clear_result(self):
        del self.out
    def forward(self, x):
        self.out = self.forward_handler(x)
        return self.out

class DecoderBlock(BaseBlock):
    def __init__(self, block_index, related_block, layer_data, sampler, layer_builder=None):
        super().__init__()
        self.sampler = sampler
        if layer_builder is None:
            layer_builder = VGGDecoderBlock
        self.related_block = related_block
        self.layers = layer_builder(block_index, layer_data)

    def forward_handler(self, x):
        upsampled = self.sampler.up(self.related_block.out)
        x = [*x, upsampled]
        concated = torch.cat(x, 1)
        out = self.layers(concated)
        return out


class EncoderBlock(BaseBlock):
    def __init__(self, layer_data, layer_builder=None):
        super().__init__()
        if layer_builder is None:
            layer_builder = VGGEncoderBlock
        self.layers = layer_builder(layer_data)

    def forward_handler(self, x):
        return self.layers(x)


class ModelLayer(nn.Module):
    def __init__(self, layer_number, layer_data, prev_layer=None, sampler=None, encoder_layer_builder=None, decoder_layer_builder=None):
        super().__init__()
        #set default forward function
        self.forward = self.__no_final_layer

        self.layer_data = layer_data
        self.deep_supervision_level = layer_number

        if prev_layer is None and layer_number != 0:
                raise Exception("if layer_number > 0, the previous layer must be passed as argument")
        self.prev_layer = prev_layer

        if sampler is None:
            sampler = BasicSamplerPair
        self.sampler = sampler()

        self.elements = [EncoderBlock(layer_data, layer_builder=encoder_layer_builder)]

        #build decoder blocks of current layer
        for ind in range(layer_number):
            self.elements.append(DecoderBlock(ind, prev_layer.elements[ind], layer_data, self.sampler, layer_builder=decoder_layer_builder))


    def __no_final_layer(self, x):
        x = self.elements[0](x)

        out = [x]
        if self.deep_supervision_level > 0:
            #execute lower level
            x = self.sampler.down(x)
            self.prev_layer(x)

            elems = self.elements[1:self.deep_supervision_level + 1]
            for elem in elems:
                out.append(elem(out))

        return out[-1]
    

    def __has_final_layer(self, x):
        out = self.__no_final_layer(x)
        out = self.final(out)
        out = self.softmax(out)
        return out
        
    def set_deep_supervision_level(self, level:int):
        if level > len(self.elements) - 1:
            raise Exception("deep supervision level can not be greater than number of decoder blocks in layer, which is " + str(len(self.elements) - 1))
        self.deep_supervision_level = level
        if self.prev_layer is not None:
            self.prev_layer.set_deep_supervision_level(level - 1)

    def add_final_layer(self, out_channels):
        self.softmax = nn.Softmax2d()
        self.final = nn.Conv2d(self.layer_data[2], out_channels, kernel_size=1, stride=1)
        self.forward = self.__has_final_layer
    
    def remove_final_layer(self):
        if self.final is not None:
            del self.final
        if self.softmax is not None:
            del self.softmax
        self.forward = self.__no_final_layer


def build_model(model_data, out_channels):
    network = ModelLayer(0, model_data[-1])

    for ind, layer_data in enumerate(reversed(model_data[:-1]), 1):
        current_layer = ModelLayer(ind, layer_data, network)
        network = current_layer
    
    network.add_final_layer(out_channels)

    return network

def default_model(in_channels, out_channels):
    data = [
        (2, in_channels, 32, 64),
        (3,  32,  64, 128),
        (3,  64, 128, 256),
        (3, 128, 256, 512),
        (3, 256, 512, 512)
    ]

    return build_model(data, out_channels)