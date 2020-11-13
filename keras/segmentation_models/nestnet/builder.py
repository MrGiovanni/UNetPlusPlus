from keras.layers import Conv2D
from keras.layers import Activation
from keras.models import Model

from .blocks import Transpose2D_block
from .blocks import Upsample2D_block
from ..utils import get_layer_number, to_tuple

import copy


def build_nestnet(backbone, classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               use_batchnorm=True):

    input = backbone.input
    # print(n_upsample_blocks)

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    if len(skip_connection_layers) > n_upsample_blocks:
        downsampling_layers = skip_connection_layers[int(len(skip_connection_layers)/2):]
        skip_connection_layers = skip_connection_layers[:int(len(skip_connection_layers)/2)]
    else:
        downsampling_layers = skip_connection_layers


    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])
    skip_layers_list = [backbone.layers[skip_connection_idx[i]].output for i in range(len(skip_connection_idx))]
    downsampling_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in downsampling_layers])
    downsampling_list = [backbone.layers[downsampling_idx[i]].output for i in range(len(downsampling_idx))]
    downterm = [None] * (n_upsample_blocks+1)
    for i in range(len(downsampling_idx)):
        # print(downsampling_list[0])
        # print(backbone.output)
        # print("")
        if downsampling_list[0] == backbone.output:
            # print("VGG16 should be!")
            downterm[n_upsample_blocks-i] = downsampling_list[i]
        else:
            downterm[n_upsample_blocks-i-1] = downsampling_list[i]
    downterm[-1] = backbone.output
    # print("downterm = {}".format(downterm))

    interm = [None] * (n_upsample_blocks+1) * (n_upsample_blocks+1)
    for i in range(len(skip_connection_idx)):
        interm[-i*(n_upsample_blocks+1)+(n_upsample_blocks+1)*(n_upsample_blocks-1)] = skip_layers_list[i]
    interm[(n_upsample_blocks+1)*n_upsample_blocks] = backbone.output

    for j in range(n_upsample_blocks):
        for i in range(n_upsample_blocks-j):
            upsample_rate = to_tuple(upsample_rates[i])
            # print(j, i)
            
            if i == 0 and j < n_upsample_blocks-1 and len(skip_connection_layers) < n_upsample_blocks:
                interm[(n_upsample_blocks+1)*i+j+1] = None
            elif j == 0:
                if downterm[i+1] is not None:
                    interm[(n_upsample_blocks+1)*i+j+1] = up_block(decoder_filters[n_upsample_blocks-i-2], 
                                      i+1, j+1, upsample_rate=upsample_rate,
                                      skip=interm[(n_upsample_blocks+1)*i+j], 
                                      use_batchnorm=use_batchnorm)(downterm[i+1])
                else:
                    interm[(n_upsample_blocks+1)*i+j+1] = None
                # print("\n{} = {} + {}\n".format(interm[(n_upsample_blocks+1)*i+j+1],
                #                             interm[(n_upsample_blocks+1)*i+j], 
                #                             downterm[i+1]))
            else:
                interm[(n_upsample_blocks+1)*i+j+1] = up_block(decoder_filters[n_upsample_blocks-i-2], 
                                  i+1, j+1, upsample_rate=upsample_rate,
                                  skip=interm[(n_upsample_blocks+1)*i+j], 
                                  use_batchnorm=use_batchnorm)(interm[(n_upsample_blocks+1)*(i+1)+j])
                # print("\n{} = {} + {}\n".format(interm[(n_upsample_blocks+1)*i+j+1],
                #                             interm[(n_upsample_blocks+1)*i+j], 
                #                             interm[(n_upsample_blocks+1)*(i+1)+j]))
    # print('\n\n\n')
    # for x in range(n_upsample_blocks+1):
    #     for y in range(n_upsample_blocks+1):
    #         print(interm[x*(n_upsample_blocks+1)+y], end=' ', flush=True)
    #     print('\n')
    # print('\n\n\n')
    #print(interm)

    """
    for i in range(n_upsample_blocks-2):
        interm = []
        x = skip_layers_list[n_upsample_blocks-i-2]

    
    x = {}
    for stage in range(n_upsample_blocks-1):
        i = n_upsample_blocks - stage - 1
        x = backbone.layers[skip_connection_idx[i-1]].output
        for col in range(stage+1):
            print("i = {}, col = {}, index = {}".format(i, col, i+col))
            skip_connection = None
            if i-col < len(skip_connection_idx):
                skip_connection = skip_layers_list[i-col]
            upsample_rate = to_tuple(upsample_rates[i-col])
            x = up_block(decoder_filters[i-col], stage-col+1, col+1, upsample_rate=upsample_rate,
                         skip=skip_connection, use_batchnorm=use_batchnorm)(x)
            skip_layers_list[i+col] = x

    x = backbone.output
    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            # skip_connection = backbone.layers[skip_connection_idx[i]].output
            skip_connection = skip_layers_list[i]

        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], n_upsample_blocks-i, 0, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)
    """

    """
    i = n_upsample_blocks - 1
    xx = backbone.layers[skip_connection_idx[i-0-1]].output
    skip_connection = skip_layers_list[i-0]
    upsample_rate = to_tuple(upsample_rates[i-0])
    xx = up_block(decoder_filters[i-0], n_upsample_blocks-i-0, 1+0, upsample_rate=upsample_rate,
                 skip=skip_connection, use_batchnorm=use_batchnorm)(xx)
    skip_layers_list[i-0] = xx

    i = n_upsample_blocks - 2
    xx = backbone.layers[skip_connection_idx[i-0-1]].output

    skip_connection = skip_layers_list[i-0]
    upsample_rate = to_tuple(upsample_rates[i-0])
    xx = up_block(decoder_filters[i-0], n_upsample_blocks-i-0, 1+0, upsample_rate=upsample_rate,
                 skip=skip_connection, use_batchnorm=use_batchnorm)(xx)
    skip_layers_list[i-0] = xx

    skip_connection = skip_layers_list[i-1]
    upsample_rate = to_tuple(upsample_rates[i-1])
    xx = up_block(decoder_filters[i-1], n_upsample_blocks-i-1, 1+1, upsample_rate=upsample_rate,
                 skip=skip_connection, use_batchnorm=use_batchnorm)(xx)
    skip_layers_list[i-1] = xx
    """
    
    """
    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            # skip_connection = backbone.layers[skip_connection_idx[i]].output
            skip_connection = skip_layers_list[i]

        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], n_upsample_blocks-i, 0, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)
    """

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(interm[n_upsample_blocks])
    x = Activation(activation, name=activation)(x)

    model = Model(input, x)

    return model
