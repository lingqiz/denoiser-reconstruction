import torch
import torch.nn as nn
import pickle
import argparse

class UNet_flex(nn.Module):
    def __init__(self, args):
        super(UNet_flex,self).__init__()

        # self.pool_window = args.pool_window
        args.num_blocks = len(args.num_kernels)-1
        self.num_blocks = args.num_blocks
        # self.activations = args.activations
        self.RF = self.compute_RF(args)
        if args.conditional_inp:
            self.inp_channels = args.num_channels * 2
        else:
            self.inp_channels = args.num_channels
        # Encoder
        self.encoder = nn.ModuleDict([])
        for b in range(self.num_blocks):
            self.encoder[str(b)] = self.init_encoder_block(b,args)

        # Mid-layers
        self.mid = self.init_mid_block(b,args)

        # Decoder
        self.decoder = nn.ModuleDict([])
        self.upsample = nn.ModuleDict([])
        for b in range(self.num_blocks-1,-1,-1):
            self.upsample[str(b)], self.decoder[str(b)] = self.init_decoder_block(b,args)

    def forward(self, x):
        # activations = {}
        # pool =  nn.AvgPool2d(kernel_size=self.pool_window, stride=2, padding=int((self.pool_window-1)/2) )
        pool =  nn.AvgPool2d(kernel_size=2, stride=2, padding=0 )
        # Encoder
        unpooled = []
        for b in range(self.num_blocks):
            x_unpooled = self.encoder[str(b)](x)
            x = pool(x_unpooled)
            unpooled.append(x_unpooled)
            # if self.activations:
                # activations['enc'+str(b)] = x_unpooled
        # Mid-layers
        x = self.mid(x)
        # if self.activations:
            # activations['mid'] = x
        # Decoder
        for b in range(self.num_blocks-1, -1, -1):
            x = self.upsample[str(b)](x)
            x = torch.cat([x, unpooled[b]], dim = 1)
            x = self.decoder[str(b)](x)
            # if self.activations:
                # activations['dec'+str(b)] = x

        # if self.activations:
            # return activations
        else:
            return x


    def init_encoder_block(self, b, args):
        enc_layers = nn.ModuleList([])
        if b==0:
            enc_layers.append(nn.Conv2d(self.inp_channels ,args.num_kernels[b], args.kernel_size, padding=args.padding, bias=args.bias))
            enc_layers.append(nn.ReLU(inplace=True))
            for l in range(1,args.num_enc_conv[b]):
                enc_layers.append(nn.Conv2d(args.num_kernels[b] ,args.num_kernels[b], args.kernel_size, padding=args.padding, bias=args.bias))
                enc_layers.append(BF_batchNorm(args.num_kernels[b],args.instanceNorm))
                # enc_layers.append(nn.GroupNorm(num_groups=args.num_kernels[b], num_channels=args.num_kernels[b]) )
                enc_layers.append(nn.ReLU(inplace=True))
        else:
            for l in range(args.num_enc_conv[b]):
                if l==0:
                    enc_layers.append(nn.Conv2d(args.num_kernels[b-1] ,args.num_kernels[b], args.kernel_size, padding=args.padding, bias=args.bias))
                else:
                    enc_layers.append(nn.Conv2d(args.num_kernels[b] ,args.num_kernels[b], args.kernel_size, padding=args.padding, bias=args.bias))
                enc_layers.append(BF_batchNorm(args.num_kernels[b],args.instanceNorm))
                # enc_layers.append(nn.GroupNorm(num_groups=args.num_kernels[b], num_channels=args.num_kernels[b]) )
                enc_layers.append(nn.ReLU(inplace=True))


        return nn.Sequential(*enc_layers)

    def init_mid_block(self, b, args):
        mid_block = nn.ModuleList([])
        for l in range(args.num_mid_conv):
            if l==0:
                mid_block.append(nn.Conv2d(args.num_kernels[b] ,args.num_kernels[b+1], args.kernel_size, padding=args.padding , bias=args.bias))
            else:
                mid_block.append(nn.Conv2d(args.num_kernels[b+1] ,args.num_kernels[b+1], args.kernel_size, padding=args.padding , bias=args.bias))
            mid_block.append(BF_batchNorm(args.num_kernels[b+1],args.instanceNorm ))
            # mid_block.append(nn.GroupNorm(num_groups=args.num_kernels[b+1], num_channels=args.num_kernels[b+1]) )
            mid_block.append(nn.ReLU(inplace=True))

        return nn.Sequential(*mid_block)

    def init_decoder_block(self, b, args):
        dec_layers = nn.ModuleList([])

        #initiate the last block:
        if b==0:
            for l in range(args.num_dec_conv[b]-1):
                if l==0:
                    upsample = nn.ConvTranspose2d(args.num_kernels[b+1], args.num_kernels[b], kernel_size=2, stride=2,bias=False)
                    dec_layers.append(nn.Conv2d(args.num_kernels[b]*2, args.num_kernels[b], kernel_size=args.kernel_size, padding=args.padding,bias=args.bias))
                else:
                    dec_layers.append(nn.Conv2d(args.num_kernels[b] ,args.num_kernels[b], args.kernel_size, padding=args.padding, bias=args.bias))
                dec_layers.append(BF_batchNorm(args.num_kernels[b],args.instanceNorm))
                # dec_layers.append(nn.GroupNorm(num_groups=args.num_kernels[b], num_channels=args.num_kernels[b]) )
                dec_layers.append(nn.ReLU(inplace=True))

            dec_layers.append(nn.Conv2d(args.num_kernels[b], args.num_channels, kernel_size=args.kernel_size, padding=args.padding,bias=args.bias))

        #other blocks
        else:
            for l in range(args.num_dec_conv[b]):
                if l==0:
                    upsample= nn.ConvTranspose2d(args.num_kernels[b+1], args.num_kernels[b], kernel_size=2, stride=2,bias=args.bias)
                    dec_layers.append(nn.Conv2d(args.num_kernels[b]*2, args.num_kernels[b], kernel_size=args.kernel_size, padding=args.padding,bias=args.bias))

                else:
                    dec_layers.append(nn.Conv2d(args.num_kernels[b] ,args.num_kernels[b], args.kernel_size, padding=args.padding, bias=args.bias))

                dec_layers.append(BF_batchNorm(args.num_kernels[b],args.instanceNorm))
                # dec_layers.append(nn.GroupNorm(num_groups=args.num_kernels[b], num_channels=args.num_kernels[b]) )
                dec_layers.append(nn.ReLU(inplace=True))
        return upsample, nn.Sequential(*dec_layers)

    def compute_RF(self,args):
        '''
        RF is the size of the neighborhood from which one pixel in the last layer of mid block is computed.
        returns a scalar value which is the size of the RF in one dimension
        Assuming all the kernels and strides are squares, not rectangles
        '''
        r = 0
        ## RF at the end of the last encoder block
        for b in range(args.num_blocks):
            s = 2**b #effective stride
            r += args.num_enc_conv[b] * ((args.kernel_size-1) * s) + ((2-1) * s) #hard-coded 2 because of 2x2 pooling. Change if different

        ## RF at the end of the last layer of the mid block
        s = 2**(b+1)
        r += args.num_mid_conv * ((args.kernel_size-1) * s)

        r = r+1
        return r


class BF_batchNorm(nn.Module):
    def __init__(self, num_kernels, instanceNorm=False):
        super(BF_batchNorm, self).__init__()
        g = (torch.randn( (1,num_kernels,1,1) )*(2./9./64.)).clamp_(-0.025,0.025)
        self.gammas = nn.Parameter(g, requires_grad=True)
        self.instanceNorm = instanceNorm
        if instanceNorm==False:
            self.register_buffer("running_sd", torch.ones(1,num_kernels,1,1))

    def forward(self, x):
        if self.instanceNorm==False: # do batch norm
            training_mode = self.training
            sd_x = torch.sqrt(x.var(dim=(0,2,3) ,keepdim = True, unbiased=False)+ 1e-05)
            if training_mode:
                x = x / sd_x.expand_as(x)
                with torch.no_grad():
                    self.running_sd.copy_((1-.1) * self.running_sd.data + .1 * sd_x)

                x = x * self.gammas.expand_as(x)

            else:
                x = x / self.running_sd.expand_as(x)
                x = x * self.gammas.expand_as(x)

        else: ## do instance norm
            sd_x = x.std(dim=(2,3) ,keepdim = True, unbiased=False)+ 1e-05
            x = x / sd_x.expand_as(x)
            x = x * self.gammas.expand_as(x)
        return x


def initialize_network(network_name, args):
    '''
    Function to dynamically initialize a neural network by class name
    '''
    if network_name in globals() and issubclass(globals()[network_name], nn.Module):
        return globals()[network_name](args)
    else:
        raise ValueError(f"Network {network_name} not found or not a subclass of nn.Module")


def load_learned_model(folder_path, print_args=False):
    '''
    Loads dictionary of all args used to define the model for training and then loads the saved trained model with the specified parameters.
    This can only be used if the network parameters were saved in advanced.
    '''
    with (open(folder_path +'exp_arguments.pkl' , "rb")) as openfile:
        arguments = pickle.load(openfile)

    if print_args:
        print('*************** saved arguments:*************')
        for key,v in arguments.items():
            print(key, v)

    parser = argparse.ArgumentParser(description='set CNN args')

    for k,v in arguments.items():
        parser.add_argument('--' + k, default=v)
    args = parser.parse_args('')

    model = initialize_network(args.arch_name, args)
    if torch.cuda.is_available():
        model = model.cuda()

    model = read_trained_params(model, folder_path + '/model.pt')
    print('******************************************************')
    print('number of parameters is ' , sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.eval()
    print('train mode:', model.training )
    return model

def read_trained_params(model, path):
    '''reads parametres of saved models into an initialized network'''
    if torch.cuda.is_available():
        learned_params =torch.load(path)

    else:
        learned_params =torch.load(path, map_location='cpu' )

    ## unwrap if in Dataparallel
    new_state_dict = {}
    for key,value in learned_params.items():
        if key.split('.')[0] == 'module':
            new_key = '.'.join(key.split('.')[1::])
            new_state_dict[new_key] = value

        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    model.eval();

    return model