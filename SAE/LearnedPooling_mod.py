#based on https://github.com/MEPP-team/SAE
import torch
import torch.nn as nn


class LearnedPooling(nn.Module):
    def __init__(self, opt):
        super(LearnedPooling, self).__init__()

        encoder_features = opt['encoder_features']

        sizes_downsample = [opt['nb_freq'], opt['downsample'][0],opt['downsample'][1],opt['downsample'][2],opt['downsample'][3]] if 'downsample' in opt else [opt['nb_freq'], 256, 64, 32, 16]
        sizes_upsample = sizes_downsample[::-1]

        sizes_convs_encode = [opt['size_conv'], opt['size_conv'], opt['size_conv'], opt['size_conv']] if hasattr(opt['size_conv'], "__len__")==False else [opt['size_conv'][0], opt['size_conv'][1], opt['size_conv'][2], opt['size_conv'][3]]
        sizes_convs_decode = sizes_convs_encode[::-1]
        
        try: 
            layers = opt["layers"]
        except KeyError:
            layers = 1
            
        try: 
            batch = opt["batch"]
        except KeyError:
            batch = False
        
        try: 
            drop = opt["drop"]
        except KeyError:
            drop = False
            
        try: 
            act = opt["act"]
        except KeyError:
            act = False
            
        try: 
            self.drop_att = opt["drop_att"]
        except KeyError:
            self.drop_att = False
    
        if layers == 1:
            encoder_linear = [encoder_features[-1] * sizes_downsample[-1], opt['size_latent']]
            decoder_linear = [opt['size_latent'], encoder_features[-1] * sizes_downsample[-1]]
        else:
            encoder_linear = [encoder_features[-1] * sizes_downsample[-1],layers, opt['size_latent']]
            decoder_linear = [opt['size_latent'],layers, encoder_features[-1] * sizes_downsample[-1]]
            

        decoder_features = encoder_features[::-1]
        decoder_features[-1] = decoder_features[-2]

        self.latent_space = encoder_linear[-1]

        if opt['activation_func'] == "ReLU":
            self.activation = nn.ReLU
        elif opt['activation_func'] == "Tanh":
            self.activation = nn.Tanh
        elif opt['activation_func'] == "Sigmoid":
            self.activation = nn.Sigmoid
        elif opt['activation_func'] == "LeakyReLU":
            self.activation = nn.LeakyReLU
        elif opt['activation_func'] == "ELU":
            self.activation = nn.ELU
        elif opt['activation_func'] == "SiLU":
            self.activation = nn.SiLU
        else:
            print('Wrong activation')
            exit()

        # Encoder
        self.encoder_features = torch.nn.ModuleList()

        for i in range(len(encoder_features) - 1):
            self.encoder_features.append(
                torch.nn.Conv1d(
                    encoder_features[i], encoder_features[i + 1], sizes_convs_encode[i],
                    padding=sizes_convs_encode[i] // 2
                )
            )
            if batch:
                self.encoder_features.append(nn.BatchNorm1d(encoder_features[i + 1]))
            
            self.encoder_features.append(self.activation())

        self.encoder_linear = torch.nn.ModuleList()

        for i in range(len(encoder_linear) - 1):
            self.encoder_linear.append(torch.nn.Linear(encoder_linear[i], encoder_linear[i + 1]))
            if drop != False:
                self.encoder_linear.append(torch.nn.Dropout(p=drop, inplace=False))
            if act != False:
                self.encoder_linear.append( self.activation())

        # Decoder
        self.decoder_linear = torch.nn.ModuleList()

        for i in range(len(decoder_linear) - 1):
            self.decoder_linear.append(torch.nn.Linear(decoder_linear[i], decoder_linear[i + 1]))
            if drop != False:
                self.decoder_linear.append(torch.nn.Dropout(p=drop, inplace=False))
            if act != False:
                self.decoder_linear.append( self.activation())

        self.decoder_features = torch.nn.ModuleList()

        for i in range(len(decoder_features) - 1):
            self.decoder_features.append(
                torch.nn.Conv1d(
                    decoder_features[i], decoder_features[i + 1], sizes_convs_decode[i],
                    padding=sizes_convs_decode[i] // 2
                )
            )
            if batch:
                self.decoder_features.append(nn.BatchNorm1d(decoder_features[i + 1]))
            self.decoder_features.append(self.activation())

        self.last_conv = torch.nn.Conv1d(
            decoder_features[-1], 3, sizes_convs_decode[-1],
            padding=sizes_convs_decode[-1] // 2
        )

        # Downsampling mats
        self.downsampling_mats = torch.nn.ParameterList()

        k = 0

        for i, layer in enumerate(self.encoder_features):
            if isinstance(layer, self.activation):
                self.downsampling_mats.append(
                    torch.nn.Parameter(
                        torch.zeros(sizes_downsample[k], sizes_downsample[k+1]).to(opt["device"]),
                        requires_grad=True
                    )
                )

                k += 1

        self.upsampling_mats = torch.nn.ParameterList()

        k = 0

        for i, layer in enumerate(self.decoder_features):
            if isinstance(layer, torch.nn.Conv1d):
                self.upsampling_mats.append(
                    torch.nn.Parameter(
                        torch.zeros(sizes_upsample[k], sizes_upsample[k+1]).to(opt["device"]),
                        requires_grad=True
                    )
                )

                k += 1
        if self.drop_att != False:        
            self.drop_att_down = torch.nn.ModuleList()
            self.drop_att_up = torch.nn.ModuleList()
            for i in range(4):
                self.drop_att_down.append(torch.nn.Dropout(p=self.drop_att, inplace=False))
                self.drop_att_up.append(torch.nn.Dropout(p=self.drop_att, inplace=False))

    def enc(self, x):
        #print(x)
        x = x.permute(0, 2, 1)
        #print(x)
        k = 0

        for i, layer in enumerate(self.encoder_features):
            x = layer(x)
            #print("encoder_features",i,x)
            if isinstance(layer, self.activation):
                x = torch.matmul(x, self.downsampling_mats[k])
                if self.drop_att != False: 
                   x = self.drop_att_down[k](x) 
                k += 1
                #print("Downsampling",k,x)

        x = torch.flatten(x, start_dim=1, end_dim=2)
        #print("flatten",k,x)

        for i, layer in enumerate(self.encoder_linear):
            x = layer(x)
            #print("encoder_linear",i,x)

        return x

    def dec(self, x):
        for i, layer in enumerate(self.decoder_linear):
            x = layer(x)

        x = x.view(x.shape[0], -1, self.upsampling_mats[0].shape[0])

        k = 0

        for i, layer in enumerate(self.decoder_features):
            if isinstance(layer, torch.nn.Conv1d):
                x = torch.matmul(x, self.upsampling_mats[k])
                if self.drop_att != False: 
                   x = self.drop_att_up[k](x)
                k += 1

            x = layer(x)

        x = self.last_conv(x)

        x = x.permute(0, 2, 1)

        return x
    
    def forward(self,x):
        z = self.enc(x)
        x = self.dec(z)
        return x   
