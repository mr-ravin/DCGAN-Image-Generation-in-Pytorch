import torch.nn as nn

class DCGAN_Generator(nn.Module): # input image size: 3 x 64 x 64, here 3 represents number of channels i.e. r,g,b
    def __init__(self, enc_input_channels=3, enc_output_channels=8, noise_dim=512, mode="analysis"):
        super(DCGAN_Generator,self).__init__()
        self.mode = mode
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.gen_deconv5 = nn.ConvTranspose2d(
            in_channels=noise_dim, out_channels=enc_output_channels*16, kernel_size=4, 
            stride=1, bias= True, padding = 0
        )
        self.gen_deconv4 = nn.ConvTranspose2d(
            in_channels=enc_output_channels*16, out_channels=enc_output_channels*8, kernel_size=4, 
            stride=2, padding = 1
        )

        self.gen_deconv3 = nn.ConvTranspose2d(
            in_channels=enc_output_channels*8, out_channels=enc_output_channels*4, kernel_size=4, 
            stride=2, padding = 1
        )
        self.gen_deconv2 = nn.ConvTranspose2d(
            in_channels=enc_output_channels*4, out_channels=enc_output_channels*2, kernel_size=4, 
            stride=2, padding = 1
        )
        self.gen_deconv1 = nn.ConvTranspose2d(
            in_channels=enc_output_channels*2, out_channels=enc_input_channels, kernel_size=4, 
            stride=2, bias = True, padding = 1
        )
    

    def forward(self,x, batch, noise_dim):
        x = x.view(batch, noise_dim, 1, 1)
        if self.mode=="analysis":
            print("########")
        if self.mode=="analysis":
            print("input dconv5: ",x.shape)
        x = self.activation(self.gen_deconv5(x))
        if self.mode=="analysis":
            print("output dconv5: ",x.shape)
            print("input dconv4: ",x.shape)
        x = self.activation(self.gen_deconv4(x))
        if self.mode=="analysis":
            print("output dconv4: ",x.shape)
            print("input dconv3: ",x.shape)
        x = self.activation(self.gen_deconv3(x))
        if self.mode=="analysis":
            print("output dconv3: ",x.shape)
            print("input dconv2: ",x.shape)
        x = self.activation(self.gen_deconv2(x))
        if self.mode=="analysis":
            print("output dconv2: ",x.shape)
            print("input dconv1: ",x.shape)
        x = self.gen_deconv1(x)
        if self.mode=="analysis":
            print("output dconv1: ",x.shape)
        reconstruction = self.sigmoid(x)
        return reconstruction


class DCGAN_Discriminator(nn.Module): # input image size: 3 x 64 x 64, here 3 represents number of channels i.e. r,g,b
    def __init__(self, enc_input_channels=3, enc_output_channels=4, mode="analysis"):
        super(DCGAN_Discriminator,self).__init__()
        self.mode = mode
        self.activation = nn.ReLU()
        self.disc_conv1 = nn.Conv2d(enc_input_channels, enc_output_channels*2, kernel_size=4, stride=2, bias = True, padding = 1)
        self.disc_conv2 = nn.Conv2d(enc_output_channels*2, enc_output_channels*4, kernel_size=4,stride=2, padding = 1)
        self.disc_conv3 = nn.Conv2d(enc_output_channels*4, enc_output_channels*8, kernel_size=4,stride=2, padding = 1)
        self.disc_conv4 = nn.Conv2d(enc_output_channels*8, enc_output_channels*16, kernel_size=4,stride=2, padding = 1)
        self.disc_conv5 = nn.Conv2d(enc_output_channels*16, enc_output_channels*32, kernel_size=4,stride=1, padding = 0, bias = True)
        self.fc0 = nn.Linear(enc_output_channels*32, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        if self.mode=="analysis":
            print("input conv1: ",x.shape)
        x = self.activation(self.disc_conv1(x))
        if self.mode=="analysis":
            print("output conv1: ",x.shape)
            print("input conv2: ",x.shape)
        x = self.activation(self.disc_conv2(x))
        if self.mode=="analysis":
            print("output conv2: ",x.shape)
            print("input conv3: ",x.shape)
        x = self.activation(self.disc_conv3(x))
        if self.mode=="analysis":
            print("output conv3: ",x.shape)
            print("input conv4: ",x.shape)
        x = self.activation(self.disc_conv4(x))
        if self.mode=="analysis":
            print("output conv4: ",x.shape)
            print("input conv5: ",x.shape)
        x = self.activation(self.disc_conv5(x))
        if self.mode=="analysis":
            print("output conv5: ",x.shape)
            print("########")
        batch, channel, row, col = x.shape
        x = x.view(batch, -1)
        if self.mode=="analysis":
            print(batch, channel, row, col)
            print("size after flatten: ", x.shape)
        x = self.fc0(x)
        x = self.sigmoid(x)
        return x
