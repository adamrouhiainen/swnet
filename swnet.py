import fire
import time
import torch
import torch.nn.functional as F
import math
import numpy
import logging
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
#from sigmafnl.architecture.bimodal import dataloader
import dataloader


logging.getLogger("lightning").setLevel(logging.ERROR)
torch.set_default_dtype(torch.float32)
torch.manual_seed(232882)

channel_num=16


nx = 576//3

#Make k_cubed_tensor
freqs = numpy.fft.fftfreq(nx)*2.*numpy.pi
kx, ky, kz = numpy.meshgrid(freqs, freqs, freqs)
k_cubed_tensor = torch.tensor((kx**2 + ky**2 + kz**2)**(3/2), device=torch.device('cuda'))

#Set monopole value to dipole to avoid dividing by 0
k_cubed_min = (freqs[0]**2 + freqs[0]**2 + freqs[1]**2)**(3/2)
k_cubed_tensor[k_cubed_tensor == 0.] = k_cubed_min


def make_low_pass_filter(l, lp_factor):
    tensor = torch.ones((l, l, l))
    output = torch.zeros((l, l, l))
    for i in range(l):
        for j in range(l):
            for k in range(l):
                if numpy.sqrt(i**2+j**2+k**2)<=(l/lp_factor):
                    output[i, j, k] = tensor[i, j, k]
                    output[l-i-1, j, k] = tensor[l-i-1, j, k]
                    output[i, l-j-1, k] = tensor[i, l-j-1, k]
                    output[i, j, l-k-1] = tensor[i, j, l-k-1]
                    output[l-i-1, l-j-1, k] = tensor[l-i-1, l-j-1, k]
                    output[l-i-1, j, l-k-1] = tensor[l-i-1, j, l-k-1]
                    output[i, l-j-1, l-k-1] = tensor[i, l-j-1, l-k-1]
                    output[l-i-1, l-j-1, l-k-1] = tensor[l-i-1, l-j-1, l-k-1]
    return output


low_pass_filter = make_low_pass_filter(nx, 16).to(torch.device('cuda'))


loss_list = []



class ResNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size_1, kernel_size_2, bias):
        super(ResNet, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels, hidden_channels,  kernel_size=kernel_size_1, bias=bias, padding='same', padding_mode='zeros')
        self.conv2 = torch.nn.Conv3d(hidden_channels, out_channels, kernel_size=kernel_size_2, bias=bias, padding='same', padding_mode='zeros')
        
        self.bias = torch.nn.Parameter(torch.randn(1)*0.1+1.)
        
        
    def forward(self, x):
        x_skip = x
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.conv2(x)
        return 0.001*self.bias*x + x_skip
        


class net(pl.LightningModule):

    def __init__(self, final_ReLU=False, final_bias=True):
        super(net, self).__init__()
        self.resnet12 = ResNet(1,           channel_num, channel_num, 5, 1, False)
        self.resnet56 = ResNet(channel_num, channel_num, channel_num, 3, 1, False)
        self.resnet78 = ResNet(channel_num, channel_num, channel_num, 3, 1, False)
        self.final_conv = torch.nn.Conv3d(channel_num, 1, kernel_size=1, bias=False, padding='same', padding_mode='zeros')
        
        self.final_ReLU = final_ReLU
        if final_bias:
            self.final_bias = True
            self.bias = torch.nn.Parameter(torch.tensor((1.0)))
        
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.resnet12(x))
        x = torch.nn.functional.relu(self.resnet56(x))
        x = torch.nn.functional.relu(self.resnet78(x))
        x = self.final_conv(x)
        
        if self.final_ReLU: x = torch.nn.functional.relu(x)
        if self.final_bias: x = self.bias*x
        
        return x

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

        return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                                            "scheduler": scheduler,
                                            "monitor": "loss",
                                    },
                }
        #return optimizer
    
    
    def lossfn(self, output, target):
        """
        In Fourier space, scaled by 1/k**3
        """
        output_fft = torch.fft.fft(output.squeeze().type(torch.float32))
        target_fft = torch.fft.fft(target.squeeze())
        diff = output_fft - target_fft
        
        diff = diff*low_pass_filter
        
        loss = torch.mean(torch.real(torch.conj(diff)*diff)/k_cubed_tensor)
        
        loss_list.append(loss.detach().cpu().numpy())
        
        numpy.save('loss_list', loss_list)
        numpy.save('loss_list_backup', loss_list)
                        
        return loss


    def training_step(self, train_batch, batch_idx):
        snapshot, parameters = train_batch
        output = model(snapshot)
        loss = self.lossfn(output, parameters)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}


    
model = net()


def main(channel_num=1, batchsize=1, padding=0, basepath=None, checkpoint='checkpoint'):
    datalist = dataloader.datalist(basepath=basepath)[:400]
    dataset = dataloader.SnapshotDataset(datalist, padding=padding, transform=dataloader.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, num_workers=8)
    
    trainer = pl.Trainer(accelerator='gpu', devices=1, strategy=DDPPlugin(find_unused_parameters=False),
                         log_every_n_steps=450, max_epochs=600, precision=16)
    
    trainer.fit(model, data_loader) #, ckpt_path='checkpoint_1663704869.ckpt')
    trainer.save_checkpoint(f"{checkpoint}_{int(time.time())}.ckpt")


if '__main__' == __name__:
    fire.Fire(main)
