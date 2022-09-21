import fire
import time
import torch
import torch.nn.functional as F
import math
import numpy
import scipy.stats
import Pk_library as PKL
import matplotlib.pyplot as plt
import logging
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
#from sigmafnl.architecture.bimodal import dataloader
import dataloader_eval as dataloader

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
        
        loss = torch.mean(torch.real(torch.conj(diff)*diff)/k_cubed_tensor)
                        
        return loss


    def training_step(self, train_batch, batch_idx):
        snapshot, parameters = train_batch
        output = model(snapshot)
        loss = self.lossfn(output, parameters)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}



model = net()


def make_overdensity(tensor):
    overdensity = numpy.zeros(tensor.shape, dtype='float32')
    
    for b in range(tensor.shape[0]):
        tensor_mean = numpy.mean(tensor[b], axis=(-3, -2, -1))
        overdensity[b] = (tensor[b] - tensor_mean) / tensor_mean
        
    return overdensity


BoxSize = 2000./3. #Mpc/h
axis = 0
MAS = None
MAS2 = [None, None]
threads = 1
verbose = False


def main(channel_num=16, padding=0, basepath=None, checkpoint='checkpoint'):
    datalist = dataloader.datalist(basepath=basepath)[:400]
    batchsize = len(datalist)
    dataset = dataloader.SnapshotDataset(datalist, padding=padding, transform=dataloader.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, num_workers=8)
    
    dataiter = iter(data_loader)
    x, deltam = dataiter.next()

    model_loaded = model.load_from_checkpoint('checkpoint_1663789558.ckpt')
    model_loaded.eval()
    
    y = model_loaded(x)

    #Convert torch tensors to numpy arrays
    x = x.squeeze(1).cpu().detach().numpy().astype('float32') #[padding:-padding, padding:-padding, padding:-padding]
    y = y.squeeze(1).cpu().detach().numpy().astype('float32')
    deltam = deltam.cpu().detach().numpy().astype('float32')
    
    #x = make_overdensity(x) #Halo number density was normalized to mean = 0
    y = make_overdensity(y)
    #deltam = make_overdensity(deltam) #Primordial deltam has mean = 0
    
    #Plot power spectrums
    n_maps = x.shape[0]
    x_Abins_mean = 0
    y_Abins_mean = 0
    m_Abins_mean = 0
    noise_nn_Abins_mean = 0
    noise_b_Abins_mean = 0
    x_m_Abins_mean = 0
    y_m_Abins_mean = 0
    
    with torch.no_grad():
        for i in range(n_maps):
            Pk_x_m = PKL.XPk([x[i], deltam[i]], BoxSize, axis, MAS2, threads)
            Pk_y_m = PKL.XPk([y[i], deltam[i]], BoxSize, axis, MAS2, threads)
            x_Abins_mean += Pk_x_m.Pk[:, 0, 0] / n_maps
            y_Abins_mean += Pk_y_m.Pk[:, 0, 0] / n_maps
            m_Abins_mean += Pk_y_m.Pk[:, 0, 1] / n_maps
            x_m_Abins_mean += Pk_x_m.XPk[:, 0, 0] / n_maps
            y_m_Abins_mean += Pk_y_m.XPk[:, 0, 0] / n_maps
    
        kvals = Pk_x_m.k3D

        b = numpy.sqrt(x_Abins_mean[1] / m_Abins_mean[1])
        
        for i in range(n_maps):
            Pk_NNnoise = PKL.Pk(y[i]-deltam[i],   BoxSize, axis, MAS, threads)
            Pk_bnoise  = PKL.Pk(x[i]/b-deltam[i], BoxSize, axis, MAS, threads)
            noise_nn_Abins_mean += Pk_NNnoise.Pk[:, 0] / n_maps
            noise_b_Abins_mean += Pk_bnoise.Pk[:, 0]   / n_maps
    
    #Power spectrum
    plt.loglog(kvals[:-1], x_Abins_mean[:-1],        label='Halo number density')
    plt.loglog(kvals[:-1], y_Abins_mean[:-1],        label='Network delta_m')
    plt.loglog(kvals[:-1], m_Abins_mean[:-1],        label='Truth delta_m')
    plt.loglog(kvals[:-1], noise_nn_Abins_mean[:-1], label='N network') #Should be ~1/n
    plt.loglog(kvals[:-1], noise_b_Abins_mean[:-1],  label=f'N for b={b}')
    plt.legend(loc='best')
    plt.xlabel('k (Mpc/h)')
    plt.ylabel('PS')
    plt.savefig('ps_plots_mean_PKL.png')
    plt.clf()
    
    #Plot projection maps
    vmin = -0.5
    vmax = 2.1
    
    plt.imshow(numpy.mean(x[0], -1))
    plt.colorbar()
    plt.title('Halo number density')
    plt.savefig('input_sample_1.png')
    plt.clf()
    
    plt.imshow(numpy.mean(y[0], -1)) #, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Network delta_m')
    plt.savefig('output_sample_1.png')
    plt.clf()
    
    plt.imshow(numpy.mean(deltam[0], -1)) #, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Truth delta_m')
    plt.savefig('deltam_sample_1.png')
    plt.clf()
    
    plt.imshow(numpy.mean(y[0]-deltam[0], -1))
    plt.colorbar()
    plt.title('Network noise')
    plt.savefig('network_noise.png')
    plt.clf()
    
    plt.imshow(numpy.mean(x[0]/b - deltam[0], -1))
    plt.colorbar()
    plt.title(f'b={b} noise')
    plt.savefig(f'{b}_noise.png')
    plt.clf()
    
    #Fourier cross-correlation plots
    x_m_CC_mean = x_m_Abins_mean / numpy.sqrt(x_Abins_mean*m_Abins_mean)
    y_m_CC_mean = y_m_Abins_mean / numpy.sqrt(y_Abins_mean*m_Abins_mean)
    plt.plot(kvals[:-1], y_m_CC_mean[:-1], label='r_(truth,network)')
    plt.plot(kvals[:-1], x_m_CC_mean[:-1], label='r_(truth,halo)')
    #plt.xlim(numpy.min(kvals), 0.2)
    #plt.ylim(0.75, 1.005)
    plt.xscale('log')
    plt.xlabel('k(Mpc/h)')
    plt.ylabel('r')
    plt.legend()
    plt.savefig('CC_3D_mean.pdf')
    plt.clf()
    
    plt.plot(kvals[:-1], 1 - y_m_CC_mean[:-1], label='(1-r)_(truth,network)')
    plt.plot(kvals[:-1], 1 - x_m_CC_mean[:-1], label='(1-r)_(truth,halo)')
    #plt.xlim(numpy.min(kvals), 0.2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k(Mpc/h)')
    plt.ylabel('1-r')
    plt.legend()
    plt.savefig('1_minus_CC_3D_mean.pdf')
    plt.clf()
    
    
if '__main__' == __name__:
    fire.Fire(main)