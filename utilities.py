import quicklens as ql
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import flow_architecture

r2d = 180./np.pi
d2r = np.pi/180.
rad2arcmin = 180.*60./np.pi


def grab(var):
  return var.detach().cpu().numpy()


def plot_lists(list_1=None, list_2=[], idmin=0, trunc=0, ymin=None, ymax=None, offset=0, figsize=(5, 3), ylog=False, label1='', label2='', xlabel='', ylabel='', title='', file_name=None):
    idmax = len(list_1) - trunc
    fig=plt.figure(figsize=figsize)
    plt.plot(np.arange(idmin, idmax-offset, 1), list_1[idmin:idmax-offset], color='red', label=label1)
    if len(list_2) > 0: plt.plot(np.arange(idmin, idmax-offset, 1), list_2[idmin:idmax], label=label2)
    if not label1=='': plt.legend(loc=1, frameon=False, fontsize=14)
    plt.grid(True)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title)
    if ylog: ax.set_yscale('log')
    if not (ymin==None and ymax==None): plt.ylim([ymin, ymax])
    fig.tight_layout()
    if not file_name==None: plt.savefig(file_name)
    plt.show()
    

def imshow(array, vmin=None, vmax=None, figsize=(8, 8), title='', axis=True, colorbar=True, file_name=None):
    plt.figure(figsize=figsize)
    if (vmin==None and vmax==None): plt.imshow(array)
    else: plt.imshow(array, vmin=vmin, vmax=vmax)
    if not axis: plt.axis('off')
    if colorbar: plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    if not file_name==None: plt.savefig(file_name)
    
    
##################### patching maps


def make_small_maps_from_big_map(big_map, n, displace=None):
    """
    Makes small maps of shape ((N//n)**2, n, n) from big_map
    big_map must be periodic
    
    big_map: torch.tensor of shape (N, N)
    n: Length of small_map; assumes N%n == 0
    displace: If True, big_map is shifted by (n//2, n//2) pixels
    """
    N = big_map.shape[-1]
    small_maps = torch.zeros(((N//n)**2, n, n))
    
    if   displace == 1: big_map = torch.roll(big_map, n//2, dims=-2)
    elif displace == 2: big_map = torch.roll(big_map, n//2, dims=-1)
    elif displace == 3: big_map = torch.roll(big_map, (n//2, n//2), dims=(-2, -1))
    
    for i in range(N//n):
        for j in range(N//n):
            small_maps[i+(N//n)*j] = big_map[i*n:(i+1)*n, j*n:(j+1)*n]
            
    return small_maps


def make_big_map_from_small_maps(small_maps_0, small_maps_1, small_maps_2, small_maps_3, N):
    """
    Makes big_map of shape(N, N) from small maps of shape ((N//n)**2, n, n)
    
    small_maps_0: small maps
    small_maps_1: small maps displaced by n//2
    small_maps_2: small maps displaced by n//2 in the other dimension
    small_maps_3: small maps displaced by (n//2, n//2)
    N: length of big_map
    """
    n = small_maps_0.shape[-1]
    d = n//4
    
    #Shorten edges of small maps
    pad = (-d, -d, -d, -d)
    small_maps_0 = F.pad(small_maps_0, pad)
    small_maps_1 = F.pad(small_maps_1, pad)
    small_maps_2 = F.pad(small_maps_2, pad)
    small_maps_3 = F.pad(small_maps_3, pad)
    
    big_map = torch.zeros((N, N))
    
    for i in range(N//n):
        for j in range(N//n):
            big_map[i*n+d:i*n+d+n//2, j*n+d:j*n+d+n//2] = small_maps_0[i+(N//n)*j]
            
    #Roll big_map to get to small_maps_1 position
    big_map = torch.roll(big_map, n//2, dims=-2)
    for i in range(N//n):
        for j in range(N//n):
            big_map[i*n+d:i*n+d+n//2, j*n+d:j*n+d+n//2] = small_maps_1[i+(N//n)*j]
            
    #This roll is now diagonal from the original big_map, so use small_maps_3
    big_map = torch.roll(big_map, n//2, dims=-1)
    for i in range(N//n):
        for j in range(N//n):
            big_map[i*n+d:i*n+d+n//2, j*n+d:j*n+d+n//2] = small_maps_3[i+(N//n)*j]
            
    #Roll backwards to get to small_maps_2
    big_map = torch.roll(big_map, -n//2, dims=-2)
    for i in range(N//n):
        for j in range(N//n):
            big_map[i*n+d:i*n+d+n//2, j*n+d:j*n+d+n//2] = small_maps_2[i+(N//n)*j]
            
    #Go back to original big_map position
    big_map = torch.roll(big_map, -n//2, dims=-1)
    
    return big_map
    

##################### power spectrum


def estimate_ps_ensemble(samples_true, nx, dx, lbins):
    ell_binned = lbins[:-1] + 0.5*np.diff(lbins)
    
    nmaps = samples_true.shape[0]
    
    cl_avg = np.zeros(ell_binned.shape[0])
    
    for map_id in range(nmaps):
        tmap = samples_true[map_id]
        tmap_cfft = ql.maps.rmap(nx, dx, map=tmap).get_cfft()
        cl = tmap_cfft.get_cl(lbins)
        cl_avg += cl.cl.real
        
    cl_avg = cl_avg / nmaps
    
    return cl_avg, ell_binned


def get_lxly(nx, dx):
    #returns the (lx, ly) pair associated with each Fourier mode
    return np.meshgrid(np.fft.fftfreq(nx, dx)*2.*np.pi,np.fft.fftfreq(nx, dx)*2.*np.pi)


def get_ell_squared(nx, dx):
    #returns the wavenumber l = lx**2 + ly**2 for each Fourier mode
    lx, ly = get_lxly(nx, dx)
    return lx**2 + ly**2


def add_noise(array, std=1.):
    """ Adds std noise per pixel to a 2d np.ndarray """
    batch_size = np.shape(array)[0]
    L = np.shape(array)[1]
    W = np.shape(array)[2]
    noise_prior = flow_architecture.SimpleNormal(torch.zeros((L, W)), torch.ones((L, W))*std)
    noise = grab(noise_prior.sample_n(batch_size))
    return array + noise


def make_cross_correlation_plots(params, y_true, y_pred_flow, y_pred_wf, y_masked, save_dir, nbins=50):
    nx = params.nx
    dx = params.dx
    lmax = params.lmax * 0.95
    #lbins = np.linspace(100, lmax, nbins)
    lbins = np.logspace(2, np.log10(lmax), nbins)
    ell_binned = lbins[:-1] + np.diff(lbins)
    nsims = y_true.shape[0]
    
    y_noise = y_masked - y_true

    #mapWF
    corr_coeff_flow_avg = np.zeros(ell_binned.shape[0])
    corr_coeff_wf_avg = np.zeros(ell_binned.shape[0])
    corr_coeff_masked_avg = np.zeros(ell_binned.shape[0])
    auto_flow_avg = np.zeros(ell_binned.shape[0])
    auto_wf_avg = np.zeros(ell_binned.shape[0])
    auto_masked_avg = np.zeros(ell_binned.shape[0])
    auto_noise_avg = np.zeros(ell_binned.shape[0])
    auto_true_avg = np.zeros(ell_binned.shape[0])
    diffpower_flow_avg = np.zeros(ell_binned.shape[0])
    diffpower_wf_avg = np.zeros(ell_binned.shape[0])
    diffpower_masked_avg = np.zeros(ell_binned.shape[0])

    for map_id in range(nsims): 
        #make these rmaps and get cffts from which we can get cls and mls
        map_true_cfft   = ql.maps.rmap(nx, dx, map=y_true[map_id]).get_cfft()
        map_flow_cfft   = ql.maps.rmap(nx, dx, map=y_pred_flow[map_id]).get_cfft()
        map_wf_cfft     = ql.maps.rmap(nx, dx, map=y_pred_wf[map_id]).get_cfft()
        map_masked_cfft = ql.maps.rmap(nx, dx, map=y_masked[map_id]).get_cfft()
        map_noise_cfft  = ql.maps.rmap(nx, dx, map=y_noise[map_id]).get_cfft()

        #cross powers
        cross_map_cfft_flow = ql.maps.cfft(nx, dx, fft=(map_flow_cfft.fft * np.conj(map_true_cfft.fft)))
        cross_power_flow = cross_map_cfft_flow.get_ml(lbins)  #use ml because the cfft already is a power/multiple of two maps    
        cross_map_cfft_wf = ql.maps.cfft(nx, dx, fft=(map_wf_cfft.fft * np.conj(map_true_cfft.fft)))
        cross_power_wf = cross_map_cfft_wf.get_ml(lbins)  #use ml because the cfft already is a power/multiple of two maps   
        cross_map_cfft_masked = ql.maps.cfft(nx, dx, fft=(map_masked_cfft.fft * np.conj(map_true_cfft.fft)))
        cross_power_masked = cross_map_cfft_masked.get_ml(lbins)  #use ml because the cfft already is a power/multiple of two maps   

        #auto powers
        auto_true = map_true_cfft.get_cl(lbins) #use cl because we really want the power of this map
        auto_flow = map_flow_cfft.get_cl(lbins)
        auto_wf = map_wf_cfft.get_cl(lbins)
        auto_masked = map_masked_cfft.get_cl(lbins)
        auto_noise = map_noise_cfft.get_cl(lbins)
        auto_true_avg += auto_true.cl.real
        auto_flow_avg += auto_flow.cl.real
        auto_wf_avg += auto_wf.cl.real
        auto_masked_avg += auto_masked.cl.real
        auto_noise_avg += auto_noise.cl.real

        #corr coeff from spectra
        corr_coeff_flow = cross_power_flow.specs['cl'] / ((auto_flow.specs['cl']*auto_true.specs['cl'])**(1./2))
        corr_coeff_flow_avg += corr_coeff_flow.real
        corr_coeff_wf = cross_power_wf.specs['cl'] / ((auto_wf.specs['cl']*auto_true.specs['cl'])**(1./2))
        corr_coeff_wf_avg += corr_coeff_wf.real
        corr_coeff_masked = cross_power_masked.specs['cl'] / ((auto_masked.specs['cl']*auto_true.specs['cl'])**(1./2))
        corr_coeff_masked_avg += corr_coeff_masked.real

        #difference maps
        diff_flow_cfft = map_flow_cfft - map_true_cfft  
        diff_wf_cfft = map_wf_cfft - map_true_cfft 
        diff_masked_cfft = map_masked_cfft - map_true_cfft
        diffpower_flow = diff_flow_cfft.get_cl(lbins) #use cl because we really want the power of this map
        diffpower_wf = diff_wf_cfft.get_cl(lbins)
        diffpower_masked = diff_masked_cfft.get_cl(lbins)
        diffpower_flow_avg += diffpower_flow.cl.real
        diffpower_wf_avg += diffpower_wf.cl.real  
        diffpower_masked_avg += diffpower_masked.cl.real

    #averages
    corr_coeff_flow_avg = corr_coeff_flow_avg / nsims
    corr_coeff_wf_avg = corr_coeff_wf_avg / nsims
    corr_coeff_masked_avg = corr_coeff_masked_avg / nsims
    auto_flow_avg = auto_flow_avg / nsims
    auto_true_avg = auto_true_avg / nsims
    auto_wf_avg = auto_wf_avg / nsims
    auto_masked_avg = auto_masked_avg / nsims
    auto_noise_avg = auto_noise_avg / nsims
    diffpower_flow_avg = diffpower_flow_avg / nsims
    diffpower_wf_avg = diffpower_wf_avg / nsims
    diffpower_masked_avg = diffpower_masked_avg / nsims
    
    #Change from l to k
    k_min = 2*3.14159 / 128
    k_max = 32*k_min
    kbins = np.logspace(np.log10(k_min), np.log10(k_max), nbins)
    k_binned = kbins[:-1] + np.diff(kbins)
    
    xlim_min = .23
    xlim_max = 1.74
    
    fig=plt.figure(figsize=(12, 8))
    ax1=fig.add_subplot(221)
    ax1.plot(k_binned, corr_coeff_flow_avg, color='red', label='$r_{\ \mathrm{True,Flow}}$')
    ax1.plot(k_binned, corr_coeff_wf_avg, color='blue', label='$r_{\ \mathrm{True,WF}}$')
    ax1.plot(k_binned, corr_coeff_masked_avg, color='green', label='$r_{\ \mathrm{True,Masked}}$')
    plt.legend(frameon=False, fontsize=14)
    ax1.set_ylabel('r', fontsize=20)
    ax1.set_xscale('log') ####
    ax1.set_xlim(xlim_min, xlim_max)
    ax2=fig.add_subplot(223)
    ax2.plot(k_binned, 1-corr_coeff_flow_avg, color='red', label='$r_{\ \mathrm{True,Flow}}$')
    ax2.plot(k_binned, 1-corr_coeff_wf_avg, color='blue', label='$r_{\ \mathrm{True,WF}}$')
    ax2.plot(k_binned, 1-corr_coeff_masked_avg, color='green', label='$r_{\ \mathrm{True,Masked}}$')
    ax2.set_xlim(xlim_min, xlim_max)
    ax2.set_xscale('log') ####
    ax2.set_yscale('log')
    ax2.set_ylabel('1-r', fontsize=20)
    ax2.set_xlabel('$k(h\mathrm{Mpc}^{-1})$', fontsize=20)
    plt.legend(frameon=False, fontsize=14)

    #cl power
    ax = fig.add_subplot(222)
    ax.plot(k_binned, auto_flow_avg*k_binned**2., color='red', label='$C_\ell^{\mathrm{Flow}}$')
    ax.plot(k_binned, auto_true_avg*k_binned**2., color='black', label=r'$C_\ell^{\mathrm{True}}$')
    ax.plot(k_binned, auto_wf_avg*k_binned**2., color='blue', label=r'$C_\ell^{\mathrm{WF}}$')
    ax.plot(k_binned, auto_masked_avg*k_binned**2., color='green', label=r'$C_\ell^{\mathrm{Masked}}$')
    ax.plot(k_binned, auto_noise_avg*k_binned**2., '--', color='green', label=r'$N_\ell$')
    ax.set_ylabel('$P(k)(h^{-1}\mathrm{Mpc})^3$', fontsize=20)
    ax.set_xlim(xlim_min, xlim_max)
    #ax.set_ylim(0.007, 5.) ####
    plt.legend(loc='lower center', frameon=False, fontsize=14)
    ax.set_xscale('log') ####
    ax.set_yscale('log')

    #diff power
    ax2=fig.add_subplot(224)
    ax2.plot(k_binned, diffpower_flow_avg / auto_true_avg, color='red', label='$\Delta_\ell^{Flow}$')
    ax2.plot(k_binned, diffpower_wf_avg / auto_true_avg, color='blue', label=r'$\Delta_\ell^{WF}$')
    ax2.plot(k_binned, diffpower_masked_avg / auto_masked_avg, color='green', label=r'$\Delta_\ell^{Masked}$')
    ax2.set_xscale('log') ####
    ax2.set_yscale('log')
    ax2.set_xlim(xlim_min, xlim_max)
    #ax2.set_ylim([0.001,10])
    ax2.set_xlabel('$k(h\mathrm{Mpc}^{-1})$', fontsize=20)
    ax2.set_ylabel('$\Delta_k$', fontsize=20)
    plt.legend(frameon=False, fontsize=14)
    fig.tight_layout()
    plt.show()

    fig.savefig(save_dir+'/quality_measures_t.pdf')

####################### phi / kappa

def kappa_to_phi(kappa, dx, method='fft'):
    #Takes position space numpy array, returns position space numpy array
    nx = np.shape(kappa)[-1]
    ell_squared = get_ell_squared(nx, dx)
    ell_squared[ell_squared==0] = 0.01
    inv_ell_squared = 1. / ell_squared
    kappa_fft = np.fft.ifft2(kappa)
    phi_fft = -2 * inv_ell_squared * kappa_fft
    phi = np.fft.fft2(phi_fft)
    return phi.real

def phi_to_kappa(phi, dx, method='fft'):
    #Takes position space numpy array, returns position space numpy array
    nx = np.shape(phi)[-1]
    ell_squared = get_ell_squared(nx, dx)
    phi_fft = np.fft.fft2(phi)
    kappa_fft = -0.5 * ell_squared * phi_fft
    kappa = np.fft.ifft2(kappa_fft)
    return kappa.real

###################### local non-gaussianity

def estimate_fnl_local_ensemble(samples_nongauss, samples_gauss, cl_theo, nx, dx):
    #estimates fnl local on a set of maps. you need to provide also gaussian maps which are used to estimate the variance
    #both sets need to be equal size and we need at least 1000 maps to make F converge sufficiently
    cl_theo_inv_nonan = np.copy(cl_theo) 
    #cl_theo_inv_nonan[0]=1.
    #cl_theo_inv_nonan[1]=1.
    cl_theo_inv_nonan[cl_theo_inv_nonan<0.00001] = 0.00001
    
    cl_theo_inv_nonan = 1./cl_theo_inv_nonan 
    
    fnl_unnormed_nongauss = np.zeros(samples_gauss.shape[0])
    fnl_unnormed_gauss = np.zeros(samples_gauss.shape[0])
    for i in range(samples_gauss.shape[0]):
        rmap = samples_nongauss[i]
        fnl_unnormed_nongauss[i] = estimate_fnl_local_unnormed_singlemap(rmap, cl_theo_inv_nonan, nx, dx)

        rmap = samples_gauss[i] 
        fnl_unnormed_gauss[i] = estimate_fnl_local_unnormed_singlemap(rmap, cl_theo_inv_nonan, nx, dx)

    F = np.var(fnl_unnormed_gauss)
    fnl_normed_nongauss = fnl_unnormed_nongauss/F
    fnl_normed_gauss = fnl_unnormed_gauss/F
    return fnl_normed_nongauss,fnl_normed_gauss
    
    
    
def estimate_fnl_local_unnormed_singlemap(rmap, cl_theo_inv_nonan, nx, dx):
    #https://github.com/dhanson/quicklens/blob/master/quicklens/maps.py
    #FT conventions as in np.rfft2 irfft2
    rmap = ql.maps.rmap(nx, dx,map=rmap)
    rfft = rmap.get_rfft()
    
    #A map
    A = rmap.copy()
    
    #B map
    rfft_B = rfft * cl_theo_inv_nonan
    B = rfft_B.get_rmap()
    
    #multiply and integrate
    fnl_unnormed = (1.)*np.sum(A.map*A.map*B.map) * (dx**2) 
    
    return fnl_unnormed

###################### equilateral non-gaussianity

def estimate_fnl_equilateral_ensemble(samples_nongauss, samples_gauss, cl_theo_ell, nx, dx):
    ell_nonan = np.copy(cl_theo_ell) 
    ell_nonan[0]=1.

    fnlequi_unnormed_nongauss = np.zeros(samples_gauss.shape[0])
    fnlequi_unnormed_gauss = np.zeros(samples_gauss.shape[0])
    for i in range(samples_gauss.shape[0]):
        rmap = samples_nongauss[i]
        fnlequi_unnormed_nongauss[i] = estimate_fnl_equilateral_unnormed_singlemap(rmap,ell_nonan, nx, dx)
    
        rmap = samples_gauss[i] 
        fnlequi_unnormed_gauss[i] = estimate_fnl_equilateral_unnormed_singlemap(rmap,ell_nonan, nx, dx)
  
    Fequi = np.var(fnlequi_unnormed_gauss)
    fnlequi_normed_nongauss = fnlequi_unnormed_nongauss/Fequi
    fnlequi_normed_gauss = fnlequi_unnormed_gauss/Fequi

    return fnlequi_normed_nongauss,fnlequi_normed_gauss

    
def estimate_fnl_equilateral_unnormed_singlemap(rmap,ell_nonan, nx, dx):
    rmap = ql.maps.rmap(nx, dx,map=rmap)
    rfft = rmap.get_rfft()
    
    #A map
    rfft_A = rfft * np.power(ell_nonan,8/3)
    A = rfft_A.get_rmap()
    
    #B map
    rfft_B = rfft * np.power(ell_nonan,-1/3)
    B = rfft_B.get_rmap()
    
    #C map
    rfft_C = rfft * np.power(ell_nonan,5/3)
    C = rfft_C.get_rmap()
    
    #D map
    rfft_D = rfft * np.power(ell_nonan,2/3)
    D = rfft_D.get_rmap()
    
    #multiply and integrate
    As = 1. #placeholder
    fnl_unnormed = (1./(As))*np.sum(-3*A.map*B.map*B.map + 6*B.map*C.map*D.map - 2*D.map*D.map*D.map) * (dx**2) 
    
    return fnl_unnormed