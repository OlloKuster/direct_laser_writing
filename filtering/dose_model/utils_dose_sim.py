import torch
import numpy as np
from scipy.special import jv
from typing import Any
import logging
import os
from scipy.ndimage import rotate
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad_kernel(kernel: torch.Tensor, tensor_shape: torch.Size) -> torch.Tensor:
    """
    For the convolution in the frequency domain, the kernel has to be padded to the size of the object such that they can be multiplied in the frequency domain.
    This function does the padding of the kernel depending on the shape of the tensor.

    Args:
    kernel (torch.Tensor): 3D convolution kernel. The last three dimensions are the spatial dimensions.
    tensor_shape (torch.Size): Shape of the object.

    Returns:
    torch.Tensor: Padded kernel.
    """
    depth_pad = (tensor_shape[-2] - kernel.shape[-2]) // 2
    height_pad = (tensor_shape[-1] - kernel.shape[-1]) // 2
    width_pad = (tensor_shape[-3] - kernel.shape[-3]) // 2

    # If the differences are odd, additional padding might be needed on one side
    depth_pad_extra = (tensor_shape[-2] - kernel.shape[-2]) % 2
    height_pad_extra = (tensor_shape[-1] - kernel.shape[-1]) % 2
    width_pad_extra = (tensor_shape[-3] - kernel.shape[-3]) % 2

    # It is no bug that height and width are exchanged! The pytorch function does not work as one expects it
    pad = [height_pad, height_pad + height_pad_extra, depth_pad, depth_pad + depth_pad_extra, width_pad,
           width_pad + width_pad_extra]

    kernel = torch.nn.functional.pad(kernel, pad)

    return kernel


def fft_convolution(obj: torch.Tensor, padded_kernel: torch.Tensor) -> torch.Tensor:
    """
    Translates the object and the kernel into the frequency domain and performs the convolution (multiplication) in the frequency domain.
    Finally, the result is transformed back into the spatial domain.
    The kernel needs to be padded to the size of the object such that they can be multiplied in the frequency domain.

    Args:
    obj (torch.Tensor): 3D object. The last three dimensions are the spatial dimensions.
    padded_kernel (torch.Tensor): Padded 3D convolution kernel. The last three dimensions are the spatial dimensions.

    Returns:
    torch.Tensor: 3D convolution result. The last three dimensions are the spatial dimensions.
    """

    # Fouriertransform the padded kernel and the object
    object_fft = torch.fft.rfftn(torch.fft.fftshift(obj))
    kernel_fft = torch.fft.rfftn(torch.fft.fftshift(padded_kernel))

    # Intermediate Tensor
    interm_tensor = torch.fft.ifftshift(torch.fft.irfftn(object_fft * kernel_fft)).real
    return interm_tensor


def calc_dose(interm_tensor: torch.Tensor,
              focus_intensity: torch.Tensor,
              v: float = 1.0,
              r_p: float = 0.8,
              t_p: float = 1.4,
              p_th: float = 4.4e-3,
              v_th=9.88e-2,
              lp: float = 20e-3,
              res_lat: float = 50e-9,
              res_ax: float = 50e-9,
              r_r: float = 4e-6,
              r_z: float = 10e-6,
              lam: float = 790e-9,
              n: float = 1.52,
              nonlin: float = 3,
              M: float = 40,
              NA: float = 1.4,
              tubus: float = 165e-3,
              my_version: bool = False
              # d_th: float = 1,
              ) -> torch.Tensor:
    """
    res = 50e-9
    r_r = 4e-6
    r_z = 10e-6
    x = np.linspace(-r_r, r_r, int(round(2*r_r/res))+1)
    absEx = np.abs(Ex - 1 / np.exp(2))
    idx = np.where(absEx == absEx.min())[0]
    r = x[idx]
    """
    w = NA * tubus / M
    f = tubus / M

    ex = focus_intensity[:, focus_intensity.shape[1] // 2, focus_intensity.shape[2] // 2] / focus_intensity.max()
    # interm_tensor[:, interm_tensor.shape[1]//2, interm_tensor.shape[2]//2]/interm_tensor.max()

    x = torch.linspace(-r_r, r_r, int(round(2 * r_r / res_lat)) + 1, device=ex.device)

    abs_ex = torch.abs(ex - 1 / np.exp(2))
    idx = torch.argmin(abs_ex)
    # idx = torch.where(abs_ex == abs_ex.min())[0]
    r = x[idx]
    print('r = ', r)

    if not my_version:
        dose = interm_tensor * v_th / v * ((np.pi * f / lam) ** 2 * n * lp / p_th * (r / w) ** 2) ** nonlin
    else:
        dose = interm_tensor * v_th / v * (lp / p_th) ** nonlin

    # obj = dose > d_th

    return dose


def calc_dose_obj(obj: torch.Tensor, intensity_kernel: torch.Tensor, intensity_kernel_pad: torch.Tensor = None,
                  v: float = 1.0,
                  r_p: float = 0.8,
                  t_p: float = 1.4,
                  p_th: float = 4.4e-3,
                  v_th=9.88e-2,
                  lp: float = 20e-3,
                  res_lat: float = 50e-9,
                  res_ax: float = 50e-9,
                  r_r: float = 4e-6,
                  r_z: float = 10e-6,
                  lam: float = 790e-9,
                  n: float = 1.52,
                  nonlin: float = 2,
                  M: float = 40,
                  NA: float = 1.4,
                  tubus: float = 165 - 3,
                  my_version: bool = False
                  ) -> torch.Tensor:
    if intensity_kernel_pad is None:
        intensity_kernel_pad = pad_kernel(intensity_kernel, obj.shape)
    interm_tensor = fft_convolution(obj, intensity_kernel_pad ** nonlin)

    dose = calc_dose(interm_tensor, intensity_kernel, v, r_p, t_p, p_th, v_th, lp, res_lat, res_ax, r_r, r_z, lam, n,
                     nonlin, M, NA, tubus, my_version)

    return dose


class BesselJ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        ctx.save_for_backward(x)
        ctx.n = n
        return torch.from_numpy(jv(n, x.cpu().numpy())).to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        n = ctx.n
        grad_x = grad_output * 0.5 * (jv(n - 1, x.cpu().numpy()) - jv(n + 1, x.cpu().numpy()))
        return torch.from_numpy(grad_x).to(x.device), None


def calc_laser_intensity(lam: torch.Tensor = torch.tensor(780e-9), n: torch.Tensor = torch.tensor(1.483),
                         NA: torch.Tensor = torch.tensor(0.8),
                         M: torch.Tensor = torch.tensor(25.0),
                         tubus: torch.Tensor = torch.tensor(165.0e-3), r_r: torch.Tensor = torch.tensor(3.2e-6),
                         r_z: torch.Tensor = torch.tensor(4.8e-6),
                         res_ax: torch.Tensor = torch.tensor(300e-9), res_lat: torch.Tensor = torch.tensor(200e-9),
                         n_theta: int = 500,
                         torch_device: str = 'cpu',
                         complex_type: torch.dtype = torch.complex64,
                         normalize: bool = True,
                         linear_polarization: bool = False,
                         **kwargs) -> torch.Tensor:
    """
    Calculates the laser intensity I/I_0 around the laser focus.

    Args:
        lam (float, optional): wavelength in meters. Defaults to 780e-9m.
        n (float, optional): index of refraction. Defaults to 1.483.
        NA (float, optional): numerical aperture. Defaults to 0.8.
        M (float, optional): magnification. Defaults to 25.0.
        tubus (float, optional): tubus length in meters. Defaults to 165.0e-3m (for zeiss objectives).
        r_r (float, optional): Lateral size of the focus in meters. Defaults to 3.2e-6m.
        r_z (float, optional): Axial size of the focus in meters. Defaults to 4.8e-6m.
        res_ax (float, optional): axial resolution in meters. Defaults to 300e-9m.
        res_lat (float, optional): lateral resolution in meters. Defaults to 200e-9m.
        n_theta (int, optional): number of angles to integrate over. Defaults to 500.
        torch_device (str, optional): torch device. Defaults to 'cpu'.
        complex_type (torch.dtype, optional): complex type. Defaults to torch.complex64.
        linear_polarization (bool, optional): whether to use linear polarization. Defaults to True.
    Returns:
        torch.Tensor: laser intensity.
    """
    if 'device' in kwargs:
        torch_device = kwargs['device']
    if 'n_monomer' in kwargs:
        n = kwargs['n_monomer']
    fl = tubus / M  # Focal length of the objective lens
    w = NA * tubus / M / n

    k = 2 * torch.pi * n / lam
    theta_max = torch.arcsin(NA / n).to(torch_device)
    f_0 = w / (fl * NA / n)

    bessel_j2 = BesselJ.apply

    x = torch.linspace(-r_r, r_r, int(torch.round(2 * r_r / res_lat)) + 1, device=torch_device)
    y = torch.linspace(-r_r, r_r, int(torch.round(2 * r_r / res_lat)) + 1, device=torch_device)
    z = torch.linspace(-r_z, r_z, int(torch.round(2 * r_z / res_ax)) + 1, device=torch_device)

    theta = torch.linspace(0, theta_max, n_theta, device=torch_device)

    xx, yy = torch.meshgrid(x, y, indexing='ij')
    phi = torch.angle(torch.complex(xx, yy))
    dist = torch.sqrt(xx ** 2 + yy ** 2)

    # Integrals are based on Novotny-Hecht- 'Principles of Nano Optics' p. 62
    term_00_1 = torch.exp(-(torch.sin(theta) / (f_0 * torch.sin(theta_max))).pow(2)) * torch.cos(theta).pow(
        0.5) * torch.sin(theta) * (1 + torch.cos(theta))
    term_00_bessel = torch.special.bessel_j0(
        k * torch.tile(dist, (theta.shape[0], 1, 1)).permute(1, 2, 0) * torch.tile(torch.sin(theta),
                                                                                   (x.shape[0], y.shape[0], 1)))

    term_01_1 = torch.exp(-(torch.sin(theta) / (f_0 * torch.sin(theta_max))).pow(2)) * torch.cos(theta).pow(
        0.5) * torch.sin(theta) * torch.sin(theta)
    term_01_bessel = torch.special.bessel_j1(
        k * torch.tile(dist, (theta.shape[0], 1, 1)).permute(1, 2, 0) * torch.tile(torch.sin(theta),
                                                                                   (x.shape[0], y.shape[0], 1)))

    term_02_2 = torch.exp(-(torch.sin(theta) / (f_0 * torch.sin(theta_max))).pow(2)) * torch.cos(theta).pow(
        0.5) * torch.sin(theta) * (1 - torch.cos(theta))
    term_02_bessel = bessel_j2(
        k * torch.tile(dist, (theta.shape[0], 1, 1)).permute(1, 2, 0) * torch.tile(torch.sin(theta),
                                                                                   (x.shape[0], y.shape[0], 1)), 2)

    I00 = torch.zeros((x.shape[0], y.shape[0], z.shape[0]), device=torch_device, dtype=complex_type)
    I01 = torch.zeros((x.shape[0], y.shape[0], z.shape[0]), device=torch_device, dtype=complex_type)
    I02 = torch.zeros((x.shape[0], y.shape[0], z.shape[0]), device=torch_device, dtype=complex_type)

    for i, z in enumerate(z):
        I00[:, :, i] = torch.trapz(term_00_1 * term_00_bessel * torch.exp(1j * k * z * torch.cos(theta)),
                                   dx=theta_max / (n_theta - 1))
        I01[:, :, i] = torch.trapz(term_01_1 * term_01_bessel * torch.exp(1j * k * z * torch.cos(theta)),
                                   dx=theta_max / (n_theta - 1))
        I02[:, :, i] = torch.trapz(term_02_2 * term_02_bessel * torch.exp(1j * k * z * torch.cos(theta)),
                                   dx=theta_max / (n_theta - 1))

    if linear_polarization:
        e_x = I00 + I02 * torch.cos(2 * phi).unsqueeze(-1)
        e_y = I02 * torch.sin(2 * phi).unsqueeze(-1)
        e_z = -2 * 1j * I01 * torch.cos(phi).unsqueeze(-1)
    else:  # Circular polarization (right circular polarization assumed, but doesn't matter for the intensity)
        e_x = I00 + I02 * torch.exp(2j * phi).unsqueeze(-1)
        e_y = 1j * (I00 - I02 * torch.exp(2j * phi).unsqueeze(-1))
        e_z = -2 * 1j * I01 * torch.exp(1j * phi).unsqueeze(-1)
    intensity = (torch.abs(e_x).pow(2) + torch.abs(e_y).pow(2) + torch.abs(e_z).pow(2))  # / n * (k * fl / 2)**2

    if normalize:
        ex = intensity[:, intensity.shape[1] // 2, intensity.shape[2] // 2] / intensity.max()

        x = torch.linspace(-r_r, r_r, int(torch.round(2 * r_r / res_lat)) + 1, device=ex.device)
        abs_ex = torch.abs(ex - 1 / np.exp(2))
        idx = torch.argmin(abs_ex)
        r = (intensity.shape[1] // 2 - idx) * res_lat * 1e6  # radius in microns

        intensity = intensity / intensity[idx, intensity.shape[1] // 2, intensity.shape[2] // 2]

    return intensity


def calc_laser_intensity_OLD(lam: float = 780e-9, n: float = 1.483, NA: float = 0.8,
                             M: float = 25.0,
                             tubus: float = 165.0e-3, r_r: float = 3.2e-6, r_z: float = 4.8e-6,
                             res_ax: float = 300e-9, res_lat: float = 200e-9,
                             n_theta: int = 500,
                             torch_device: str = 'cpu',
                             complex_type: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    Calculates the laser intensity I/I_0 around the laser focus.

    Args:
        lam (float, optional): wavelength in meters. Defaults to 780e-9m.
        n (float, optional): index of refraction. Defaults to 1.483.
        NA (float, optional): numerical aperture. Defaults to 0.8.
        M (float, optional): magnification. Defaults to 25.0.
        tubus (float, optional): tubus length in meters. Defaults to 165.0e-3m (for zeiss objectives).
        r_r (float, optional): Lateral size of the focus in meters. Defaults to 3.2e-6m.
        r_z (float, optional): Axial size of the focus in meters. Defaults to 4.8e-6m.
        res_ax (float, optional): axial resolution in meters. Defaults to 300e-9m.
        res_lat (float, optional): lateral resolution in meters. Defaults to 200e-9m.
        n_theta (int, optional): number of angles to integrate over. Defaults to 500.
        torch_device (str, optional): torch device. Defaults to 'cpu'.
        complex_type (torch.dtype, optional): complex type. Defaults to torch.complex64.

    Returns:
        torch.Tensor: laser intensity.
    """
    w = NA * tubus / M
    fl = tubus / M
    k = 2 * torch.pi * n / lam
    theta_max = torch.arcsin(torch.tensor(NA / n, device=torch_device))
    f_0 = w / (fl * NA / n)

    bessel_j2 = BesselJ.apply

    x = torch.arange(-r_r, r_r + res_lat, res_lat, device=torch_device)
    y = torch.arange(-r_r, r_r + res_lat, res_lat, device=torch_device)
    z = torch.arange(-r_z, r_z + res_ax, res_ax, device=torch_device)

    theta = torch.linspace(0, theta_max, n_theta, device=torch_device)

    xx, yy = torch.meshgrid(x, y, indexing='ij')
    phi = torch.angle(torch.complex(xx, yy))
    dist = torch.sqrt(xx ** 2 + yy ** 2)

    # Integrals are based on Novotny-Hecht- 'Principles of Nano Optics' p. 63
    term_00_1 = torch.exp(-(torch.sin(theta) / (f_0 * torch.sin(theta_max))).pow(2)) * torch.cos(theta).pow(
        0.5) * torch.sin(theta) * (1 + torch.cos(theta))
    term_00_bessel = torch.special.bessel_j0(
        k * torch.tile(dist, (theta.shape[0], 1, 1)).permute(1, 2, 0) * torch.tile(torch.sin(theta),
                                                                                   (x.shape[0], y.shape[0], 1)))

    term_01_1 = torch.exp(-(torch.sin(theta) / (f_0 * torch.sin(theta_max))).pow(2)) * torch.cos(theta).pow(
        0.5) * torch.sin(theta) * torch.sin(theta)
    term_01_bessel = torch.special.bessel_j1(
        k * torch.tile(dist, (theta.shape[0], 1, 1)).permute(1, 2, 0) * torch.tile(torch.sin(theta),
                                                                                   (x.shape[0], y.shape[0], 1)))

    term_02_2 = torch.exp(-(torch.sin(theta) / (f_0 * torch.sin(theta_max))).pow(2)) * torch.cos(theta).pow(
        0.5) * torch.sin(theta) * (1 - torch.cos(theta))
    term_02_bessel = bessel_j2(
        k * torch.tile(dist, (theta.shape[0], 1, 1)).permute(1, 2, 0) * torch.tile(torch.sin(theta),
                                                                                   (x.shape[0], y.shape[0], 1)), 2)

    I00 = torch.zeros((x.shape[0], y.shape[0], z.shape[0]), device=torch_device, dtype=complex_type)
    I01 = torch.zeros((x.shape[0], y.shape[0], z.shape[0]), device=torch_device, dtype=complex_type)
    I02 = torch.zeros((x.shape[0], y.shape[0], z.shape[0]), device=torch_device, dtype=complex_type)

    for i, z in enumerate(z):
        I00[:, :, i] = torch.trapz(term_00_1 * term_00_bessel * torch.exp(1j * k * z * torch.cos(theta)),
                                   dx=theta_max / (n_theta-1))
        I01[:, :, i] = torch.trapz(term_01_1 * term_01_bessel * torch.exp(1j * k * z * torch.cos(theta)),
                                   dx=theta_max / (n_theta-1))
        I02[:, :, i] = torch.trapz(term_02_2 * term_02_bessel * torch.exp(1j * k * z * torch.cos(theta)),
                                   dx=theta_max / (n_theta-1))

    e_x = I00 + I02 * torch.cos(2 * phi).unsqueeze(-1)
    e_y = I02 * torch.sin(2 * phi).unsqueeze(-1)
    e_z = -2 * 1j * I01 * torch.cos(phi).unsqueeze(-1)
    intensity = (torch.abs(e_x).pow(2) + torch.abs(e_y).pow(2) + torch.abs(e_z).pow(2))  # / n * (k * fl / 2)**2

    return intensity


def create_gaussian_intensity(r_r: torch.Tensor = torch.tensor(3.2e-6, device='cuda'),
                              r_z: torch.Tensor = torch.tensor(9.6e-6, device='cuda'),
                              res_lat: torch.Tensor = torch.tensor(0.2e-6, device='cuda'),
                              res_ax: torch.Tensor = torch.tensor(0.3e-6, device='cuda'),
                              lam: torch.Tensor = torch.tensor(780e-9, device='cuda'),
                              NA: torch.Tensor = torch.tensor(0.8, device='cuda'),
                              n: torch.Tensor = torch.tensor(1.483, device='cuda'),
                              device: str = 'cuda',
                              dtype=torch.float) -> torch.Tensor:
    if device != 'cuda':
        r_r = r_r.to(device=device)
        r_z = r_z.to(device=device)
        res_lat = res_lat.to(device=device)
        res_ax = res_ax.to(device=device)
        lam = lam.to(device=device)
        NA = NA.to(device=device)
        n = n.to(device=device)
    x = torch.linspace(-r_r, r_r, torch.round(2 * r_r / res_lat).to(torch.int) + 1, device=device, dtype=dtype)
    y = torch.linspace(-r_r, r_r, torch.round(2 * r_r / res_lat).to(torch.int) + 1, device=device, dtype=dtype)
    z = torch.linspace(-r_z, r_z, torch.round(2 * r_z / res_ax).to(torch.int) + 1, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    r = torch.tile(torch.sqrt(X ** 2 + Y ** 2).unsqueeze(-1), (1, 1, z.shape[0]))

    w0 = lam / (np.pi * NA) * torch.sqrt(n ** 2 - NA ** 2)  # Beam waist radius for high NA objective lens
    w_z = w0 * torch.sqrt(1 + (z / (torch.pi * w0 ** 2 * n / lam)) ** 2)[None, None, :]

    intensity = (w0 / w_z) ** 2 * torch.exp(- r ** 2 / (2 * w_z ** 2))

    return intensity


def msbpm_torch(ri_tensor: torch.Tensor, n_media: torch.Tensor = torch.tensor(1.483),
                psx: torch.Tensor = torch.tensor(0.4), psz: torch.Tensor = torch.tensor(0.3),
                lambda_val: torch.Tensor = torch.tensor(0.63), NA: torch.Tensor = torch.tensor(0.8),
                z_shift: torch.Tensor = torch.tensor(-24),
                dtype_c: torch.dtype = torch.complex64,
                print_until_layer: int = -1
                ) -> torch.Tensor:
    """
    Can calculate a batch of MSBPM images for a given tensor containing the refractive indices.

    Args:
        ri_tensor (torch.Tensor): Tensor of refractive indices. Shape (n_samples, n_x, n_y, n_z).
        n_media (torch.Tensor, optional): Refractive index of the medium. Defaults to 1.483 for IP-S.
        psx (torch.Tensor, optional): Pixel size in the x direction. Defaults to 0.4um.
        psz (torch.Tensor, optional): Pixel size in the z direction. Defaults to 0.3um.
        lambda_val (torch.Tensor, optional): Wavelength. Defaults to 0.63um.
        NA (torch.Tensor, optional): Numerical aperture. Defaults to 0.8.
        z_shift (torch.Tensor, optional): z shift. Defaults to -24um.
        dtype_c (torch.dtype, optional): Complex data type. Defaults to torch.complex64. For NA=1.4 you might need torch.complex128
        print_until_layer (int, optional): Number of layers to print. Defaults to -1. which means, that the field is propagated through the entire stack.
            By setting print_until_layer = 20, the image after the 20th layer will be returned.

    returns:
        torch.Tensor: A tensor of MSBPM images. Shape (n_samples, intensity_x, intensity_y).
    """

    device = ri_tensor.device
    n_samples = ri_tensor.shape[0]
    n_x, n_y, n_z = ri_tensor.shape[1:]
    if print_until_layer < 0:
        print_until_layer = n_z
    dfx = 1 / (psx * n_x)
    dfy = 1 / (psx * n_y)

    x = torch.arange(-n_x / 2, n_x / 2, device=device) * psx
    y = torch.arange(-n_y / 2, n_y / 2, device=device) * psx
    xx, yy = torch.meshgrid(y, x, indexing='ij')

    fx = (dfx * torch.arange(-n_x / 2, n_x / 2, device=device)).to(dtype_c)
    fy = (dfy * torch.arange(-n_y / 2, n_y / 2, device=device)).to(dtype_c)

    fxx, fyy = torch.meshgrid(fy, fx, indexing='ij')
    fxx = torch.fft.ifftshift(fxx)
    fyy = torch.fft.ifftshift(fyy)

    prop = 1j * 2 * torch.pi * torch.sqrt((n_media / lambda_val) ** 2 - (fxx ** 2 + fyy ** 2))
    prop_z = torch.exp(prop * psz).unsqueeze(0)

    NA_crop = torch.real(fxx ** 2 + fyy ** 2) < torch.real((NA / lambda_val) ** 2)

    # Initialize the incident field as a plane wave with normal incidence.
    u_in = torch.tile(torch.exp(1j * 2 * np.pi * (0 * xx.to(dtype_c) + 0 * yy.to(dtype_c))), (n_samples, 1, 1))
    fu_current = torch.fft.fft2(u_in)

    # Transmission function
    t_ri = torch.exp(1j * 2 * np.pi * ri_tensor.to(dtype_c) * psz / lambda_val)

    for i_layer in range(print_until_layer):
        field = torch.fft.ifft2(fu_current * prop_z)
        fu_current = torch.fft.fft2(field * t_ri[:, :, :, i_layer])

    prop_kernel = torch.exp(prop.unsqueeze(0) * (z_shift / 2).to(dtype_c)) * NA_crop.to(torch.float)
    field = torch.fft.ifft2(fu_current * prop_kernel)
    fu_current = torch.fft.fft2(field)
    # prop_kernel = prop_kernel
    intensity = torch.abs(torch.fft.ifft2(fu_current * prop_kernel)) ** 2

    return intensity.to(ri_tensor.dtype)


def wasserstein_distance(img1, img2):
    # Ensure images are normalized
    img1 = img1.view(img1.size(0), -1)
    img2 = img2.view(img2.size(0), -1)

    # Sort the flattened tensors
    img1_sorted, _ = torch.sort(img1.ravel())
    img2_sorted, _ = torch.sort(img2.ravel())

    # Compute the L1 distance between sorted tensors
    return torch.sum(torch.abs(img1_sorted - img2_sorted))


def wasserstein_distance_vectorized(img1: torch.Tensor, img2: torch.Tensor):
    # Flatten the images (batch_size, 1, npx, npy) -> (batch_size, npx * npy)
    img1_flat = img1.view(img1.size(0), -1)
    img2_flat = img2.view(img2.size(0), -1)

    # Sort the flattened images along the last dimension (batch-wise sorting)
    img1_sorted, _ = torch.sort(img1_flat, dim=1)
    img2_sorted, _ = torch.sort(img2_flat, dim=1)

    # Compute the element-wise L1 distance (Wasserstein distance approximation)
    wasserstein_dist = torch.mean(torch.abs(img1_sorted - img2_sorted), dim=1)

    return wasserstein_dist


def calculate_contrast(int_sim: np.ndarray):
    """
    Calculate the contrast for each object individually of the simulated images using the standard deviation.

    Args:
        int_sim (np.ndarray): The simulated images.

    Returns:
        np.ndarray: The contrast of the simulated images.

    Example:
        i, o = forward_model.training_adapter_bin_obj(obj_all, substrate=torch.tensor(10), lp=torch.tensor([[0.02]]))
        contrast = claculate_contrast(i)
    """
    if type(int_sim) == torch.Tensor:
        int_sim = int_sim.detach().cpu().numpy()
    if len(int_sim.shape) == 3:
        int_sim = int_sim[None]
    elif len(int_sim.shape) == 5:
        int_sim = int_sim[:, 0]

    return np.array(
        [[np.std(int_sim[j, :, :, i]) for i in range(int_sim[j].squeeze().shape[-1])] for j in range(int_sim.shape[0])])


def detect_substrate(obj: np.ndarray, fm: callable, contrast_exp: np.ndarray,
                     try_substrates: np.ndarray = np.arange(0, 20, 1),
                     forward_function_args: dict[str, Any] = {}, logger_path: logging.Logger = None, n_diff: int = 0) -> \
tuple[np.ndarray, np.ndarray]:
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    substrates = np.zeros((obj.shape[0]))
    contrast_sim = np.zeros((obj.shape[0], try_substrates.shape[0], contrast_exp.shape[-1]))
    if logger_path is not None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(logger_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Cuda visible devices: " + str(torch.cuda.device_count()))

        cuda_key = False
        for key, value in os.environ.items():
            # print(f"{key}: {value}")
            if 'CUDA_VISIBLE_DEVICES' in key:
                logger.info(f"{key}: {value}")
                cuda_key = True

        if cuda_key == False:
            logger.info('CUDA_VISIBLE_DEVICES not set')

    for i, s in enumerate(try_substrates):
        fm_kwargs = {'substrate': torch.tensor(s), **forward_function_args}
        int_sim, obj_sim = fm.training_adapter_bin_obj(obj, **fm_kwargs)
        int_sim += np.random.normal(0, contrast_exp[:, 0][:, None, None, None, None], int_sim.shape)
        int_sim = np.diff(int_sim, n=n_diff, axis=-1)
        # int_sim = int_sim[:, :, :, :, :int_sim.shape[-1]-fm.n_diff]
        int_sim = int_sim[:, :, :, :, :contrast_exp.shape[-1]]
        contrast_sim[:, i, :] = calculate_contrast(int_sim)
    contrast_mean_abs_diff = np.abs(contrast_sim - contrast_exp[:, None, :]).mean(axis=-1)
    if logger_path is not None:
        print('LOGGER')
        logger.info(
            f"forward_function_args: {forward_function_args}, rho_0: {fm.rho_0}, sig_2_r_exp: {fm.sig_2_r_exp}, sig_2_r_base: {fm.sig_2_r_base}, substrate: {try_substrates[np.argmin(contrast_mean_abs_diff, axis=1)]}, contrast: {contrast_mean_abs_diff.min(axis=1)}")

    return try_substrates[np.argmin(contrast_mean_abs_diff, axis=1)], contrast_mean_abs_diff.min(axis=1)


def create_3d_psf(x_fwhm, y_fwhm, z_fwhm,
                  rotation_xy=0, rotation_xz=0, rotation_yz=0,
                  astigmatism_xy=0,
                  size_lat=32,
                  size_ax=32,
                  voxel_size=(1.0, 1.0, 1.0)):
    """
    Generate a 3D Gaussian PSF with real-world voxel size.
    The parameters of the focus are infered from Vincent Hahns script.

    Example:
    Our 63x zeiss objective has in the current setup the following parameters:
    x_fwhm = .415995 #um
    y_fwhm =.354342 #um
    z_fwhm = .963721 #um
    rot_xy = 24.7
    rot_xz = 10.5
    rot_yz = 6.3
    astigmatism_xy = .142847
    voxel_size=(0.025, 0.025, 0.05)
    size = 61

    psf = create_3d_psf(x_fwhm=x_fwhm, y_fwhm=y_fwhm, z_fwhm=z_fwhm,
                    rotation_xy=rot_xy, rotation_xz=rot_xz, rotation_yz=rot_yz,
                    astigmatism_xy=astigmatism_xy,
                    voxel_size=(0.025, 0.025, 0.05),
                    size = 61)

    Parameters:
        x_fwhm, y_fwhm, z_fwhm : float
            FWHM in micrometers.
        rotation_xy, rotation_xz, rotation_yz : float
            Rotation angles in degrees.
        astigmatism_xy : float
            Difference in FWHM between x and y.
        size : int
            Number of pixels in each direction (cubic volume).
        voxel_size : tuple of 3 floats
            Physical size of a voxel in µm (x, y, z).

    Returns:
        psf : np.ndarray
            3D PSF volume normalized to max 1.
    """
    # Convert FWHM to standard deviation in µm
    fwhm_to_sigma = lambda fwhm: fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma_x = fwhm_to_sigma(x_fwhm)
    sigma_y = fwhm_to_sigma(y_fwhm + astigmatism_xy)
    sigma_z = fwhm_to_sigma(z_fwhm)

    # Create real-world coordinate grid (µm)
    vx, vy, vz = voxel_size
    x = np.linspace(-size_lat // 2 * vx, size_lat // 2 * vx, size_lat)
    y = np.linspace(-size_lat // 2 * vy, size_lat // 2 * vy, size_lat)
    z = np.linspace(-size_ax // 2 * vz, size_ax // 2 * vz, size_ax)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Multivariate Gaussian
    mean = [0, 0, 0]
    cov = [[sigma_x ** 2, 0, 0],
           [0, sigma_y ** 2, 0],
           [0, 0, sigma_z ** 2]]
    pos = np.stack((xx, yy, zz), axis=-1)
    gaussian = multivariate_normal(mean, cov).pdf(pos)

    # Apply rotations
    gaussian = rotate(gaussian, angle=rotation_xy, axes=(0, 1), reshape=False, order=1)
    gaussian = rotate(gaussian, angle=rotation_xz, axes=(0, 2), reshape=False, order=1)
    gaussian = rotate(gaussian, angle=rotation_yz, axes=(1, 2), reshape=False, order=1)

    # Normalize
    gaussian /= np.max(gaussian)

    return gaussian


def plot_crossection(psf, voxel_size=(0.025, 0.025, 0.05), savepath=None):
    """
    For plotting the crosssections of the PSF in a simmilar manner to Vincent Hahns script.
    Example:
    psf = create_3d_psf(x_fwhm=x_fwhm, y_fwhm=y_fwhm, z_fwhm=z_fwhm,
                    rotation_xy=rot_xy, rotation_xz=rot_xz, rotation_yz=rot_yz,
                    astigmatism_xy=astigmatism_xy,
                    voxel_size=(0.025, 0.025, 0.05),
                    size = 61)
    plot_crossection(psf, voxel_size=(0.025, 0.025, 0.05), savepath=None)

    Parameters:
        psf : np.ndarray
            3D PSF volume.
        voxel_size : tuple of 3 floats
            Physical size of a voxel in µm (x, y, z).
        savepath : str or None
            Path to save the figure. If None, the figure will not be saved.

    Returns:
        None
    """
    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(15, 5))
    ax_xy.imshow(psf[:, :, psf.shape[-1] // 2], cmap='gray')
    ax_xz.imshow(np.rot90(psf[:, psf.shape[1] // 2, :]), cmap='gray')
    ax_yz.imshow(np.rot90(psf[psf.shape[0] // 2, :, :]), cmap='gray')
    ax_xy.set_title('XY Plane')
    ax_xz.set_title('XZ Plane')
    ax_yz.set_title('YZ Plane')
    # Set ticks to real-world coordinates and such that they are centered
    x_ticks = np.arange(0, psf.shape[0], 10)
    y_ticks = np.arange(0, psf.shape[1], 10)
    ax_xy.set_xticks(x_ticks)
    ax_xy.set_xticklabels((x_ticks - psf.shape[0] // 2) * voxel_size[0])
    ax_xy.set_yticks(y_ticks)
    ax_xy.set_yticklabels((y_ticks - psf.shape[1] // 2) * voxel_size[1])
    ax_xz.set_xticks(x_ticks)
    ax_xz.set_xticklabels((x_ticks - psf.shape[0] // 2) * voxel_size[0])
    ax_xz.set_yticks(y_ticks)
    ax_xz.set_yticklabels((y_ticks - psf.shape[1] // 2) * voxel_size[2])
    ax_yz.set_xticks(x_ticks)
    ax_yz.set_xticklabels((x_ticks - psf.shape[0] // 2) * voxel_size[1])
    ax_yz.set_yticks(y_ticks)
    ax_yz.set_yticklabels((y_ticks - psf.shape[1] // 2) * voxel_size[2])

    # Set labels
    ax_xy.set_xlabel('X (µm)')
    ax_xy.set_ylabel('Y (µm)')
    ax_xz.set_xlabel('X (µm)')
    ax_xz.set_ylabel('Z (µm)')
    ax_yz.set_xlabel('Y (µm)')
    ax_yz.set_ylabel('Z (µm)')

    # Set aspect ratio

    ax_xz.set_aspect(voxel_size[2] / voxel_size[1])
    ax_yz.set_aspect(voxel_size[2] / voxel_size[1])
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()


def create_3d_psf_torch(x_fwhm: torch.Tensor,
                        y_fwhm: torch.Tensor,
                        z_fwhm: torch.Tensor,
                        rotation_xy: float = 0,
                        rotation_xz: float = 0,
                        rotation_yz: float = 0,
                        astigmatism_xy: float = 0,
                        res_lat: torch.Tensor = torch.tensor(0.025, device=DEVICE),
                        res_ax: torch.Tensor = torch.tensor(0.05, device=DEVICE),
                        size_lat: int = 61,
                        size_ax: int = 61,
                        device: str = DEVICE,
                        dtype=torch.float,
                        n: torch.Tensor = torch.nan,
                        NA: torch.Tensor = torch.nan) -> torch.Tensor:
    """
    This function creates a 3D Gaussian Point Spread Function (PSF) using PyTorch from Vincent's fitted measurement.
    Arguments:
    x_fwhm: Full width at half maximum (FWHM) in the x-direction (μm).
    y_fwhm: FWHM in the y-direction (μm).
    z_fwhm: FWHM in the z-direction (μm).
    rotation_xy: Rotation angle in the xy-plane (degrees).
    rotation_xz: Rotation angle in the xz-plane (degrees).
    rotation_yz: Rotation angle in the yz-plane (degrees).
    astigmatism_xy: Astigmatism in the xy-plane (μm).
    res_lat: Lateral resolution (μm/voxel).
    res_ax: Axial resolution (μm/voxel).
    size_lat: Size of the PSF in the lateral direction (number of voxels).
    size_ax: Size of the PSF in the axial direction (number of voxels).
    device: Device to perform calculations on (default is 'cuda' if available, otherwise 'cpu').
    dtype: Data type for calculations (default is torch.float).
    n: Dummy argument to make it compatible with the simulation.
    NA: Dummy argument to make it compatible with the simulation.
    Returns:
    A 3D tensor representing the PSF.
    """
    rotation_xy = rotation_xy + 180  # to match the original function
    # Move inputs to device if not already
    x_fwhm = x_fwhm.to(device=device, dtype=dtype)
    y_fwhm = (y_fwhm + astigmatism_xy).to(device=device, dtype=dtype)
    z_fwhm = z_fwhm.to(device=device, dtype=dtype)
    res_lat = res_lat.to(device=device, dtype=dtype)
    res_ax = res_ax.to(device=device, dtype=dtype)

    def fwhm_to_sigma(fwhm):
        return fwhm / (2 * torch.sqrt(2 * torch.log(torch.tensor(2., dtype=dtype, device=device))))

    sigma_x = fwhm_to_sigma(x_fwhm)
    sigma_y = fwhm_to_sigma(y_fwhm)
    sigma_z = fwhm_to_sigma(z_fwhm)

    # Coordinate grid in real-world units (μm)
    x = torch.linspace(-size_lat // 2 * res_lat, size_lat // 2 * res_lat, size_lat, device=device, dtype=dtype)
    y = torch.linspace(-size_lat // 2 * res_lat, size_lat // 2 * res_lat, size_lat, device=device, dtype=dtype)
    z = torch.linspace(-size_ax // 2 * res_ax, size_ax // 2 * res_ax, size_ax, device=device, dtype=dtype)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    coords = torch.stack([X, Y, Z], dim=-1)

    # Rotation matrices
    def get_rotation_matrix(angle_deg, axis1, axis2):
        angle = torch.deg2rad(torch.tensor(angle_deg, dtype=dtype, device=device))
        mat = torch.eye(3, device=device, dtype=dtype)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        mat[axis1, axis1] = cos_a
        mat[axis2, axis2] = cos_a
        mat[axis1, axis2] = -sin_a
        mat[axis2, axis1] = sin_a
        return mat

    rot_xy = get_rotation_matrix(rotation_xy, 0, 1)
    rot_xz = get_rotation_matrix(rotation_xz, 0, 2)
    rot_yz = get_rotation_matrix(rotation_yz, 1, 2)
    rotation = rot_yz @ rot_xz @ rot_xy

    coords_rot = coords @ rotation.T

    Xr, Yr, Zr = coords_rot[..., 0], coords_rot[..., 1], coords_rot[..., 2]

    # Gaussian formula
    gauss = torch.exp(-0.5 * ((Xr / sigma_x) ** 2 + (Yr / sigma_y) ** 2 + (Zr / sigma_z) ** 2))

    # Normalize to max 1
    gauss /= gauss.max()

    gauss = gauss.to(device=device, dtype=dtype)

    return gauss


def linear_contrast_adaption(intensity: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Applies linear contrast adaption to the intensity image.

    Args:
        intensity (np.ndarray): The input intensity image.
        alpha (float): The factor by which to scale the contrast.

    Returns:
        np.ndarray: The contrast-adapted intensity image.
    """
    return alpha * (intensity - intensity.mean()) + intensity.mean()


class GeneralizedLogisticODE(torch.nn.Module):
    """
    Defines the ODE: f'(t) = k * f(t)^m * (1 - f(t))^n
    Handles tensor parameters k, m, n that can vary spatially
    """

    def __init__(self, m: float = 0, n: float = 1, **kwargs):
        super().__init__()
        # Store parameters as buffers to ensure they stay on the same device
        # self.register_buffer('k', torch.as_tensor(k))
        self.register_buffer('m', torch.as_tensor(m))
        self.register_buffer('n', torch.as_tensor(n))

    def forward(self, t, f, k):
        """
        Args:
            t: time (scalar)
            f: function values at time t (tensor of shape matching initial conditions)
        Returns:
            df/dt: derivative (same shape as f)
        """
        # Ensure f is in valid range [0, 1] to avoid numerical issues
        f = torch.clamp(f, min=1e-8, max=1 - 1e-8)

        # Compute the derivative - broadcasting handles different tensor shapes
        # dfdt = self.k * torch.pow(f, self.m) * torch.pow(1 - f, self.n)
        dfdt = k * torch.pow(f, self.m) * torch.pow(1 - f, self.n)

        return dfdt



